"""Three-way engine comparison for the pipeline's hot operators.

    python -m newsrec.bench --dataset ebnerd --variant small --engines polars,gpu,cudf

Backends
  polars : CPU, streaming executor
  gpu    : the identical polars LazyFrame run through cudf-polars, with
           raise_on_fail=True so a silent CPU fallback is reported as a failure
           rather than logged as a fast GPU time
  cudf   : hand-written native cuDF, i.e. the ceiling if we abandoned one codebase

Operators are the ones that actually dominate the build: list explodes, the id
remap join, MIND's string parsing, and the per-impression rank that produces a
submission file (200M rows at large scale).
"""
from __future__ import annotations

import argparse
import gc
import json
import resource
import time
from pathlib import Path

import polars as pl

from .config import DATA_ROOT


def _peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def _gpu_mem_mb() -> float:
    try:
        import rmm  # noqa: F401
        from pynvml import (nvmlInit, nvmlDeviceGetHandleByIndex,
                            nvmlDeviceGetMemoryInfo)
        nvmlInit()
        return nvmlDeviceGetMemoryInfo(nvmlDeviceGetHandleByIndex(0)).used / 1e6
    except Exception:
        return float("nan")


# --------------------------------------------------------------- operators
# Each returns a LazyFrame; the polars and gpu backends share them verbatim.

def op_explode_groupby(paths: dict) -> pl.LazyFrame:
    """Popularity: explode the in-view list and count exposures per article."""
    return (
        pl.scan_parquet(paths["impressions"])
        .select("candidates")
        .explode("candidates")
        .group_by("candidates")
        .agg(pl.len().alias("n"))
        .sort("n", descending=True)
        .head(20)
    )


def op_remap_join(paths: dict) -> pl.LazyFrame:
    """The id remap: explode, join the id map, regroup preserving element order."""
    id_map = pl.scan_parquet(paths["articles"]).select("src_id").with_row_index("idx")
    return (
        pl.scan_parquet(paths["impressions"])
        .select("impression_id", "candidates")
        .with_columns(pl.int_ranges(0, pl.col("candidates").list.len()).alias("_pos"))
        .explode(["candidates", "_pos"])
        .join(id_map.with_columns(pl.col("idx").cast(pl.UInt32)),
              left_on="candidates", right_on="idx", how="inner")
        .group_by("impression_id")
        .agg(pl.col("src_id").sort_by("_pos"))
    )


def op_rank_submission(paths: dict) -> pl.LazyFrame:
    """Submission generation: score every candidate, rank within impression."""
    pop = pl.scan_parquet(paths["popularity"]).select("article_idx", "clicks")
    return (
        pl.scan_parquet(paths["impressions"])
        .select("impression_id", "candidates")
        .with_columns(pl.int_ranges(0, pl.col("candidates").list.len()).alias("_pos"))
        .explode(["candidates", "_pos"])
        .join(pop, left_on="candidates", right_on="article_idx", how="left")
        .with_columns(pl.col("clicks").fill_null(0))
        .with_columns(pl.col("clicks").rank("ordinal", descending=True)
                        .over("impression_id").alias("rank"))
        .group_by("impression_id")
        .agg(pl.col("rank").sort_by("_pos"))
    )


def op_history_agg(paths: dict) -> pl.LazyFrame:
    """User profile: explode history and aggregate per user."""
    return (
        pl.scan_parquet(paths["history"])
        .select("user_idx", "article_idx")
        .explode("article_idx")
        .group_by("user_idx")
        .agg(pl.len().alias("n"), pl.col("article_idx").n_unique().alias("uniq"))
        .sort("n", descending=True)
        .head(20)
    )


OPS = {
    "explode_groupby": op_explode_groupby,
    "remap_join": op_remap_join,
    "rank_submission": op_rank_submission,
    "history_agg": op_history_agg,
}


# ------------------------------------------------------------ native cuDF
# Deliberately separate: this is the "second dialect" cost the design note weighs.

def cudf_explode_groupby(paths: dict):
    import cudf
    df = cudf.read_parquet(paths["impressions"], columns=["candidates"])
    return df["candidates"].explode().value_counts().head(20)


def cudf_remap_join(paths: dict):
    import cudf
    arts = cudf.read_parquet(paths["articles"], columns=["src_id"]).reset_index(names="idx")
    df = cudf.read_parquet(paths["impressions"], columns=["impression_id", "candidates"])
    ex = df.explode("candidates")
    ex["_pos"] = ex.groupby("impression_id").cumcount()
    m = ex.merge(arts, left_on="candidates", right_on="idx", how="inner")
    return m.sort_values(["impression_id", "_pos"]).groupby("impression_id").agg({"src_id": "collect"})


def cudf_rank_submission(paths: dict):
    import cudf
    pop = cudf.read_parquet(paths["popularity"], columns=["article_idx", "clicks"])
    df = cudf.read_parquet(paths["impressions"], columns=["impression_id", "candidates"])
    ex = df.explode("candidates")
    ex["_pos"] = ex.groupby("impression_id").cumcount()
    m = ex.merge(pop, left_on="candidates", right_on="article_idx", how="left")
    m["clicks"] = m["clicks"].fillna(0)
    m["rank"] = m.groupby("impression_id")["clicks"].rank(method="first", ascending=False)
    return m.sort_values(["impression_id", "_pos"]).groupby("impression_id").agg({"rank": "collect"})


def cudf_history_agg(paths: dict):
    import cudf
    df = cudf.read_parquet(paths["history"], columns=["user_idx", "article_idx"])
    ex = df.explode("article_idx")
    return ex.groupby("user_idx").agg({"article_idx": ["count", "nunique"]}).head(20)


CUDF_OPS = {
    "explode_groupby": cudf_explode_groupby,
    "remap_join": cudf_remap_join,
    "rank_submission": cudf_rank_submission,
    "history_agg": cudf_history_agg,
}


# ------------------------------------------------------------------ driver

def run_one(op: str, backend: str, paths: dict) -> dict:
    gc.collect()
    rss0, gpu0 = _peak_rss_mb(), _gpu_mem_mb()
    t0 = time.perf_counter()
    try:
        if backend == "cudf":
            out = CUDF_OPS[op](paths)
            rows = len(out)
        else:
            lf = OPS[op](paths)
            if backend == "gpu":
                # raise_on_fail: a fallback must show up as a failure, not a fast time
                df = lf.collect(engine=pl.GPUEngine(raise_on_fail=True))
            else:
                df = lf.collect(engine="streaming")
            rows = df.height
        dt = time.perf_counter() - t0
        return {"seconds": round(dt, 3), "rows": rows,
                "peak_rss_mb": round(_peak_rss_mb(), 1),
                "gpu_mb": round(_gpu_mem_mb() - gpu0, 1) if gpu0 == gpu0 else None}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {str(e)[:200]}"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--engines", default="polars")
    ap.add_argument("--ops", default=",".join(OPS))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    root = DATA_ROOT / "processed" / a.dataset / a.variant
    paths = {
        "impressions": str(root / "splits" / a.split / "impressions.parquet"),
        "history": str(root / "splits" / a.split / "history.parquet"),
        "popularity": str(root / "splits" / a.split / "popularity.parquet"),
        "articles": str(root / "articles.parquet"),
    }
    n = pl.scan_parquet(paths["impressions"]).select(pl.len()).collect().item()
    print(f"== {a.dataset}/{a.variant} split={a.split}  ({n:,} impressions)")

    results = {}
    for op in a.ops.split(","):
        results[op] = {}
        for backend in a.engines.split(","):
            r = run_one(op, backend, paths)
            results[op][backend] = r
            if "error" in r:
                print(f"  {op:18s} {backend:7s} FAILED  {r['error']}")
            else:
                print(f"  {op:18s} {backend:7s} {r['seconds']:8.3f}s  {r['rows']:>10,} rows  "
                      f"rss {r['peak_rss_mb']:7.0f} MB" +
                      (f"  gpu {r['gpu_mb']:+.0f} MB" if r.get("gpu_mb") is not None else ""))
    if a.out:
        Path(a.out).write_text(json.dumps(
            {"dataset": a.dataset, "variant": a.variant, "split": a.split,
             "n_impressions": n, "results": results}, indent=2))
        print(f"== wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
