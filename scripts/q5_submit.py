"""Q5 / Q7.1: train at Codabench scale and write both leaderboard submissions.

Three things make this different from the dev-scale runs, and each is a place a
submission gets silently rejected rather than scored badly.

**1. The submit split has no labels, by construction.** So the model that ships
here is the `shipped` family set -- everything computable on a split with no
clicks in it. The rolling *exposure* counter survives (candidate lists are
published, so an article's impression count is observable); the rolling *click*
counter does not. That is exactly the `shipped` / `production` split
`newsrec.rerank` draws, and this script is why it exists: a model leaning on
`roll_clicks` would train beautifully and then meet a column of zeros.

**2. One line per raw behaviours row, in the raw file's order.** Not per
impression id -- EB-NeRD's test split reuses `impression_id = 0` for all 200,000
beyond-accuracy rows, so grouping by id collapses that whole track into one line.
Both graders walk their ground-truth file line by line and assert that line *i*
carries the same impression id, so a file built from a re-sorted store is
rejected on line 1. Scores are therefore re-attached by `src_row`, which indexes
the raw file and is asserted by `tests/test_row_order.py`, and the candidate
count per row is checked against the raw file before anything is written.

**3. It does not fit in memory in one piece.** EB-NeRD's submit split is 13.5M
impressions over ~150M candidate slots. Feature assembly is linear in rows, so
the split is processed in chunks of impressions and only the per-slot score is
kept -- 4 bytes per candidate rather than the 128 bytes its feature row costs.

Training subsamples impressions (never candidates within one) because a GBDT
over 32 features converges long before 12M impressions; the sweep that matters
at large scale is whether the *configuration* transfers, which
`--train-impressions` makes cheap to re-check. A1 measured the BM25 `b` optimum
reversing between small and large, so that check is not theatre.

Writes reports/sub/<dataset>_rerank.{txt,zip} and reports/sub/submit_<dataset>.json.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import polars as pl

from newsrec.behaviour import assemble
from newsrec.config import DATA_ROOT
from newsrec.rerank import (SHIPPED, add_matching_features, feature_names,
                            importances, predict, train)
from newsrec.store import FeatureStore
from newsrec.submit import ARCNAME, _rank_expr, raw_behaviours, write_submission, zip_submission


def build(fs, split, emb, k1, b, cache: Path, max_impressions=0, rebuild=False,
          labelled=True) -> pl.DataFrame:
    if cache.exists() and not rebuild:
        df = pl.read_parquet(cache)
        print(f"   [{split}] {df.height:,} rows from cache")
        return df
    t0 = time.perf_counter()
    df = assemble(fs, split, max_impressions=max_impressions, labelled=labelled)
    df = add_matching_features(fs, split, df, emb, k1, b)
    df = df.sort("imp", "position")
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(cache)
    print(f"   [{split}] {df.height:,} rows over {df['imp'].n_unique():,} impressions "
          f"in {time.perf_counter() - t0:.0f}s")
    return df


def score_submit_chunked(fs, model, feats, emb, k1, b, chunk: int,
                         out_dir: Path, rebuild: bool,
                         slot_budget: int = 14_000_000) -> pl.DataFrame:
    """Score the unlabelled split a slice of impressions at a time.

    Returns (src_row, position, score) for every candidate slot. Only the score
    is kept per slot -- the feature row that produced it is 30x larger and is
    never needed again.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    n_imp = fs.impressions("submit").select(pl.len()).collect().item()
    # Cost is linear in candidate *slots*, not in impressions, and EB-NeRD's
    # submit split is not uniform: its last 536,710 rows are the beyond-accuracy
    # track, which carries ~100 candidates each against ~11.7 everywhere else --
    # 53.9M slots where a normal chunk holds 11.7M. A fixed impression chunk put
    # 4.6x the work in the smallest slice and was killed for it, twice, silently.
    # So a chunk that exceeds the slot budget is sub-divided; the legacy cache
    # names for already-scored chunks still resolve.
    n_cand = (fs.impressions("submit").with_row_index("_i")
              .select("_i", "n_candidates").collect())
    print(f"   submit split: {n_imp:,} impressions, "
          f"{int(n_cand['n_candidates'].sum()):,} slots, "
          f"chunk {chunk:,} impressions capped at {slot_budget:,} slots")
    parts = []
    for start in range(0, n_imp, chunk):
        stop = min(start + chunk, n_imp)
        # No legacy single-name cache here any more: it was keyed on `start`
        # alone and assumed to cover [start, stop), which stops being true the
        # moment the chunk size or the slot budget changes. The slot-bounded
        # names below carry both ends, so a stale one cannot be mistaken for a
        # range it does not hold.
        # split this impression range into slot-bounded pieces
        seg = n_cand.slice(start, stop - start)
        cum = seg["n_candidates"].cast(pl.Int64).cum_sum().to_numpy()
        bounds, lo = [], 0
        while lo < len(cum):
            target = (cum[lo - 1] if lo else 0) + slot_budget
            hi = int(np.searchsorted(cum, target, side="right"))
            hi = max(hi, lo + 1)
            bounds.append((start + lo, start + min(hi, len(cum))))
            lo = hi
        for (a_, b_) in bounds:
            cache = out_dir / f"submit_scores_{a_}_{b_}.parquet"
            if cache.exists() and not rebuild:
                parts.append(pl.read_parquet(cache)); print(f"     [{a_:,}-{b_:,}] cached")
                continue
            _score_range(fs, model, feats, emb, k1, b, n_cand, a_, b_, cache, parts)
    return pl.concat(parts)


def _score_range(fs, model, feats, emb, k1, b, n_cand, a_, b_, cache, parts):
    t0 = time.perf_counter()
    keep = (fs.impressions("submit").with_row_index("_i")
            .filter((pl.col("_i") >= a_) & (pl.col("_i") < b_))
            .select("src_row").collect()["src_row"])
    sub = _assemble_rows(fs, emb, k1, b, keep)
    s = predict(model, sub, feats)
    part = sub.select("src_row", "position").with_columns(
        pl.Series("score", s.astype(np.float32)))
    part.write_parquet(cache)
    parts.append(part)
    print(f"     [{a_:,}-{b_:,}] {part.height:,} slots in {time.perf_counter() - t0:.0f}s")




def _assemble_rows(fs, emb, k1, b, src_rows: pl.Series) -> pl.DataFrame:
    """Feature matrix for a specific set of submit rows."""
    from newsrec.behaviour import attach_features, pairs as _pairs
    imp = fs.impressions("submit").filter(pl.col("src_row").is_in(src_rows.implode()))
    p = (imp.with_row_index("imp")
         .with_columns(pl.int_ranges(0, pl.col("candidates").list.len()).alias("position"))
         .explode(["candidates", "position"])
         .drop_nulls("candidates")
         .rename({"candidates": "article_idx"})
         .with_columns(pl.lit(False).alias("label"),
                       (pl.col("position") / pl.max_horizontal(
                           pl.col("n_candidates") - 1, pl.lit(1))).cast(pl.Float32)
                       .alias("position_frac"))
         .drop("clicked")
         .collect(engine="streaming"))
    df = attach_features(fs, "submit", p, labelled=False)
    df = add_matching_features(fs, "submit", df, emb, k1, b)
    return df.sort("src_row", "position")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["ebnerd", "mind"])
    ap.add_argument("--variant", default="large")
    ap.add_argument("--embedding", default=None)
    ap.add_argument("--objective", default="lambdarank")
    ap.add_argument("--rounds", type=int, default=600)
    ap.add_argument("--train-impressions", type=int, default=1_500_000,
                    help="impressions sampled for training (never candidates "
                         "within one). 0 = all")
    ap.add_argument("--chunk", type=int, default=1_000_000,
                    help="submit impressions scored per pass")
    ap.add_argument("--slot-budget", type=int, default=14_000_000,
                    help="hard cap on candidate slots per pass; cost is linear "
                         "in slots and EB-NeRD's beyond-accuracy tail carries "
                         "~100 candidates per impression against ~11.7 elsewhere")
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--skip-train", action="store_true",
                    help="reuse the saved large model and only re-score")
    ap.add_argument("--out", type=Path, default=Path("reports/sub"))
    a = ap.parse_args()

    emb = a.embedding or ("contrastive" if a.dataset == "ebnerd"
                          else "sentence-transformers/all-MiniLM-L6-v2")
    fs = FeatureStore(a.dataset, a.variant)
    bmf = Path("reports/q2") / f"q2_bm25_{a.dataset}_{a.variant}.json"
    bm = json.loads(bmf.read_text())["best"] if bmf.exists() else {"k1": 2.0, "b": 1.0}
    k1, b = float(bm["k1"]), float(bm["b"])
    print(f"[{a.dataset}/{a.variant}] {fs}\n   embedding {emb}; BM25 k1={k1} b={b}")

    root = Path("artifacts/features") / a.dataset / a.variant
    model_path = a.out / f"model_{a.dataset}_{a.variant}.txt"

    import lightgbm as lgb
    if a.skip_train and model_path.exists():
        model = lgb.Booster(model_file=str(model_path))
        feats = model.feature_name()
        print(f"   reusing {model_path} ({len(feats)} features)")
    else:
        tr = build(fs, "train", emb, k1, b, root / "train.parquet",
                   a.train_impressions, a.rebuild)
        va = build(fs, "val", emb, k1, b, root / "val.parquet", 0, a.rebuild)
        feats = feature_names(tr, SHIPPED)
        print(f"   {len(feats)} shipped features")
        t0 = time.perf_counter()
        model = train(tr, va, feats, objective=a.objective, num_boost_round=a.rounds)
        print(f"   fitted {model.best_iteration} trees in {time.perf_counter() - t0:.0f}s")
        a.out.mkdir(parents=True, exist_ok=True)
        model.save_model(str(model_path))
        print(importances(model, feats).head(10))

    scores = score_submit_chunked(fs, model, feats, emb, k1, b, a.chunk,
                                  root / "submit_chunks", a.rebuild, a.slot_budget)

    # --- re-attach to the raw file, in its own order
    lists = (scores.lazy().sort("src_row", "position")
             .group_by("src_row", maintain_order=True)
             .agg(pl.col("score").alias("_scores")))
    raw, _ = raw_behaviours(fs, DATA_ROOT / "raw" / a.dataset)
    raw = raw.with_row_index("src_row").with_columns(
        pl.col("src_row").cast(pl.UInt32))
    joined = raw.join(lists, on="src_row", how="left")

    # a missing or mis-sized list is a rejected submission, not a bad score
    chk = (joined.select(
        pl.len().alias("rows"),
        pl.col("_scores").is_null().sum().alias("missing"),
        (pl.col("_scores").list.len() != pl.col("candidates").list.len())
        .sum().alias("size_mismatch")).collect().to_dicts()[0])
    print(f"\n   raw rows {chk['rows']:,}; missing score lists {chk['missing']:,}; "
          f"candidate-count mismatches {chk['size_mismatch']:,}")
    if chk["missing"] or chk["size_mismatch"]:
        raise SystemExit("submission would not line up with the raw file -- refusing to write")

    a.out.mkdir(parents=True, exist_ok=True)
    txt = a.out / f"{a.dataset}_rerank.txt"
    n = write_submission(joined.select("src_row", "impression_id", "_scores"), txt,
                         order_by="src_row")

    # Verify the file against the raw split rather than trusting the writer.
    # The line count alone cannot see a chunk boundary that emits one row twice
    # and drops another, which is exactly the bug this guards -- so compare the
    # id sequence itself, in order.
    want = raw.select("impression_id").collect()["impression_id"].to_numpy()
    got = np.fromiter((int(ln.split(" ", 1)[0]) for ln in open(txt)),
                      dtype=np.int64, count=n)
    if len(got) != len(want) or not np.array_equal(got, want):
        bad = int(np.flatnonzero(got[:len(want)] != want[:len(got)])[0]) \
            if len(got) == len(want) else -1
        raise SystemExit(
            f"submission does not reproduce the raw id sequence "
            f"(wrote {len(got):,}, raw {len(want):,}"
            + (f", first mismatch at line {bad + 1:,}" if bad >= 0 else "")
            + ") -- refusing to ship")
    print(f"   id sequence matches the raw file exactly ({len(want):,} rows)")
    zp = zip_submission(txt, a.out / f"{a.dataset}_rerank.zip", arcname=ARCNAME[a.dataset])
    print(f"   wrote {n:,} lines -> {txt}")
    print(f"   zipped as {ARCNAME[a.dataset]} -> {zp} "
          f"({zp.stat().st_size / 2**20:.1f} MB)")

    meta = {"dataset": a.dataset, "variant": a.variant, "embedding": emb,
            "features": list(feats), "n_features": len(feats),
            "best_iteration": int(model.best_iteration),
            "train_impressions": a.train_impressions,
            "rows_written": int(n), "zip": str(zp),
            "archive_name": ARCNAME[a.dataset]}
    (a.out / f"submit_{a.dataset}.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
