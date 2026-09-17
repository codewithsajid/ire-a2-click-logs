"""One-command rebuild: raw files -> feature store.

    python -m newsrec.build --config configs/mind_small.yaml [--force] [--engine gpu]

Every stage writes one parquet and records its wall time, peak RSS and row count
in `manifest.json`. Stages are skipped when their output exists and the config
fingerprint is unchanged, so a rebuild after a config tweak only redoes what the
tweak touched. The manifest timings are what the 10x scale analysis is built on.
"""
from __future__ import annotations

import argparse
import json
import resource
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import polars as pl

from . import features, ingest_ebnerd, ingest_mind
from .config import Config
from .engine import collect
from .ids import build_id_map


def _mod(cfg: Config):
    return ingest_ebnerd if cfg.dataset == "ebnerd" else ingest_mind


def _peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


class Builder:
    def __init__(self, cfg: Config, force: bool = False):
        self.cfg, self.force = cfg, force
        self.mod = _mod(cfg)
        self.out = cfg.out
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / "splits").mkdir(exist_ok=True)
        self.manifest = {
            "config": {k: (asdict(v) if hasattr(v, "__dataclass_fields__") else v)
                       for k, v in asdict(cfg).items()},
            "fingerprint": cfg.fingerprint(),
            "built_at": datetime.now().isoformat(timespec="seconds"),
            "stages": {},
        }
        prev = self.out / "manifest.json"
        self.prev_fp = json.loads(prev.read_text()).get("fingerprint") if prev.exists() else None
        self.reuse = self._reusable()

    def _reusable(self) -> bool:
        """Can cached stages be trusted? Says out loud which part moved.

        A pre-fingerprint-split manifest carries a bare string, which cannot be
        compared part-by-part -- treat it as stale rather than guess.
        """
        now = self.cfg.fingerprint()
        if self.prev_fp is None:
            return False
        if not isinstance(self.prev_fp, dict):
            print("  [rebuild] manifest predates the split fingerprint", flush=True)
            return False
        changed = [k for k, v in now.items() if self.prev_fp.get(k) != v]
        if changed:
            what = {"config": "the config", "code": "build-relevant source",
                    "raw": "the raw inputs"}
            print(f"  [rebuild] {', '.join(what[c] for c in changed)} changed "
                  f"since the last build", flush=True)
            return False
        return True

    def _checkpoint(self) -> None:
        """Persist the manifest after each stage: a 13.5M-impression build that
        dies at hour two should resume, not restart."""
        (self.out / "manifest.json").write_text(json.dumps(self.manifest, indent=2, default=str))

    def stage(self, name: str, path: Path, fn):
        """Run `fn` unless `path` is already valid for this config."""
        if path.exists() and not self.force and self.reuse:
            n = pl.scan_parquet(path).select(pl.len()).collect().item()
            print(f"  [skip] {name:22s} {n:>12,} rows", flush=True)
            self.manifest["stages"][name] = {"skipped": True, "rows": n, "path": str(path)}
            return
        t0 = time.perf_counter()
        rows = fn()                       # DataFrame, or a row count if it wrote itself
        if isinstance(rows, pl.DataFrame):
            rows.write_parquet(path, compression="zstd")
            rows = rows.height
        dt = time.perf_counter() - t0
        print(f"  [ok]   {name:22s} {rows:>12,} rows  {dt:7.2f}s  peak {_peak_rss_mb():7.0f} MB", flush=True)
        self.manifest["stages"][name] = {
            "rows": rows, "seconds": round(dt, 3),
            "peak_rss_mb": round(_peak_rss_mb(), 1), "path": str(path),
        }
        self._checkpoint()

    # ---------------------------------------------------------------- stages

    def run(self):
        cfg, mod = self.cfg, self.mod
        print(f"== building {cfg.dataset}/{cfg.variant}  ->  {self.out}")

        self.stage("articles", self.out / "articles.parquet",
                   lambda: collect(mod.articles(cfg), cfg.engine))
        self.stage("users", self.out / "users.parquet",
                   lambda: collect(build_id_map(mod.user_ids(cfg), "src_id"), cfg.engine))

        art_map = pl.scan_parquet(self.out / "articles.parquet").select("src_id").with_row_index("idx") \
                    .with_columns(pl.col("idx").cast(pl.UInt32))
        user_map = pl.scan_parquet(self.out / "users.parquet")

        for name, spec in cfg.splits.items():
            d = self.out / "splits" / name
            d.mkdir(parents=True, exist_ok=True)
            self.stage(f"{name}/impressions", d / "impressions.parquet",
                       lambda s=spec, n=name: self._impressions(n, s, art_map, user_map))
            self.stage(f"{name}/history", d / "history.parquet",
                       lambda s=spec: collect(mod.history(cfg, s, art_map, user_map), cfg.engine))

        self.stage("clickstream", self.out / "clickstream.parquet", self._clickstream)
        self._time_bounds()
        for name, spec in cfg.splits.items():
            d = self.out / "splits" / name
            self.stage(f"{name}/exposures", d / "exposures.parquet",
                       lambda n=name: collect(features.exposures(self._split(n)), cfg.engine))
        for name, spec in cfg.splits.items():
            d = self.out / "splits" / name
            # Only labelled splits are ever scored with augmented history (it is
            # the Q9 ablation), and on EB-NeRD large each copy is 660 MB. The
            # unlabelled leaderboard split would just be dead weight.
            if spec.labelled:
                self.stage(f"{name}/history_augmented", d / "history_augmented.parquet",
                           lambda n=name: self._augmented_history(n))
            else:
                # gating a stage does not remove what an earlier build left behind
                stale = d / "history_augmented.parquet"
                if stale.exists():
                    print(f"  [drop] {name}/history_augmented    "
                          f"{stale.stat().st_size / 1e6:,.0f} MB (unlabelled split)", flush=True)
                    stale.unlink()
            self.stage(f"{name}/popularity", d / "popularity.parquet",
                       lambda n=name: self._popularity(n))
            self.stage(f"{name}/article_features", d / "article_features.parquet",
                       lambda n=name: self._article_features(n))
            self.stage(f"{name}/user_features", d / "user_features.parquet",
                       lambda n=name: self._user_features(n))
        self.stage("article_stats", self.out / "article_stats.parquet", self._article_stats)
        self._embeddings()

        (self.out / "manifest.json").write_text(json.dumps(self.manifest, indent=2, default=str))
        print(f"== manifest -> {self.out / 'manifest.json'}")

    def _impressions(self, name: str, spec, art_map, user_map):
        """Ingest one split's impressions.

        This used to ingest in day-sized chunks, because the first version
        remapped ids by exploding the candidate lists and 13.5M impressions x
        ~15 candidates OOMed a 125 GB box. The in-place `replace_strict` remap in
        `ids.remap_list` removed the reason: measured on EB-NeRD large's train
        split, chunked was 5.16 s at 11.6 GB peak and unchunked is 1.37 s at
        8.9 GB. The chunking code was kept for a while as an escape hatch and
        never re-enabled, so it is gone.
        """
        return collect(self.mod.impressions(self.cfg, spec, art_map, user_map), self.cfg.engine)

    def _split(self, name: str, table: str = "impressions") -> pl.LazyFrame:
        return pl.scan_parquet(self.out / "splits" / name / f"{table}.parquet")

    def _clickstream(self) -> pl.DataFrame:
        """(user, article, time) for every observed click on a labelled split.
        The single source `history_mode: augmented` draws on."""
        frames = [
            self._split(n).select("user_idx", "time", "clicked")
                        .explode("clicked").drop_nulls("clicked")
                        .select("user_idx", pl.col("clicked").alias("article_idx"), pl.col("time").alias("ts"))
            for n, s in self.cfg.splits.items() if s.labelled
        ]
        return collect(pl.concat(frames, how="vertical_relaxed").unique(), self.cfg.engine)

    def _time_bounds(self):
        """Record each split's window; the leakage test asserts they don't overlap."""
        bounds = {}
        for name in self.cfg.splits:
            b = collect(self._split(name).select(
                pl.col("time").min().alias("min"), pl.col("time").max().alias("max")), self.cfg.engine)
            bounds[name] = {"min": str(b["min"][0]), "max": str(b["max"][0])}
        self.manifest["time_bounds"] = bounds
        self.bounds = {k: pl.scan_parquet(self.out / "splits" / k / "impressions.parquet")
                       for k in self.cfg.splits}

    def _split_start(self, name: str) -> datetime:
        return datetime.fromisoformat(self.manifest["time_bounds"][name]["min"])

    def _augmented_history(self, name: str) -> pl.DataFrame:
        """Shipped history extended with clicks observed strictly *before* this
        split starts. The cutoff is the split boundary, not the impression time:
        conservative, and it makes the no-leakage assertion a one-liner."""
        cutoff = self._split_start(name)
        shipped = self._split(name, "history")
        extra = (
            pl.scan_parquet(self.out / "clickstream.parquet")
            .filter(pl.col("ts") < pl.lit(cutoff))
            .group_by("user_idx")
            .agg(pl.col("article_idx").sort_by("ts").alias("_add"),
                 pl.col("ts").sort().alias("_add_ts"))
        )
        joined = shipped.join(extra, on="user_idx", how="left")
        return collect(
            joined.with_columns(
                # append only clicks the shipped history does not already contain;
                # never deduplicate the shipped part -- repeated reads are signal
                pl.col("article_idx").list.concat(
                    pl.col("_add")
                      .fill_null(pl.lit([], dtype=pl.List(pl.UInt32)))
                      .list.set_difference(pl.col("article_idx"))
                ).alias("article_idx"),
            )
            .with_columns(pl.col("article_idx").list.len().cast(pl.UInt32).alias("n_hist"),
                          pl.lit(str(cutoff)).alias("cutoff"))
            .drop("_add", "_add_ts"),
            self.cfg.engine,
        )

    def _popularity(self, name: str) -> pl.DataFrame:
        """Click counts over a window ending strictly at the split start, so a
        popularity feature never sees a click from the window it scores."""
        cutoff = self._split_start(name)
        return collect(
            pl.scan_parquet(self.out / "clickstream.parquet")
            .filter(pl.col("ts") < pl.lit(cutoff))
            .group_by("article_idx")
            .agg(pl.len().alias("clicks"), pl.col("ts").max().alias("last_click"))
            .with_columns(pl.lit(str(cutoff)).alias("computed_through"))
            .sort("clicks", descending=True),
            self.cfg.engine,
        )

    def _splits_before(self, name: str) -> list[str]:
        """Splits that end before `name` starts. Splits never overlap in time, so
        this is exactly the data legitimately visible when scoring `name`."""
        start = self._split_start(name)
        return [
            other for other in self.cfg.splits
            if datetime.fromisoformat(self.manifest["time_bounds"][other]["max"]) < start
        ]

    def _prior_exposures(self, name: str) -> pl.LazyFrame:
        prior = self._splits_before(name)
        if not prior:
            return pl.LazyFrame(schema={"article_idx": pl.UInt32, "n_inview": pl.UInt32})
        return (
            pl.concat([pl.scan_parquet(self.out / "splits" / p / "exposures.parquet") for p in prior],
                      how="vertical_relaxed")
            .group_by("article_idx").agg(pl.col("n_inview").sum())
        )

    def _article_features(self, name: str) -> pl.DataFrame:
        return collect(features.article_features(
            pl.scan_parquet(self.out / "clickstream.parquet"),
            self._prior_exposures(name),
            self._split_start(name),
            self.cfg.popularity_halflife_hours,
        ), self.cfg.engine)

    def _user_features(self, name: str) -> pl.DataFrame:
        return collect(features.user_features(
            self._split(name, "history"),
            pl.scan_parquet(self.out / "clickstream.parquet"),
            pl.scan_parquet(self.out / "articles.parquet"),
            self._split_start(name),
        ), self.cfg.engine)

    def _embeddings(self):
        """Align EB-NeRD's four shipped article embedding sets to the dense index
        so Q3's ANN index can consume them directly. MIND ships none; its vectors
        are encoded on demand by newsrec.semantic."""
        if self.cfg.dataset != "ebnerd":
            return
        from .embeddings import align_ebnerd
        t0 = time.perf_counter()
        report = align_ebnerd(self.cfg)
        if report:
            self.manifest["stages"]["embeddings"] = {"seconds": round(time.perf_counter() - t0, 3), **report}
            for name, r in report.items():
                print(f"  [ok]   {'embed/' + name:22s} {r['rows']:>12,} rows  dim {r['dim']:<5} "
                      f"coverage {r['coverage']:.1%}", flush=True)
            self._checkpoint()

    def _article_stats(self) -> pl.DataFrame:
        """Catalogue-level article timeline.

        Only `first_seen_time` lives here, and only because MIND ships no
        publication date and the candidate universe needs *some* notion of when
        an article existed. Exposure and click counts are deliberately NOT here:
        aggregated across all splits they would let a tail article that goes
        viral during the test week be labelled 'head'. Those counts are per-split
        and prior-window-only, in splits/<split>/article_features.parquet.
        """
        frames = [
            self._split(n).select("time", "candidates").explode("candidates")
                          .drop_nulls("candidates").rename({"candidates": "idx"})
            for n in self.cfg.splits
        ]
        return collect(
            pl.concat(frames, how="vertical_relaxed")
            .group_by("idx")
            .agg(pl.col("time").min().alias("first_seen_time"))
            .sort("idx"),
            self.cfg.engine,
        )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--force", action="store_true", help="rebuild every stage")
    ap.add_argument("--engine", choices=["polars", "gpu"], help="override the config's engine")
    a = ap.parse_args()
    cfg = Config.load(a.config)
    if a.engine:
        cfg.engine = a.engine
    Builder(cfg, force=a.force).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
