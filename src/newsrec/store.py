"""Read side of the feature store.

Everything is parquet on disk plus `.npy` matrices for embeddings, so there is
no server and no format lock-in; this class just hides the paths and the two
history modes behind one object.

    fs = FeatureStore("mind", "small")
    fs.impressions("val")                  # LazyFrame
    fs.history("val", mode="augmented")    # LazyFrame
    fs.texts()                             # list[str], row i == article idx i
    fs.embeddings("xlm_roberta")           # np.ndarray (n_articles, dim)
    fs.candidate_universe(t, days=7)       # articles "live" at time t
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from functools import cached_property
from pathlib import Path

import numpy as np
import polars as pl

from .config import DATA_ROOT


class FeatureStore:
    def __init__(self, dataset: str, variant: str, root: Path | None = None):
        self.root = (root or DATA_ROOT / "processed") / dataset / variant
        if not (self.root / "manifest.json").exists():
            raise FileNotFoundError(f"no store at {self.root}; run `python -m newsrec.build --config ...`")
        self.dataset, self.variant = dataset, variant

    @cached_property
    def manifest(self) -> dict:
        return json.loads((self.root / "manifest.json").read_text())

    @property
    def splits(self) -> list[str]:
        return list(self.manifest["time_bounds"])

    def bounds(self, split: str) -> tuple[datetime, datetime]:
        b = self.manifest["time_bounds"][split]
        return datetime.fromisoformat(b["min"]), datetime.fromisoformat(b["max"])

    # ------------------------------------------------------------- tables

    def articles(self) -> pl.LazyFrame:
        return pl.scan_parquet(self.root / "articles.parquet")

    def impressions(self, split: str) -> pl.LazyFrame:
        return pl.scan_parquet(self.root / "splits" / split / "impressions.parquet")

    def history(self, split: str, mode: str = "shipped") -> pl.LazyFrame:
        name = "history.parquet" if mode == "shipped" else "history_augmented.parquet"
        return pl.scan_parquet(self.root / "splits" / split / name)

    def popularity(self, split: str) -> pl.LazyFrame:
        return pl.scan_parquet(self.root / "splits" / split / "popularity.parquet")

    def article_stats(self) -> pl.LazyFrame:
        """Catalogue timeline only (first_seen_time). Exposure/click counts are
        deliberately per-split -- see `article_features`."""
        return pl.scan_parquet(self.root / "article_stats.parquet")

    def user_features(self, split: str) -> pl.LazyFrame:
        """Recency, activity and category profile as of `split`'s start."""
        return pl.scan_parquet(self.root / "splits" / split / "user_features.parquet")

    def article_features(self, split: str) -> pl.LazyFrame:
        """CTR and decayed popularity from strictly-prior windows only."""
        return pl.scan_parquet(self.root / "splits" / split / "article_features.parquet")

    def exposures(self, split: str) -> pl.LazyFrame:
        """How often each article was shown *within* this split. Only safe to use
        for splits strictly earlier than the one being scored."""
        return pl.scan_parquet(self.root / "splits" / split / "exposures.parquet")

    # ------------------------------------------------- retrieval helpers

    @cached_property
    def n_articles(self) -> int:
        return self.articles().select(pl.len()).collect().item()

    def lang(self) -> str:
        """Corpus language -- picks the Snowball stemmer for BM25 (da vs en)."""
        return self.articles().select("lang").first().collect().item()

    def texts(self) -> list[str]:
        """Article text in idx order -- the corpus BM25 indexes."""
        return (
            self.articles()
            .select(pl.col("text").fill_null(""))
            .collect()["text"]
            .to_list()
        )

    def embeddings(self, name: str) -> np.ndarray:
        path = self.root / "embeddings" / f"{name}.npy"
        if not path.exists():
            have = sorted(p.stem for p in (self.root / "embeddings").glob("*.npy")) \
                if (self.root / "embeddings").exists() else []
            raise FileNotFoundError(f"no embedding '{name}' in {self.root}; have {have}")
        return np.load(path, mmap_mode="r")

    def candidate_universe(self, start: datetime, end: datetime | None = None,
                           days: int = 7) -> np.ndarray:
        """Articles that could plausibly be shown during [start, end].

        Corpus-wide recall@K is meaningless against the whole catalogue -- EB-NeRD
        carries articles published in 1993 -- so retrieval is restricted to
        articles fresh enough to be in circulation. The window is anchored at the
        *start* of the scored period and runs to its end: an impression at `start`
        can be shown an article published up to `days` earlier, so anchoring at
        the end instead silently excludes every article that was already popular
        when the period began (which made the popularity baseline score exactly
        zero -- a bug, not a result).

        Uses published_time where it exists (EB-NeRD) and first_seen_time
        otherwise (MIND ships no publication date).
        """
        end = end or start
        stats = self.article_stats().join(
            self.articles().with_row_index("idx").select("idx", "published_time"),
            on="idx", how="left")
        t0 = start - timedelta(days=days)
        alive = stats.with_columns(
            pl.coalesce(pl.col("published_time"), pl.col("first_seen_time")).alias("_t")
        ).filter((pl.col("_t") >= pl.lit(t0)) & (pl.col("_t") <= pl.lit(end)))
        return alive.select("idx").collect()["idx"].to_numpy()

    def cold_start_mask(self, split: str, threshold: int = 5, mode: str = "shipped") -> pl.LazyFrame:
        """Per-user cold/warm label for the Q4 slice.

        Keyed on history length rather than "user seen in training": MINDsmall's
        train and dev user pools overlap by only 11.9%, so a seen/unseen split
        would label almost every dev user cold and measure nothing.
        """
        return self.history(split, mode).select(
            "user_idx",
            "n_hist",
            (pl.col("n_hist") < threshold).alias("is_cold"),
        )

    def __repr__(self) -> str:
        return (f"FeatureStore({self.dataset}/{self.variant}, "
                f"{self.n_articles:,} articles, splits={self.splits})")
