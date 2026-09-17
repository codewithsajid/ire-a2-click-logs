"""EB-NeRD (parquet) -> canonical tables."""
from __future__ import annotations

import polars as pl

from .config import Config, SplitSpec
from .ids import remap_list


def articles(cfg: Config) -> pl.LazyFrame:
    """Union the article tables of every source bundle, deduplicated by id.

    demo/small ship a subset of the 125,541-article catalogue; `large` pairs
    ebnerd_large with ebnerd_testset (identical article tables, but unioning is
    cheap insurance against a future bundle where they differ).
    """
    frames = [
        pl.scan_parquet(cfg.raw / src / "articles.parquet")
        for src in cfg.article_sources
    ]
    return (
        pl.concat(frames, how="vertical_relaxed")
        .unique(subset=["article_id"], keep="first")
        .sort("article_id")
        .select(
            pl.col("article_id").cast(pl.Utf8).alias("src_id"),
            pl.col("title"),
            pl.col("subtitle").alias("abstract"),
            pl.col("body"),
            pl.col("category_str").alias("category"),
            pl.col("subcategory").cast(pl.List(pl.Utf8)),
            pl.col("ner_clusters").alias("entities"),
            pl.col("published_time"),
            # EB-NeRD extras kept because the eval harness uses them:
            # sentiment for diversity, pageviews for the head/tail slice.
            pl.col("sentiment_score"),
            pl.col("sentiment_label"),
            pl.col("total_pageviews"),
            pl.col("premium"),
            pl.col("article_type"),
        )
        .with_columns(
            pl.concat_str([pl.col("title"), pl.col("abstract")], separator=" ", ignore_nulls=True).alias("text"),
            pl.lit("da").alias("lang"),
            (pl.col("abstract").fill_null("").str.strip_chars().str.len_chars() > 0).alias("has_abstract"),
        )
    )


def _behaviors(cfg: Config, spec: SplitSpec) -> pl.LazyFrame:
    """Raw behaviours, tagged with their position in the source file.

    `src_row` is assigned before the time filter, so it always means "row n of
    that behaviours file" -- the identifier both leaderboards score against.
    """
    part = f"{spec.part}/" if spec.part else ""
    lf = pl.scan_parquet(cfg.raw / spec.source / part / "behaviors.parquet") \
           .with_row_index("src_row")
    if spec.time_from is not None:
        lf = lf.filter(pl.col("impression_time") >= pl.lit(spec.time_from))
    if spec.time_to is not None:
        lf = lf.filter(pl.col("impression_time") < pl.lit(spec.time_to))
    return lf


def impressions(cfg: Config, spec: SplitSpec, art_map: pl.LazyFrame, user_map: pl.LazyFrame) -> pl.LazyFrame:
    """One canonical row per behaviours row.

    Row-preserving, and enforced rather than assumed: EB-NeRD's test split reuses
    `impression_id = 0` for all 200,000 beyond-accuracy rows, so anything keyed on
    impression_id silently merges them, and the official submission writer emits
    one line per row. A plain join returns rows in hash order -- this cost a
    rejected leaderboard submission once -- so the join is pinned with
    `maintain_order` and `src_row` carries the position regardless.
    """
    lf = _behaviors(cfg, spec).with_columns(pl.col("user_id").cast(pl.Utf8))
    lf = remap_list(lf, "article_ids_inview", art_map, out="candidates")
    if spec.labelled:
        lf = remap_list(lf, "article_ids_clicked", art_map, out="clicked")
    else:
        lf = lf.with_columns(pl.lit([], dtype=pl.List(pl.UInt32)).alias("clicked"))
    return (
        lf.select(
            "src_row",
            "impression_id",
            "user_id",
            pl.col("impression_time").alias("time"),
            "candidates", "clicked",
            # serving-time context only; next_* deliberately absent
            "device_type", "session_id", "is_subscriber", "is_sso_user",
            "read_time", "scroll_percentage",
        )
        .join(user_map.rename({"src_id": "user_id"}), on="user_id", how="left",
              maintain_order="left")
        .rename({"idx": "user_idx"})
        .drop("user_id")
        .with_columns(pl.col("candidates").list.len().cast(pl.UInt16).alias("n_candidates"))
    )


def history(cfg: Config, spec: SplitSpec, art_map: pl.LazyFrame, user_map: pl.LazyFrame) -> pl.LazyFrame:
    """EB-NeRD ships a fixed 21-day lookback ending exactly at the split start,
    so the shipped history is leakage-free by construction."""
    part = f"{spec.part}/" if spec.part else ""
    lf = pl.scan_parquet(cfg.raw / spec.source / part / "history.parquet").with_columns(
        pl.col("user_id").cast(pl.Utf8)
    )
    lf = remap_list(lf, "article_id_fixed", art_map, out="article_idx")
    return (
        lf.select("user_id", "article_idx", pl.col("impression_time_fixed").alias("ts"))
        .join(user_map.rename({"src_id": "user_id"}), on="user_id", how="left",
              maintain_order="left")
        .rename({"idx": "user_idx"})
        .drop("user_id")
        .with_columns(pl.col("article_idx").fill_null(pl.lit([], dtype=pl.List(pl.UInt32))))
        .with_columns(pl.col("article_idx").list.len().cast(pl.UInt32).alias("n_hist"))
    )


def user_ids(cfg: Config) -> pl.LazyFrame:
    """Every user id appearing in any split, so user_idx is stable across them."""
    frames = []
    for spec in cfg.splits.values():
        part = f"{spec.part}/" if spec.part else ""
        frames.append(
            pl.scan_parquet(cfg.raw / spec.source / part / "behaviors.parquet")
            .select(pl.col("user_id").cast(pl.Utf8).alias("src_id"))
        )
    return pl.concat(frames, how="vertical_relaxed")
