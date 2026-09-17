"""Derived user and article features, all computed on a strictly-prior window.

Every function takes the target split's start as `cutoff` and may only read data
timestamped before it. That single rule is what the leakage tests check, and it
is why the earliest split legitimately produces empty feature tables.

Exposure counts exploit the fact that splits do not overlap in time: the
articles shown before split S are exactly the union of the splits that end
before S starts, so per-split exposure counts are computed once and prefix-summed
rather than re-exploding 200M candidate rows per split.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import polars as pl


def exposures(impressions: pl.LazyFrame) -> pl.LazyFrame:
    """How often each article was shown in this split."""
    return (
        impressions.select("candidates")
        .explode("candidates")
        .drop_nulls("candidates")
        .group_by("candidates")
        .agg(pl.len().alias("n_inview"))
        .rename({"candidates": "article_idx"})
    )


def article_features(
    clickstream: pl.LazyFrame,
    prior_exposures: pl.LazyFrame,
    cutoff: datetime,
    halflife_hours: float,
    smoothing: float = 100.0,
) -> pl.LazyFrame:
    """Per-article CTR and time-decayed popularity as of `cutoff`.

    CTR is Bayesian-smoothed towards the corpus mean, because a single click on a
    single exposure is not a 100% CTR -- and news catalogues are full of articles
    with a handful of exposures.
    """
    clicks = (
        clickstream.filter(pl.col("ts") < pl.lit(cutoff))
        .with_columns(
            # exponential recency weight: a click one half-life old counts half
            (0.5 ** ((pl.lit(cutoff) - pl.col("ts")).dt.total_seconds() / (3600.0 * halflife_hours)))
            .alias("_w")
        )
        .group_by("article_idx")
        .agg(
            pl.len().alias("clicks"),
            pl.col("_w").sum().alias("clicks_decayed"),
            pl.col("ts").max().alias("last_click"),
            pl.col("ts").min().alias("first_click"),
        )
    )
    joined = prior_exposures.join(clicks, on="article_idx", how="full", coalesce=True).with_columns(
        pl.col("n_inview").fill_null(0),
        pl.col("clicks").fill_null(0),
        pl.col("clicks_decayed").fill_null(0.0),
    )
    # corpus mean CTR is the smoothing prior
    totals = joined.select(
        (pl.col("clicks").sum() / pl.max_horizontal(pl.col("n_inview").sum(), pl.lit(1))).alias("_prior")
    )
    return (
        joined.join(totals, how="cross")
        .with_columns(
            ((pl.col("clicks") + smoothing * pl.col("_prior")) /
             (pl.col("n_inview") + smoothing)).alias("ctr_smoothed"),
            (pl.col("clicks") / pl.max_horizontal(pl.col("n_inview"), pl.lit(1))).alias("ctr_raw"),
            pl.lit(str(cutoff)).alias("computed_through"),
        )
        .drop("_prior")
    )


def user_features(
    history: pl.LazyFrame,
    clickstream: pl.LazyFrame,
    articles: pl.LazyFrame,
    cutoff: datetime,
) -> pl.LazyFrame:
    """Recency, activity and category profile per user, as of `cutoff`.

    Recency is drawn from two sources so both datasets get something: EB-NeRD's
    history carries per-click timestamps, while MIND's does not and can only be
    dated through clicks observed in earlier splits. MIND users who appear for
    the first time in this split therefore have null recency -- which is the
    honest answer, not a zero.
    """
    hist_ts = (
        history.select("user_idx", "ts")
        .explode("ts")
        .drop_nulls("ts")
        .select("user_idx", pl.col("ts"))
    )
    stream_ts = clickstream.filter(pl.col("ts") < pl.lit(cutoff)).select("user_idx", "ts")
    events = pl.concat([hist_ts, stream_ts], how="vertical_relaxed").filter(pl.col("ts") < pl.lit(cutoff))

    recency = events.group_by("user_idx").agg(
        pl.col("ts").max().alias("last_click_ts"),
        (pl.col("ts") >= pl.lit(cutoff - timedelta(hours=24))).sum().alias("clicks_24h"),
        (pl.col("ts") >= pl.lit(cutoff - timedelta(days=7))).sum().alias("clicks_7d"),
    ).with_columns(
        ((pl.lit(cutoff) - pl.col("last_click_ts")).dt.total_seconds() / 3600.0).alias("hours_since_last_click")
    )

    # category profile over the user's history
    cats = articles.with_row_index("idx").select(pl.col("idx").cast(pl.UInt32), "category")
    per_cat = (
        history.select("user_idx", "article_idx")
        .explode("article_idx")
        .drop_nulls("article_idx")
        .join(cats, left_on="article_idx", right_on="idx", how="inner")
        .group_by("user_idx", "category")
        .agg(pl.len().alias("n"))
    )
    profile = (
        per_cat.with_columns((pl.col("n") / pl.col("n").sum().over("user_idx")).alias("share"))
        .group_by("user_idx")
        .agg(
            pl.col("category").sort_by("n", descending=True).head(3).alias("top_categories"),
            pl.col("share").max().alias("top_category_share"),
            pl.col("category").n_unique().alias("n_categories"),
            # Shannon entropy of the category distribution: breadth of interest
            (-(pl.col("share") * pl.col("share").log()).sum()).alias("category_entropy"),
        )
    )
    return (
        history.select("user_idx", "n_hist")
        .join(recency, on="user_idx", how="left")
        .join(profile, on="user_idx", how="left")
        .with_columns(
            pl.col("clicks_24h").fill_null(0),
            pl.col("clicks_7d").fill_null(0),
            pl.lit(str(cutoff)).alias("computed_through"),
        )
    )
