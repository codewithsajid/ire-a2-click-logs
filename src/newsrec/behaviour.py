"""Q1: behavioural features over the click logs, for the re-ranker to learn on.

Assignment 1 built two *content* signals -- BM25 over the user's history as a bag
of terms, and cosine to a mean-pooled history vector -- plus two *prior* tables
(per-article CTR and time-decayed popularity) computed on a strictly earlier
window. This module turns those into one flat (impression, candidate) design
matrix and adds what A1 did not have: session context, article freshness
measured at the moment of the impression, and the match between a candidate's
category and the user's own click distribution.

Three rules the whole module is built around.

**Everything is dated.** A feature may read an event only if that event happened
strictly before the impression it describes. The A1 feature store already
enforces this at split granularity -- `article_features(split)` and
`user_features(split)` carry a `computed_through` stamp earlier than the split
they serve, and `tests/test_no_leakage.py` asserts it. The features added here
are finer-grained than a split, so they enforce it per row instead: a session
feature for the third impression in a session may read the first two and not the
fourth. `tests/test_behaviour_window.py` asserts that directly.

**Session length is not a feature; session position is.** How many impressions a
session eventually contains is future information -- at the third impression the
system cannot know whether a fourth follows. The causal half of the same signal
is the position reached so far and the time elapsed since the session opened,
and that is what is emitted. This distinction is worth a sentence because the
non-causal version is both easy to write and strongly predictive, which is
exactly the combination Q9 is asking about.

**Rendered position is quarantined.** `position` is where the platform put the
candidate in the list it showed, and clicks concentrate at the top of that list
for reasons that have nothing to do with relevance. It is genuinely available at
serving time -- the system chooses it -- but it is also the mechanism that
generated the labels, so a model that reads it is partly predicting its own
input. It is emitted, flagged in `SERVING_GREY`, and left out of the shipped
feature set; `scripts/q2_rerank.py --grey` measures what including it buys.

The two datasets are not symmetric here and nothing is faked to hide it. EB-NeRD
carries `session_id`, `device_type`, subscriber flags, `read_time` and
`scroll_percentage`; MIND carries none of them, so its session columns are null
and the re-ranker sees a smaller feature set. MIND's history also has no
timestamps, so its recency features come only from clicks observed in earlier
splits.
"""
from __future__ import annotations

from datetime import datetime

import numpy as np
import polars as pl

from .store import FeatureStore

# Columns that describe the visit that is still in progress at prediction time.
# Both ship fully populated in the unlabelled Codabench test sets, so the
# leaderboards permit them, but a production ranker choosing what to show has
# not yet observed how long the user will read or how far they will scroll.
# A1 excluded them from every ranker; A2 Q9 asks for the number, so they are
# built and reported separately rather than dropped.
SERVING_GREY = ("read_time", "scroll_percentage", "position", "position_frac")

# EB-NeRD-only impression context. Absent on MIND, where these come back null.
CONTEXT_COLS = ("device_type", "session_id", "is_subscriber", "is_sso_user",
                "read_time", "scroll_percentage")


def _present(lf: pl.LazyFrame, cols) -> list[str]:
    have = set(lf.collect_schema().names())
    return [c for c in cols if c in have]


# --------------------------------------------------------------- pair table

def pairs(fs: FeatureStore, split: str, max_impressions: int = 0, seed: int = 0,
          labelled: bool = True) -> pl.DataFrame:
    """One row per candidate the platform actually showed, with its label.

    This is A1's `scripts/q4_eval.build_pairs` plus two things the re-ranker
    needs and the A1 rankers did not: the candidate's rendered position, and the
    per-impression context columns. `imp` is a row index over the impressions
    kept, not `impression_id` -- EB-NeRD's 200,000 beyond-accuracy rows all carry
    `impression_id = 0`, so the id does not identify an impression there.

    `max_impressions` subsamples impressions and never candidates within one: a
    partial candidate list would change what AUC and nDCG mean for that row.
    """
    imp = fs.impressions(split)
    if labelled:
        imp = imp.filter(pl.col("clicked").list.len() > 0)
    keep = ["imp", "src_row", "user_idx", "time", "candidates", "clicked",
            "n_candidates", *_present(imp, CONTEXT_COLS)]
    imp = imp.with_row_index("imp").select(keep)

    if max_impressions:
        n = imp.select(pl.len()).collect().item()
        if n > max_impressions:
            sel = np.sort(np.random.default_rng(seed).choice(n, max_impressions, replace=False))
            imp = imp.filter(pl.col("imp").is_in(pl.Series(sel, dtype=pl.UInt32).implode()))

    return (
        imp
        # explode candidates and their rendered index together, so position
        # survives the flattening instead of being recoverable only by accident
        .with_columns(pl.int_ranges(0, pl.col("candidates").list.len()).alias("position"))
        .explode(["candidates", "position"])
        .drop_nulls("candidates")
        .rename({"candidates": "article_idx"})
        .with_columns(
            (pl.col("article_idx").is_in(pl.col("clicked")) if labelled
             else pl.lit(False)).alias("label"),
            # position as a share of the list: EB-NeRD shows 11 candidates and
            # MIND 37, so raw rank is not comparable across the two
            (pl.col("position") / pl.max_horizontal(pl.col("n_candidates") - 1, pl.lit(1)))
            .cast(pl.Float32).alias("position_frac"),
        )
        .drop("clicked")
        .collect(engine="streaming")
    )


# ----------------------------------------------------------- session context

def session_features(imp: pl.LazyFrame) -> pl.LazyFrame:
    """Where this impression sits inside its session, using only what precedes it.

    Emitted per impression (not per candidate):

      * `session_rank`     -- how many impressions of this session came first
      * `session_seconds`  -- elapsed since the session's first impression
      * `secs_since_prev`  -- gap to the previous impression in the session

    All three are cumulative-past quantities. The tempting fourth -- the
    session's total length -- is not here: at impression *k* the number of
    impressions the session will eventually hold is not observable, and a model
    given it learns "long sessions click more" from a column that could not be
    filled at serving time. `tests/test_behaviour_window.py` asserts that no
    emitted column moves when later impressions of the same session are deleted.

    MIND has no `session_id`; there the whole block comes back null.
    """
    if "session_id" not in imp.collect_schema().names():
        return imp.select("imp").with_columns(
            pl.lit(None, dtype=pl.UInt32).alias("session_rank"),
            pl.lit(None, dtype=pl.Float32).alias("session_seconds"),
            pl.lit(None, dtype=pl.Float32).alias("secs_since_prev"),
        )
    return (
        imp.select("imp", "user_idx", "session_id", "time")
        .sort("user_idx", "session_id", "time")
        .with_columns(
            pl.int_range(pl.len()).over("user_idx", "session_id")
              .cast(pl.UInt32).alias("session_rank"),
            ((pl.col("time") - pl.col("time").min().over("user_idx", "session_id"))
             .dt.total_seconds()).cast(pl.Float32).alias("session_seconds"),
            ((pl.col("time") - pl.col("time").shift(1).over("user_idx", "session_id"))
             .dt.total_seconds()).cast(pl.Float32).alias("secs_since_prev"),
        )
        .select("imp", "session_rank", "session_seconds", "secs_since_prev")
    )


# ------------------------------------------------------ user x article match

def user_category_profile(fs: FeatureStore, split: str,
                          mode: str = "shipped") -> pl.LazyFrame:
    """Each user's click share per category, over their history only.

    A1's `user_features` keeps the top three categories and the share of the
    largest; that is enough to describe a user but not to score a *candidate*,
    which needs the share of this particular category. This is the full
    distribution, joined per (user, candidate category) downstream.

    The history table is prior by construction -- it is what the platform shipped
    as "what this user had read before the split" -- so no cutoff argument is
    needed here, unlike the clickstream-derived features in `features.py`.
    """
    cats = fs.articles().with_row_index("idx").select(
        pl.col("idx").cast(pl.UInt32).alias("article_idx"), "category")
    return (
        fs.history(split, mode).select("user_idx", "article_idx")
        .explode("article_idx")
        .drop_nulls("article_idx")
        .join(cats, on="article_idx", how="inner")
        .group_by("user_idx", "category")
        .agg(pl.len().alias("_n"))
        .with_columns(
            (pl.col("_n") / pl.col("_n").sum().over("user_idx")).cast(pl.Float32)
            .alias("cat_share"),
            pl.col("_n").cast(pl.UInt32).alias("cat_clicks"),
        )
        .drop("_n")
    )


def history_recency(fs: FeatureStore, split: str, halflife_hours: float = 24.0,
                    mode: str = "shipped") -> pl.LazyFrame:
    """Exponentially-decayed weight of each (user, previously-clicked article).

    Q1.1 asks for a recency-weighted history. EB-NeRD timestamps every history
    click, so the weight is a real half-life in hours. MIND ships history as an
    ordered list with no times at all, so there the decay runs over *rank* -- the
    k-th most recent click -- which is the only recency information the data
    contains. Both are exponential with the same half-life parameter; they are
    not the same quantity and the design note says so rather than presenting one
    number for both.

    Returned per (user, article) so downstream code can ask "how recently did
    this user click something in this category / this article's neighbourhood".
    """
    h = fs.history(split, mode).select("user_idx", "article_idx", "ts")
    # Probe the data, not the schema. MIND declares `ts` as List(Datetime) to keep
    # one schema across both datasets, but every list is *null* there rather than
    # a list of nulls -- and a two-column explode of a null list yields one null
    # row against N article rows, silently misaligning the history.
    has_ts = bool(
        h.select(pl.col("ts").list.len().fill_null(0).sum() > 0).collect().item()
    )

    if has_ts:
        e = (h.explode(["article_idx", "ts"]).drop_nulls("article_idx"))
        # anchor at the user's own last click, not at the split boundary: the
        # weight then means "recent for this user", which is what the feature is
        # asked to express, and it is defined even where `ts` is null
        e = e.with_columns(
            pl.col("ts").max().over("user_idx").alias("_last")
        ).with_columns(
            (0.5 ** (((pl.col("_last") - pl.col("ts")).dt.total_seconds()
                      / (3600.0 * halflife_hours)))).cast(pl.Float32).alias("hist_w")
        ).drop("_last")
    else:
        e = (
            h.explode("article_idx").drop_nulls("article_idx")
            .with_columns(pl.int_range(pl.len()).over("user_idx").alias("_i"))
            .with_columns(
                ((pl.col("_i").max().over("user_idx") - pl.col("_i"))
                 .cast(pl.Float32)).alias("_age_rank")
            )
            .with_columns((0.5 ** (pl.col("_age_rank") / 10.0)).cast(pl.Float32).alias("hist_w"))
            .drop("_i", "_age_rank")
        )
    return e.select("user_idx", "article_idx", "hist_w")


def category_recency(fs: FeatureStore, split: str, halflife_hours: float = 24.0,
                     mode: str = "shipped") -> pl.LazyFrame:
    """Decayed click mass per (user, category) -- "how warm is this topic for them".

    The un-decayed version of this is `user_category_profile.cat_share`. Keeping
    both lets the ablation separate *what* a user reads from *what they read
    lately*, which on a news corpus with an 80-hour median article age are not
    the same question.
    """
    cats = fs.articles().with_row_index("idx").select(
        pl.col("idx").cast(pl.UInt32).alias("article_idx"), "category")
    return (
        history_recency(fs, split, halflife_hours, mode)
        .join(cats, on="article_idx", how="inner")
        .group_by("user_idx", "category")
        .agg(pl.col("hist_w").sum().alias("cat_recency"))
        .with_columns(
            (pl.col("cat_recency")
             / pl.max_horizontal(pl.col("cat_recency").sum().over("user_idx"), pl.lit(1e-9)))
            .cast(pl.Float32).alias("cat_recency_share")
        )
    )


# ------------------------------------------------------------ article timing

def article_timing(fs: FeatureStore) -> pl.LazyFrame:
    """Publication time per article, for the freshness feature.

    EB-NeRD ships `published_time`. MIND ships no publication date at all, so A1
    derives `first_seen_time` -- the earliest impression the article appears in
    -- which is the only honest proxy and is what `candidate_universe` already
    uses. Coalescing here keeps one definition of "when did this article exist"
    across the two datasets.
    """
    return (
        fs.article_stats()
        .join(fs.articles().with_row_index("idx").select("idx", "published_time",
                                                         "category", "subcategory"),
              on="idx", how="left")
        .select(
            pl.col("idx").cast(pl.UInt32).alias("article_idx"),
            pl.coalesce(pl.col("published_time"), pl.col("first_seen_time")).alias("pub_time"),
            "category", "subcategory",
        )
    )


# ------------------------------------------------------------------ assembly

def assemble(fs: FeatureStore, split: str, max_impressions: int = 0,
             seed: int = 0, history_mode: str = "shipped",
             halflife_hours: float = 24.0, labelled: bool = True) -> pl.DataFrame:
    """The full behavioural design matrix: one row per shown candidate.

    Joins, in order: the pair table, per-impression session context, the prior
    article tables (CTR, decayed popularity, exposures), the prior user tables
    (activity, recency, category breadth), article timing, and the two
    user x candidate match features.

    Nothing here reads a click from inside `split`. The article and user tables
    come from `FeatureStore`, which A1 computes on a strictly earlier window and
    stamps with `computed_through`; the session block is causal within the split;
    freshness and category match read only the catalogue and the user's history.
    """
    p = pairs(fs, split, max_impressions=max_impressions, seed=seed, labelled=labelled)
    lf = p.lazy()

    imp_level = p.lazy().select(
        ["imp", "user_idx", "time",
         *_present(p.lazy(), ("session_id",))]).unique(subset=["imp"])
    lf = lf.join(session_features(imp_level), on="imp", how="left")

    af = fs.article_features(split).select(
        "article_idx",
        pl.col("ctr_smoothed").cast(pl.Float32),
        pl.col("clicks_decayed").cast(pl.Float32),
        pl.col("clicks").cast(pl.Float32).alias("prior_clicks"),
        pl.col("n_inview").cast(pl.Float32).alias("prior_inview"),
    )
    lf = lf.join(af, on="article_idx", how="left")

    uf = fs.user_features(split).select(
        "user_idx",
        pl.col("n_hist").cast(pl.Float32),
        pl.col("clicks_24h").cast(pl.Float32),
        pl.col("clicks_7d").cast(pl.Float32),
        pl.col("hours_since_last_click").cast(pl.Float32),
        pl.col("top_category_share").cast(pl.Float32),
        pl.col("n_categories").cast(pl.Float32),
        pl.col("category_entropy").cast(pl.Float32),
    )
    lf = lf.join(uf, on="user_idx", how="left")

    timing = article_timing(fs)
    lf = (
        lf.join(timing, on="article_idx", how="left")
        .with_columns(
            # hours between publication and the impression that showed it. News
            # decays in hours, not days -- A1 measured the median clicked EB-NeRD
            # article at 80 h -- so this is a first-order feature, not a nicety.
            ((pl.col("time") - pl.col("pub_time")).dt.total_seconds() / 3600.0)
            .cast(pl.Float32).alias("age_hours")
        )
    )

    prof = user_category_profile(fs, split, history_mode)
    lf = lf.join(prof, on=["user_idx", "category"], how="left")

    rec = category_recency(fs, split, halflife_hours, history_mode)
    lf = lf.join(rec, on=["user_idx", "category"], how="left")

    return (
        lf.with_columns(
            # a category the user has never clicked is a real zero, not a missing
            # value: the history is complete, so absence is evidence
            pl.col("cat_share").fill_null(0.0),
            pl.col("cat_clicks").fill_null(0),
            pl.col("cat_recency").fill_null(0.0),
            pl.col("cat_recency_share").fill_null(0.0),
            # an article nobody clicked before the cutoff likewise scored zero
            pl.col("clicks_decayed").fill_null(0.0),
            pl.col("prior_clicks").fill_null(0.0),
            pl.col("prior_inview").fill_null(0.0),
            # ...but a *user* with no prior activity is genuinely unknown, and
            # `n_hist` null vs 0 is the cold-start slice Q5 reports on. Left null.
        )
        .drop("pub_time", "subcategory")
        .collect(engine="streaming")
    )
