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
          labelled: bool = True, time_cutoff: datetime | None = None) -> pl.DataFrame:
    """One row per candidate the platform actually showed, with its label.

    This is A1's `scripts/q4_eval.build_pairs` plus two things the re-ranker
    needs and the A1 rankers did not: the candidate's rendered position, and the
    per-impression context columns. `imp` is a row index over the impressions
    kept, not `impression_id` -- EB-NeRD's 200,000 beyond-accuracy rows all carry
    `impression_id = 0`, so the id does not identify an impression there.

    `max_impressions` subsamples impressions and never candidates within one: a
    partial candidate list would change what AUC and nDCG mean for that row.

    `time_cutoff` deletes everything at or after an instant, which is how
    `tests/test_behaviour_window.py` checks the boundary: assemble twice, once
    with the future present and once without, and require the surviving rows to
    be identical. Keyed on `src_row` rather than `imp`, because `imp` is a row
    index over whatever survived and is not stable across the two runs.
    """
    imp = fs.impressions(split)
    if time_cutoff is not None:
        imp = imp.filter(pl.col("time") < pl.lit(time_cutoff))
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


# ---------------------------------------------------------------- dwell time

def dwell_features(imp: pl.LazyFrame) -> pl.LazyFrame:
    """Dwell signals, as expanding means over strictly earlier impressions.

    Q1.2 asks for dwell time "if available". EB-NeRD has it -- `read_time` and
    `scroll_percentage` per impression -- and MIND has nothing of the kind, so
    this block is null there.

    The distinction that makes these usable is *whose* dwell and *when*. The
    `read_time` of the impression being scored describes a visit that is still in
    progress at the moment a ranker must choose what to show, which is why A1
    excluded it and why it stays in `SERVING_GREY`. The mean read time over that
    user's *earlier* impressions is a different quantity: it is fully determined
    before the current request arrives, it is cheap to keep in a feature store,
    and it says something a click count does not -- whether this is a reader or a
    skimmer.

    Two features, plus the count they are averaged over:

      * `prior_read_time`  -- the user's mean dwell over their earlier impressions
      * `prior_scroll`     -- the same for scroll depth, which EB-NeRD leaves null
                              on 70.4% of rows, so it is a sparse column by nature

    They cold-start at each split boundary rather than carrying across splits:
    the expanding mean is computed within the split being scored, so the first
    impression of every user in the test split has a null here. That is a real
    cost -- it is the same stale-boundary problem the article priors have -- and
    `scripts/q1_report.py` reports the resulting coverage rather than hiding it.

    The window is strictly earlier **in time**, not merely earlier in row order.
    That distinction is not pedantic: EB-NeRD stamps impressions to the second and
    a busy user produces several inside one, so a `shift(1)` over rows lets a
    visit that began at the same instant contribute its dwell to the row being
    scored. It is also non-deterministic -- which of two tied rows comes first is
    whatever the sort happened to do -- so the feature moved when unrelated rows
    were deleted. `tests/test_behaviour_window.py` caught exactly that.

    So the accumulation runs over distinct timestamps: rows sharing an instant all
    see the same prior window, and none of them sees any other.
    """
    names = imp.collect_schema().names()
    if "read_time" not in names:
        return imp.select("imp").with_columns(
            pl.lit(None, dtype=pl.Float32).alias("prior_read_time"),
            pl.lit(None, dtype=pl.Float32).alias("prior_scroll"),
            pl.lit(None, dtype=pl.UInt32).alias("prior_impressions"),
        )
    rows = imp.select("imp", "user_idx", "time", "read_time", "scroll_percentage")
    per_instant = (
        rows.group_by("user_idx", "time")
        .agg(
            pl.col("read_time").sum().alias("_rt"),
            pl.len().cast(pl.UInt32).alias("_n"),
            pl.col("scroll_percentage").fill_null(0.0).sum().alias("_sc"),
            pl.col("scroll_percentage").is_not_null().sum().cast(pl.UInt32).alias("_scn"),
        )
        .sort("user_idx", "time")
        .with_columns(
            pl.col("_rt").cum_sum().shift(1).over("user_idx").alias("_rt_sum"),
            pl.col("_n").cum_sum().shift(1).over("user_idx").alias("prior_impressions"),
            pl.col("_sc").cum_sum().shift(1).over("user_idx").alias("_sc_sum"),
            pl.col("_scn").cum_sum().shift(1).over("user_idx").alias("_sc_n"),
        )
        .select("user_idx", "time", "_rt_sum", "prior_impressions", "_sc_sum", "_sc_n")
    )
    return (
        rows.join(per_instant, on=["user_idx", "time"], how="left")
        .with_columns(
            (pl.col("_rt_sum") / pl.col("prior_impressions"))
            .cast(pl.Float32).alias("prior_read_time"),
            (pl.col("_sc_sum") / pl.col("_sc_n")).cast(pl.Float32).alias("prior_scroll"),
        )
        .select("imp", "prior_read_time", "prior_scroll",
                pl.col("prior_impressions").fill_null(0))
    )


def article_dwell(imp: pl.LazyFrame) -> pl.LazyFrame:
    """Mean dwell an article earned, over impressions that closed before this one.

    The article-side counterpart of `dwell_features`, and the direct analogue of
    the lecture's `doc_ctr_30d`: a behavioural statistic about the document,
    counted on a window that strictly precedes the example using it. Clickbait is
    exactly the case where click rate and dwell disagree, so an article-level
    dwell column is the cheapest defence against optimising for attention rather
    than satisfaction.

    EB-NeRD's `read_time` belongs to the impression, not to a named article, so
    it is attributed to whatever that impression clicked. For a near-single-click
    log (1.01 clicks per impression) that attribution is unambiguous on almost
    every row.

    Returned as a time-ordered event table for an as-of join, rather than as a
    per-article scalar: a scalar would have to be computed on some window, and
    every choice of window is either stale or leaky. The as-of join lets each
    impression read the value as it stood at that instant.

    One row per (article, instant), carrying the mean *including* everything at
    that instant. The as-of join that consumes it is then run with
    `allow_exact_matches=False`, so a row at time t reads the last instant
    strictly before t -- which is the mean over exactly the visits that had
    finished when the request arrived. Collapsing to one row per instant first is
    what makes the result independent of row order: EB-NeRD's second-resolution
    timestamps tie constantly, and an expanding mean over tied rows depends on how
    the sort broke them.
    """
    names = imp.collect_schema().names()
    if "read_time" not in names:
        return pl.LazyFrame(schema={"article_idx": pl.UInt32, "time": pl.Datetime("us"),
                                    "art_read_time": pl.Float32,
                                    "art_dwell_n": pl.UInt32})
    return (
        imp.select("clicked", "time", "read_time")
        .explode("clicked")
        .drop_nulls("clicked")
        .rename({"clicked": "article_idx"})
        .group_by("article_idx", "time")
        .agg(pl.col("read_time").sum().alias("_rt"), pl.len().cast(pl.UInt32).alias("_n"))
        .sort("article_idx", "time")
        .with_columns(
            pl.col("_rt").cum_sum().over("article_idx").alias("_sum"),
            pl.col("_n").cum_sum().over("article_idx").alias("art_dwell_n"),
        )
        .with_columns((pl.col("_sum") / pl.col("art_dwell_n"))
                      .cast(pl.Float32).alias("art_read_time"))
        .select("article_idx", "time", "art_read_time", "art_dwell_n")
        .sort("time")
    )


# ------------------------------------------------- rolling article statistics

def rolling_article_stats(imp: pl.LazyFrame, labelled: bool = True) -> pl.LazyFrame:
    """Per-article counters as they stood at each instant, inside the split.

    A1's `article_features` computes popularity once, on the window strictly
    before the split, and then holds it fixed for the whole split. On a news
    corpus that is a severe approximation: 84.2% of EB-NeRD's candidate slots and
    54.9% of MIND's have a prior click count of exactly zero, because the article
    did not exist when the window closed. That frozen prior is what put A1's MIND
    popularity submission at 0.4900 -- below chance -- and it is the single
    largest gap in the feature matrix.

    This is the same statistic kept rolling: at an impression at time t, how many
    times had this article been shown, and clicked, strictly before t. It is the
    lecture's `doc_ctr_30d` with the window ending at the request instead of at
    the split boundary, and it is what a real feature store actually holds.

    **The two counters have different availability, and the difference decides
    what may ship.**

      * `roll_inview` counts *exposures*. Candidate lists are published for the
        unlabelled Codabench test sets, so this is computable there, and at
        serving time trivially so.
      * `roll_clicks` / `roll_ctr` count *clicks*. In production these are
        available -- a click that happened an hour ago is in the log. On the
        Codabench split they are not, because the labels are withheld by
        construction.

    So a model that leans on `roll_clicks` reports what a production system could
    do and cannot be submitted as-is. Both models are built and both numbers are
    reported; `FAMILIES["rolling"]` is the dividing line, and the design note
    argues the distinction rather than quietly picking one.

    Counters are cumulative *including* the current instant, and consumed through
    an as-of join with `allow_exact_matches=False`, which is what makes the read
    strictly prior. One row per (article, instant) keeps the join independent of
    how ties were sorted.
    """
    inview = (
        imp.select("candidates", "time")
        .explode("candidates")
        .drop_nulls("candidates")
        .rename({"candidates": "article_idx"})
        .group_by("article_idx", "time")
        .agg(pl.len().cast(pl.UInt32).alias("_v"))
    )
    if labelled:
        clicks = (
            imp.select("clicked", "time")
            .explode("clicked")
            .drop_nulls("clicked")
            .rename({"clicked": "article_idx"})
            .group_by("article_idx", "time")
            .agg(pl.len().cast(pl.UInt32).alias("_c"))
        )
        ev = inview.join(clicks, on=["article_idx", "time"], how="left").with_columns(
            pl.col("_c").fill_null(0))
    else:
        ev = inview.with_columns(pl.lit(0, dtype=pl.UInt32).alias("_c"))

    return (
        ev.sort("article_idx", "time")
        .with_columns(
            pl.col("_v").cum_sum().over("article_idx").alias("roll_inview"),
            pl.col("_c").cum_sum().over("article_idx").alias("roll_clicks"),
        )
        .with_columns(
            # smoothed towards zero rather than towards the corpus mean: early in
            # a split the denominator is tiny, and an unsmoothed 1/1 would read as
            # a perfect article. 20 is the exposure count at which the observed
            # rate gets half the weight.
            (pl.col("roll_clicks") / (pl.col("roll_inview") + 20.0))
            .cast(pl.Float32).alias("roll_ctr"),
            # hours since the article was first seen *in this split*, which is the
            # freshness signal `age_hours` cannot give on MIND
            pl.col("time").min().over("article_idx").alias("_first"),
        )
        .with_columns(
            ((pl.col("time") - pl.col("_first")).dt.total_seconds() / 3600.0)
            .cast(pl.Float32).alias("roll_age_hours")
        )
        .select("article_idx", "time", "roll_inview", "roll_clicks", "roll_ctr",
                "roll_age_hours")
        .sort("time")
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
                    mode: str = "shipped", newest_last: bool = True) -> pl.LazyFrame:
    """Exponentially-decayed weight of each (user, previously-clicked article).

    Q1.1 asks for a recency-weighted history. EB-NeRD timestamps every history
    click -- verified: 15,143 of 15,143 users are strictly ascending, and the
    store holds one timestamp per history article, 2,426,247 for 2,426,247 -- so
    there the weight is a real half-life in hours.

    MIND ships no history times at all. `behaviors.tsv` has five fields and the
    fourth is a bare list of ids, so the decay can only run over *rank*. Which
    end of that list is recent is an **assumption, not a measurement**, and the
    data cannot settle it: MIND's history is a fixed per-user snapshot that never
    varies within a bundle (0 of 698,365 users in MINDlarge_train have more than
    one distinct history) and is byte-identical between train and dev (210,990 of
    210,990 users at large scale). There is no second observation to difference
    against.

    `newest_last` names the assumption instead of burying it, and
    `scripts/q1_order_ablation.py` measures what it is worth by training both
    ways. An assumption that cannot be verified can still be priced.

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
                (((pl.col("_i").max().over("user_idx") - pl.col("_i")) if newest_last
                  else pl.col("_i")).cast(pl.Float32)).alias("_age_rank")
            )
            .with_columns((0.5 ** (pl.col("_age_rank") / 10.0)).cast(pl.Float32).alias("hist_w"))
            .drop("_i", "_age_rank")
        )
    return e.select("user_idx", "article_idx", "hist_w")


def category_recency(fs: FeatureStore, split: str, halflife_hours: float = 24.0,
                     mode: str = "shipped", newest_last: bool = True) -> pl.LazyFrame:
    """Decayed click mass per (user, category) -- "how warm is this topic for them".

    The un-decayed version of this is `user_category_profile.cat_share`. Keeping
    both lets the ablation separate *what* a user reads from *what they read
    lately*, which on a news corpus with an 80-hour median article age are not
    the same question.
    """
    cats = fs.articles().with_row_index("idx").select(
        pl.col("idx").cast(pl.UInt32).alias("article_idx"), "category")
    return (
        history_recency(fs, split, halflife_hours, mode, newest_last)
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
             halflife_hours: float = 24.0, labelled: bool = True,
             time_cutoff: datetime | None = None,
             newest_last: bool = True) -> pl.DataFrame:
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
    p = pairs(fs, split, max_impressions=max_impressions, seed=seed,
              labelled=labelled, time_cutoff=time_cutoff)
    lf = p.lazy()
    src = fs.impressions(split)
    if time_cutoff is not None:
        src = src.filter(pl.col("time") < pl.lit(time_cutoff))

    imp_level = p.lazy().select(
        ["imp", "user_idx", "time",
         *_present(p.lazy(), ("session_id", "read_time", "scroll_percentage"))]
    ).unique(subset=["imp"])
    lf = lf.join(session_features(imp_level), on="imp", how="left")
    lf = lf.join(dwell_features(imp_level), on="imp", how="left")

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

    # Article dwell, read as it stood at the instant of this impression. An as-of
    # join rather than a grouped aggregate: a per-article scalar would have to be
    # computed on some fixed window, and every such window is either stale by the
    # end of the split or leaky by the start of it.
    lf = lf.sort("time")
    events = article_dwell(src)
    if events.collect_schema()["art_read_time"] != pl.Null:
        lf = lf.join_asof(events, on="time", by="article_idx", strategy="backward",
                          allow_exact_matches=False)

    # Rolling exposure/click counters, read as of the instant of the impression.
    # Same join discipline: strictly before, one row per (article, instant).
    lf = lf.join_asof(rolling_article_stats(src, labelled=labelled),
                      on="time", by="article_idx", strategy="backward",
                      allow_exact_matches=False)

    prof = user_category_profile(fs, split, history_mode)
    lf = lf.join(prof, on=["user_idx", "category"], how="left")

    rec = category_recency(fs, split, halflife_hours, history_mode, newest_last)
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
