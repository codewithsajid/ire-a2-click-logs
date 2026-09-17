"""Q1.4 / Q9: the behaviour-window boundary, asserted rather than asserted-to.

A1's `test_no_leakage.py` checks the feature *store*: splits ordered in time,
every derived table stamped `computed_through` earlier than the split it serves,
no exposure counts crossing a boundary. Those checks work at split granularity,
which is the right grain for a table computed once per split.

The features A2 adds are finer than a split. `session_rank` counts the
impressions of this session that came first; `prior_read_time` averages the
user's earlier visits; `art_read_time` reads an article's dwell as it stood at
this instant. A split-level stamp cannot see whether any of those accidentally
read a row from thirty seconds in the future.

So this file tests the property directly, by the only construction that cannot be
argued with: **compute the features, delete the future, compute them again, and
require the surviving rows to be bit-identical.** If any feature reads an event
at or after the impression it describes, deleting that event changes its value
and the comparison fails. Nothing here trusts a docstring.

The suite is deliberately cheap enough to run on every commit -- it uses the demo
bundle where one exists and subsamples otherwise -- because a leakage test that
only runs before submission is a leakage test that runs once.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from newsrec.behaviour import (SERVING_GREY, article_dwell, assemble,
                               dwell_features, pairs, session_features)
from newsrec.store import FeatureStore

# The columns whose whole job is to describe the current request. They are
# expected to change when the current row changes and are excluded from the
# time-travel comparison -- they are quarantined by `SERVING_GREY`, not by this
# test, and `test_grey_columns_are_declared` is what keeps the two in step.
CURRENT_ROW = set(SERVING_GREY) | {
    "imp", "src_row", "user_idx", "time", "article_idx", "label",
    "n_candidates", "category", "device_type", "session_id",
    "is_subscriber", "is_sso_user",
}


def _store(dataset: str) -> FeatureStore:
    for variant in ("demo", "small"):
        try:
            return FeatureStore(dataset, variant)
        except FileNotFoundError:
            continue
    pytest.skip(f"no built store for {dataset}")


@pytest.fixture(scope="module", params=["ebnerd", "mind"])
def store(request) -> FeatureStore:
    return _store(request.param)


# ------------------------------------------------- the time-travel property

def _truncate(imp: pl.LazyFrame, cutoff) -> pl.LazyFrame:
    return imp.filter(pl.col("time") < pl.lit(cutoff))


def test_session_features_ignore_later_impressions(store):
    """Delete the back half of the split; the front half's features must not move.

    This is the assignment's boundary condition in its sharpest form. Session
    position is causal -- how many impressions came first -- while session
    *length* is not, and the difference is invisible in the code until the
    future is actually removed.
    """
    imp = store.impressions("train").select(
        "src_row", "user_idx", "time",
        *[c for c in ("session_id",) if c in store.impressions("train").collect_schema().names()]
    ).with_row_index("imp")

    t0, t1 = store.bounds("train")
    cutoff = t0 + (t1 - t0) / 2

    full = session_features(imp).collect().sort("imp")
    part = session_features(_truncate(imp, cutoff)).collect().sort("imp")

    keep = part["imp"]
    a = full.filter(pl.col("imp").is_in(keep)).sort("imp")
    b = part.sort("imp")
    assert a.height == b.height > 0, "truncation left nothing to compare"
    for col in ("session_rank", "session_seconds", "secs_since_prev"):
        assert a[col].equals(b[col]), (
            f"{col} changed when later impressions were deleted -- it reads the future")


def test_dwell_features_ignore_later_impressions(store):
    """The user's prior dwell must be a function of their past alone."""
    names = store.impressions("train").collect_schema().names()
    if "read_time" not in names:
        pytest.skip("no dwell columns on this dataset")

    imp = store.impressions("train").select(
        "user_idx", "time", "read_time", "scroll_percentage").with_row_index("imp")
    t0, t1 = store.bounds("train")
    cutoff = t0 + (t1 - t0) / 2

    full = dwell_features(imp).collect().sort("imp")
    part = dwell_features(_truncate(imp, cutoff)).collect().sort("imp")

    a = full.filter(pl.col("imp").is_in(part["imp"])).sort("imp")
    assert a.height == part.height > 0
    for col in ("prior_read_time", "prior_scroll", "prior_impressions"):
        assert a[col].equals(part[col]), f"{col} reads impressions from the future"


def test_article_dwell_is_strictly_backward(store):
    """An article's dwell at time t must average only visits that closed before t."""
    names = store.impressions("train").collect_schema().names()
    if "read_time" not in names:
        pytest.skip("no dwell columns on this dataset")

    ev = article_dwell(store.impressions("train")).collect()
    if ev.is_empty():
        pytest.skip("no click events with dwell")
    # The event table carries the mean *including* its own instant; it is the
    # as-of join's `allow_exact_matches=False` that makes the read strictly
    # prior. So the property to check here is that the table is well formed and
    # one row per (article, instant) -- ties collapsed is what makes the
    # downstream join independent of row order.
    assert ev.select(["article_idx", "time"]).is_duplicated().sum() == 0, (
        "more than one row per (article, instant) -- the as-of join result "
        "would depend on which one the sort put last")
    assert (ev["art_dwell_n"].to_numpy() >= 1).all()
    assert ev["art_read_time"].is_null().sum() == 0

    # and the property that matters, at the join: an article's dwell on the
    # first impression that ever showed it must be null, because nothing had
    # finished reading it yet
    df = assemble(store, "train", max_impressions=6000, seed=0)
    firsts = df.sort("time").group_by("article_idx").first()
    assert firsts["art_read_time"].is_null().any(), (
        "no article is missing a prior-dwell value at its first appearance -- "
        "the join is matching the current instant")


# --------------------------------------------------- whole-matrix property

def test_assembled_features_survive_deleting_the_future(store):
    """End-to-end: assemble twice, once with the future and once without.

    This is the test the whole file exists for. The second run cannot see any
    impression at or after the cutoff, so a feature that reads one will differ
    between the runs on rows that both produced. Keyed on `(src_row,
    article_idx)`, which identifies a candidate slot in the raw file and is
    therefore stable across the two runs -- `imp` is a row index over survivors
    and is not.

    Columns in `CURRENT_ROW` describe the request being scored rather than
    anything accumulated, so they are expected to be equal trivially and are
    excluded from the comparison rather than silently passing it.
    """
    t0, t1 = store.bounds("train")
    cutoff = t0 + (t1 - t0) * 0.6

    full = assemble(store, "train")
    part = assemble(store, "train", time_cutoff=cutoff)
    assert part.height > 0, "truncation removed everything"
    assert part.height < full.height, "truncation removed nothing -- cutoff is wrong"

    key = ["src_row", "article_idx"]
    joined = part.join(full, on=key, how="inner", suffix="_full")
    assert joined.height == part.height, "truncated rows are missing from the full run"

    compared = []
    for col, dtype in zip(part.columns, part.dtypes):
        if col in CURRENT_ROW or col in key or f"{col}_full" not in joined.columns:
            continue
        a, b = joined[col], joined[f"{col}_full"]
        if dtype.is_float():
            # NaN != NaN, so compare the null/NaN masks and the finite values apart
            va, vb = a.to_numpy().astype(float), b.to_numpy().astype(float)
            na, nb = np.isnan(va), np.isnan(vb)
            assert np.array_equal(na, nb), f"{col}: null pattern moved when the future was deleted"
            # Relative, not absolute. Several of these columns are float32 sums
            # over a group, and float32 addition is not associative: deleting
            # unrelated rows changes how polars partitions the aggregation, which
            # moves the last digit. Measured on EB-NeRD demo, the worst such
            # column (`cat_recency`) differs by 8.3e-7 relative and 0 of 150,531
            # rows exceed 1e-6 -- while a column that genuinely read the future
            # would move by a great deal more than its own epsilon.
            d = np.abs(va[~na] - vb[~nb])
            rel = d / np.maximum(np.abs(vb[~nb]), 1e-12)
            assert rel.max(initial=0.0) < 1e-5, (
                f"{col} reads an impression at or after the row it describes "
                f"(max relative change {rel.max(initial=0.0):.2e})")
        else:
            assert a.equals(b), f"{col} reads an impression at or after the row it describes"
        compared.append(col)

    # The rolling counters are integer-valued accumulations with no float
    # reassociation to hide behind, so they get the strict test: bit-identical or
    # the window boundary is broken.
    for col in ("roll_inview", "roll_clicks", "art_dwell_n", "prior_impressions",
                "session_rank"):
        if f"{col}_full" not in joined.columns:
            continue
        x = joined[col].to_numpy().astype(float)
        y = joined[f"{col}_full"].to_numpy().astype(float)
        m = ~(np.isnan(x) | np.isnan(y))
        assert np.array_equal(x[m], y[m]), (
            f"{col} is a counter and it moved when the future was deleted")

    assert len(compared) >= 10, f"only compared {compared} -- the test is not covering the matrix"

    # an infinity slips through a ratio feature and then becomes a split threshold
    for col, dtype in zip(full.columns, full.dtypes):
        if dtype.is_float():
            v = full[col].to_numpy().astype(float)
            assert not np.isinf(v[~np.isnan(v)]).any(), f"{col} contains an infinity"


def test_freshness_is_never_negative(store):
    """`age_hours` is impression time minus publication; a negative means the
    catalogue timestamp came from after the article was shown."""
    df = assemble(store, "train", max_impressions=4000, seed=0)
    age = df["age_hours"].drop_nulls().to_numpy()
    assert age.size, "no freshness values at all"
    # a handful of articles carry a publication timestamp slightly after their
    # first impression (editorial re-publication); the assertion is that this is
    # rare and small, not that it never happens
    bad = age[age < 0]
    assert bad.size / age.size < 0.01, (
        f"{bad.size / age.size:.2%} of rows are shown before they exist")
    if bad.size:
        assert bad.min() > -72, f"an article was shown {-bad.min():.0f}h before publication"


# ------------------------------------------------------ the quarantine list

def test_grey_columns_are_declared(store):
    """Every column describing the in-progress visit is in `SERVING_GREY`.

    Q9 asks for metrics with and without features unavailable at serving time,
    which only means something if the set is written down somewhere the model
    reads. This pins the list so that adding a column like `read_time` to the
    matrix without classifying it fails here rather than in the design note.
    """
    df = assemble(store, "train", max_impressions=500, seed=0)
    suspicious = {c for c in df.columns
                  if c in ("read_time", "scroll_percentage", "position", "position_frac")}
    assert suspicious <= set(SERVING_GREY), (
        f"{suspicious - set(SERVING_GREY)} describe the current visit but are not quarantined")


def test_shipped_model_excludes_grey_features(store):
    """The default feature set must not contain a quarantined column."""
    from newsrec.rerank import SHIPPED, feature_names
    df = assemble(store, "train", max_impressions=500, seed=0)
    feats = set(feature_names(df, SHIPPED))
    assert not (feats & set(SERVING_GREY)), (
        f"shipped model reads {feats & set(SERVING_GREY)}")


def test_prior_dwell_is_not_the_current_dwell(store):
    """`prior_read_time` must not simply echo this impression's `read_time`.

    A shift(1) that is applied to the wrong frame ordering produces a column that
    correlates near-perfectly with the current row -- which looks like a strong
    feature and is the label's own visit.
    """
    names = store.impressions("train").collect_schema().names()
    if "read_time" not in names:
        pytest.skip("no dwell columns on this dataset")
    df = assemble(store, "train", max_impressions=8000, seed=0)
    both = df.select("read_time", "prior_read_time").drop_nulls()
    if both.height < 100:
        pytest.skip("too few rows with both values")
    r = np.corrcoef(both["read_time"].to_numpy(), both["prior_read_time"].to_numpy())[0, 1]
    assert abs(r) < 0.9, f"prior dwell correlates {r:.3f} with the current visit"


def test_unknown_users_are_not_aliased_onto_someone_else(store):
    """A user missing from the history table must score as unknown, not as
    whoever happens to sit in the last row of the lookup array.

    The natural way to write the user_idx -> history-row map is a dense array
    sized by the largest id in the history table, indexed with a clip. The clip
    is the bug: an id past the end lands on the final slot, which belongs to a
    real user, and the pair is then scored with that stranger's query vector.
    """
    from newsrec.rerank import add_matching_features
    from newsrec.behaviour import pairs as build_pairs

    p = build_pairs(store, "train", max_impressions=300, seed=0)
    # forge a user id beyond anything the history table knows about
    hmax = int(store.history("train").select(pl.col("user_idx").max()).collect().item())
    forged = p.with_columns(
        pl.when(pl.int_range(pl.len()) < 50)
        .then(pl.lit(hmax + 10_000, dtype=pl.UInt32))
        .otherwise(pl.col("user_idx")).alias("user_idx"))

    emb = "contrastive" if store.dataset == "ebnerd" else "sentence-transformers/all-MiniLM-L6-v2"
    out = add_matching_features(store, "train", forged, emb, 2.0, 1.0)
    ghost = out.head(50)
    for col in ("bm25", "emb_cos", "emb_max", "emb_recent"):
        v = ghost[col].to_numpy()
        assert np.allclose(v, 0.0), (
            f"{col} is non-zero for a user with no history -- the lookup aliased "
            f"them onto another user")
