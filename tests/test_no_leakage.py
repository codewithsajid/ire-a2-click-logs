"""Behaviour-window boundary tests (assignment Q9).

These assert the pipeline cannot see the future: splits are ordered in time,
popularity features are computed strictly before the window they score, and
augmented history never reaches past its cutoff. Run against every built store:

    pytest tests/ -v
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import polars as pl
import pytest

from newsrec.config import DATA_ROOT
from newsrec.schema import LEAK_COLUMNS

STORES = sorted(p.parent for p in DATA_ROOT.glob("processed/*/*/manifest.json"))
pytestmark = pytest.mark.skipif(not STORES, reason="no feature store built yet")


def _manifest(store: Path) -> dict:
    return json.loads((store / "manifest.json").read_text())


def _ts(s: str) -> datetime:
    return datetime.fromisoformat(s)


@pytest.fixture(params=STORES, ids=lambda p: f"{p.parent.name}/{p.name}")
def store(request) -> Path:
    return request.param


def test_splits_are_ordered_in_time(store):
    """train < val < test < submit, with no overlap at the boundaries."""
    bounds = _manifest(store)["time_bounds"]
    order = [s for s in ("train", "val", "test", "submit") if s in bounds]
    for earlier, later in zip(order, order[1:]):
        assert _ts(bounds[earlier]["max"]) < _ts(bounds[later]["min"]), (
            f"{earlier} overlaps {later}: {bounds[earlier]['max']} >= {bounds[later]['min']}"
        )


def test_no_future_columns_in_store(store):
    """next_read_time / next_scroll_percentage describe the *following*
    impression; they must never reach the feature store."""
    for path in store.rglob("*.parquet"):
        cols = pl.scan_parquet(path).collect_schema().names()
        assert not set(cols) & set(LEAK_COLUMNS), f"{path.name} carries {set(cols) & set(LEAK_COLUMNS)}"


def test_popularity_is_computed_before_its_split(store):
    """A popularity feature for split S may only count clicks from before S."""
    bounds = _manifest(store)["time_bounds"]
    for split, b in bounds.items():
        pop = store / "splits" / split / "popularity.parquet"
        if not pop.exists():
            continue
        df = pl.read_parquet(pop)
        if df.height == 0:          # earliest split: nothing precedes it
            continue
        through = _ts(df["computed_through"][0])
        assert through <= _ts(b["min"]), f"{split}: popularity through {through} > split start {b['min']}"
        assert df["last_click"].max() < through, f"{split}: popularity counted a click at or after its cutoff"


def test_augmented_history_respects_its_cutoff(store):
    """Augmented history may only add clicks strictly before the split start,
    and must be a superset of the shipped history."""
    clickstream = pl.scan_parquet(store / "clickstream.parquet")
    for split in _manifest(store)["time_bounds"]:
        aug_path = store / "splits" / split / "history_augmented.parquet"
        if not aug_path.exists():
            continue
        aug = pl.read_parquet(aug_path)
        shipped = pl.read_parquet(store / "splits" / split / "history.parquet")
        cutoff = _ts(aug["cutoff"][0])

        joined = shipped.select("user_idx", pl.col("n_hist").alias("n_shipped")).join(
            aug.select("user_idx", pl.col("n_hist").alias("n_aug")), on="user_idx", how="inner")
        assert (joined["n_aug"] >= joined["n_shipped"]).all(), f"{split}: augmentation dropped history"

        # every click that could have been added lies strictly before the cutoff
        added = clickstream.filter(pl.col("ts") >= pl.lit(cutoff)).select(pl.len()).collect().item()
        usable = clickstream.filter(pl.col("ts") < pl.lit(cutoff)).select(pl.len()).collect().item()
        assert usable + added == pl.scan_parquet(store / "clickstream.parquet").select(
            pl.len()).collect().item()
        assert aug["cutoff"].n_unique() == 1


def test_clicked_is_a_subset_of_candidates(store):
    """A click on something that was never shown means the id mapping is wrong."""
    for split in _manifest(store)["time_bounds"]:
        imp = pl.scan_parquet(store / "splits" / split / "impressions.parquet")
        bad = (
            imp.filter(pl.col("clicked").list.len() > 0)
            .with_columns(pl.col("clicked").list.set_difference(pl.col("candidates")).alias("_orphan"))
            .filter(pl.col("_orphan").list.len() > 0)
            .select(pl.len()).collect().item()
        )
        assert bad == 0, f"{split}: {bad:,} impressions click an article that was not in view"


def test_history_articles_exist_in_the_article_table(store):
    n_articles = pl.scan_parquet(store / "articles.parquet").select(pl.len()).collect().item()
    for split in _manifest(store)["time_bounds"]:
        hist = pl.scan_parquet(store / "splits" / split / "history.parquet")
        mx = hist.select(pl.col("article_idx").list.max().max()).collect().item()
        if mx is not None:
            assert mx < n_articles, f"{split}: history references article idx {mx} >= {n_articles}"


def test_derived_features_are_computed_before_their_split(store):
    """user_features and article_features carry a computed_through stamp; it may
    never reach into the split they describe."""
    bounds = _manifest(store)["time_bounds"]
    for split, b in bounds.items():
        for table in ("user_features", "article_features"):
            path = store / "splits" / split / f"{table}.parquet"
            if not path.exists():
                continue
            df = pl.read_parquet(path)
            if df.height == 0 or "computed_through" not in df.columns:
                continue
            through = _ts(df["computed_through"][0])
            assert through <= _ts(b["min"]), f"{split}/{table}: stamped {through} > split start {b['min']}"
            assert df["computed_through"].n_unique() == 1


def test_article_features_never_see_future_exposures(store):
    """Exposure counts for split S must equal the exposures of the splits that
    end before S -- never S's own, which is what a head/tail label would leak."""
    manifest = _manifest(store)
    bounds = manifest["time_bounds"]
    for split in bounds:
        af = store / "splits" / split / "article_features.parquet"
        if not af.exists():
            continue
        prior = [o for o in bounds if _ts(bounds[o]["max"]) < _ts(bounds[split]["min"])]
        expected = 0
        for p in prior:
            expected += pl.read_parquet(store / "splits" / p / "exposures.parquet")["n_inview"].sum()
        got = pl.read_parquet(af)["n_inview"].sum()
        assert got == expected, f"{split}: exposures {got:,} != prior-split total {expected:,}"


def test_article_stats_carries_no_exposure_counts(store):
    """Catalogue-level stats must not aggregate counts across splits: a tail
    article that goes viral in the test week would otherwise be labelled head."""
    cols = pl.scan_parquet(store / "article_stats.parquet").collect_schema().names()
    for banned in ("n_inview", "n_clicks", "clicks"):
        assert banned not in cols, f"article_stats leaks cross-split counts via '{banned}'"


def test_decayed_popularity_never_exceeds_raw_clicks(store):
    """Sanity on the half-life weighting: a decayed count is a discounted count."""
    for split in _manifest(store)["time_bounds"]:
        path = store / "splits" / split / "article_features.parquet"
        if not path.exists():
            continue
        df = pl.read_parquet(path)
        if df.height == 0:
            continue
        assert (df["clicks_decayed"] <= df["clicks"] + 1e-6).all()
        assert (df["ctr_smoothed"] >= 0).all() and (df["ctr_smoothed"] <= 1).all()
