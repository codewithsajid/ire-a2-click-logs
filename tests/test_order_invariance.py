"""Results must not depend on the row order of the table they were computed from.

A feature-store rebuild can legitimately reorder rows. When it did, three things
moved that should not have: the random baseline (drawn from a stream indexed by
row position), every score tie in the metric harness (broken the same way), and
the bootstrap resample. All three now key on identity rather than position, and
these tests shuffle the input to prove it.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from newsrec.baselines import random_topk
from newsrec.evaluate import bootstrap, rank_metrics, topk_lists

METRICS = ("auc", "mrr", "ndcg@5", "ndcg@10")


def _pairs(n_imp=60, n_cand=12, seed=1) -> pl.DataFrame:
    """Impressions whose scores are mostly ties -- the case that exposed this.

    Candidates are unique within an impression, as they are in both real
    datasets (asserted by test_candidates_are_unique_within_an_impression):
    (imp, article_idx) is the key the tie-break hashes, so it has to be one.
    """
    rng = np.random.default_rng(seed)
    imp = np.repeat(np.arange(n_imp), n_cand)
    art = np.concatenate([rng.choice(500, size=n_cand, replace=False)
                          for _ in range(n_imp)]).astype(np.uint32)
    # a constant-scoring baseline plus a handful of distinct values: heavy ties
    score = np.where(rng.random(imp.size) < 0.8, 0.0, rng.integers(1, 4, imp.size) * 1.0)
    label = rng.random(imp.size) < 0.15
    return pl.DataFrame({"imp": imp.astype(np.uint32), "article_idx": art,
                         "score": score, "label": label})


def test_rank_metrics_survive_a_shuffle():
    p = _pairs()
    a = rank_metrics(p, ks=(5, 10)).sort("imp")
    shuffled = p.sample(fraction=1.0, shuffle=True, seed=7)
    b = rank_metrics(shuffled, ks=(5, 10)).sort("imp")
    for m in METRICS:
        # equal_nan: an impression with no negatives has an undefined AUC, and
        # "undefined in both" is agreement, not disagreement
        assert np.allclose(a[m].to_numpy(), b[m].to_numpy(), equal_nan=True), \
            f"{m} moved when rows were shuffled"


def test_topk_lists_survive_a_shuffle():
    p = _pairs()
    a = topk_lists(p, k=5).sort("imp")
    b = topk_lists(p.sample(fraction=1.0, shuffle=True, seed=7), k=5).sort("imp")
    assert a["rec"].to_list() == b["rec"].to_list()


def test_random_baseline_keys_on_the_user_not_the_row():
    uni = np.arange(3000, dtype=np.int32)
    uids = np.array([11, 4, 900, 37, 5, 62], dtype=np.uint32)
    a = random_topk(len(uids), uni, 40, user_ids=uids)
    perm = np.array([4, 1, 5, 0, 3, 2])
    b = random_topk(len(uids), uni, 40, user_ids=uids[perm])
    for i, src in enumerate(perm):
        assert set(a[src]) == set(b[i])


def test_random_baseline_is_uniform():
    uni = np.arange(1000, dtype=np.int32)
    users = np.arange(8000, dtype=np.uint32)
    picks = random_topk(len(users), uni, 50, user_ids=users)
    counts = np.bincount(picks.ravel(), minlength=len(uni))
    expected = len(users) * 50 / len(uni)
    sd = np.sqrt(expected * (1 - 50 / len(uni)))
    assert abs(counts.mean() - expected) < 1e-6
    assert counts.std() < 3 * sd, "selection is not close to uniform"
    assert all(len(set(row)) == 50 for row in picks[:200]), "sampled with replacement"


def test_bootstrap_survives_a_shuffle():
    rng = np.random.default_rng(3)
    v = rng.random(5000)
    assert bootstrap(v, n_boot=200) == bootstrap(rng.permutation(v), n_boot=200)


def test_candidates_are_unique_within_an_impression():
    """The precondition the tie-break relies on, checked against real stores.

    If an impression could list the same article twice, both rows would hash to
    the same jitter and the tie between them would fall back to row order --
    exactly the bug this module exists to prevent.
    """
    import pytest
    from newsrec.config import DATA_ROOT
    from newsrec.store import FeatureStore

    stores = sorted(p.parent for p in DATA_ROOT.glob("processed/*/*/manifest.json"))
    if not stores:
        pytest.skip("no feature store built yet")
    for path in stores:
        fs = FeatureStore(path.parent.name, path.name)
        for split in fs.splits:
            dupes = (fs.impressions(split)
                     .select((pl.col("candidates").list.len()
                              != pl.col("candidates").list.unique().list.len()).alias("d"))
                     .filter("d").select(pl.len()).collect().item())
            assert dupes == 0, f"{fs.dataset}/{fs.variant} {split}: {dupes} impressions repeat a candidate"
