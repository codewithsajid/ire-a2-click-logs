"""Our metrics must equal the ones the leaderboards actually compute.

Q4 asks for "the official evaluation metrics". Both graders score with
`ebrec.evaluation.metrics._ranking`, which is vendored in external/, and AUC is
sklearn's. So the test is not a reimplementation of the formulae -- it runs the
graders' own functions on the same inputs and demands equality.

This caught two real defects. MRR was the reciprocal rank of the *first*
relevant item; the official metric is the mean of 1/rank over *every* relevant
item, so on MIND (28.8% multi-click impressions) ours was materially optimistic.
AUC broke ties at random, which is unbiased but noisy -- sklearn gives a
constant-scoring baseline exactly 0.5, ours gave 0.5 +/- 0.5 per impression.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from newsrec.evaluate import rank_metrics

_BENCH = Path("external/ebnerd-benchmark/src")
if _BENCH.exists():
    sys.path.insert(0, str(_BENCH))
official = pytest.importorskip("ebrec.evaluation.metrics._ranking",
                               reason="EB-NeRD benchmark not vendored")
sklearn_metrics = pytest.importorskip("sklearn.metrics")


def _impressions(n_imp: int, seed: int, tied: bool = False, p_click: float = 0.25):
    """Impressions with a healthy share of multi-click cases -- where MRR differs."""
    rng = np.random.default_rng(seed)
    rows = []
    for imp in range(n_imp):
        n = int(rng.integers(5, 25))
        y = (rng.random(n) < p_click).astype(int)
        if y.sum() == 0:
            y[rng.integers(n)] = 1
        if y.sum() == n:
            y[rng.integers(n)] = 0
        s = np.zeros(n) if tied else rng.random(n)
        rows.append((imp, y, s))
    pairs = pl.DataFrame({
        "imp": np.concatenate([[i] * len(y) for i, y, _ in rows]).astype(np.uint32),
        "article_idx": np.concatenate([np.arange(len(y)) for _, y, _ in rows]).astype(np.uint32),
        "label": np.concatenate([y for _, y, _ in rows]).astype(bool),
        "score": np.concatenate([s for _, _, s in rows]),
    })
    return rows, pairs


def test_mrr_matches_the_graders_implementation():
    rows, pairs = _impressions(300, seed=0)
    assert sum(y.sum() > 1 for _, y, _ in rows) > 100, "fixture must exercise multi-click"
    ours = rank_metrics(pairs, ks=(5, 10)).sort("imp")["mrr"].to_numpy()
    theirs = np.array([official.mrr_score(y, s) for _, y, s in rows])
    assert np.allclose(ours, theirs, atol=1e-9)


def test_ndcg_matches_the_graders_implementation():
    rows, pairs = _impressions(300, seed=0)
    got = rank_metrics(pairs, ks=(5, 10)).sort("imp")
    for k in (5, 10):
        theirs = np.array([official.ndcg_score(y, s, k=k) for _, y, s in rows])
        assert np.allclose(got[f"ndcg@{k}"].to_numpy(), theirs, atol=1e-9), f"ndcg@{k}"


def test_auc_matches_sklearn():
    rows, pairs = _impressions(400, seed=1)
    ours = rank_metrics(pairs, ks=(5, 10)).sort("imp")["auc"].to_numpy()
    theirs = np.array([sklearn_metrics.roc_auc_score(y, s) for _, y, s in rows])
    assert np.allclose(ours, theirs, atol=1e-9)


def test_auc_of_a_constant_ranker_is_exactly_one_half():
    """The case random tie-breaking got wrong: no information means AUC = 0.5."""
    rows, pairs = _impressions(400, seed=1, tied=True)
    ours = rank_metrics(pairs, ks=(5, 10)).sort("imp")["auc"].to_numpy()
    assert np.allclose(ours, 0.5, atol=1e-12), "a constant score must score exactly 0.5"


def test_mrr_first_is_the_documented_variant():
    """`mrr_first` answers a different question and must not silently equal MRR."""
    rows, pairs = _impressions(300, seed=0)
    got = rank_metrics(pairs, ks=(5, 10)).sort("imp")
    mrr, first = got["mrr"].to_numpy(), got["mrr_first"].to_numpy()
    assert np.all(first >= mrr - 1e-12), "the first hit can only rank better than the average"
    multi = got["n_pos"].to_numpy() > 1
    assert np.allclose(mrr[~multi], first[~multi]), "they must agree on single-click impressions"
    assert not np.allclose(mrr[multi], first[multi]), "and differ on multi-click ones"
