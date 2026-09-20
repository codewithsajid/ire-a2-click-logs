"""The beyond-accuracy metrics, and the truncation bug that hid inside one.

Q5 asks for a bootstrap CI on *every* reported metric. Diversity and novelty are
means over impressions and bootstrap like the accuracy metrics; coverage is a
union and needs its own resampling. The first test below is the regression: the
pooled diversity estimator stacks lists into a matrix, which silently truncates
every list to the shortest one in the dataset, so on MIND -- whose shortest
in-impression list has two candidates -- it reported ILD@2 under the name ILD@10.
"""
import numpy as np
import pytest

from newsrec.evaluate import (coverage, coverage_ci, intra_list_diversity,
                              intra_list_diversity_per_list, novelty,
                              novelty_per_list)


def _emb(n: int, d: int = 16, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=(n, d)).astype(np.float32)


def test_pooled_and_per_list_agree_when_lists_are_equal_length():
    """With no ragged lists there is nothing to truncate, so the two coincide."""
    emb = _emb(50)
    rec = [np.arange(i, i + 10) for i in range(0, 30, 10)]
    pooled = intra_list_diversity(rec, emb)
    per = intra_list_diversity_per_list(rec, emb)
    assert np.isclose(pooled, np.nanmean(per), atol=1e-5)


def test_pooled_estimator_truncates_to_the_shortest_list():
    """One short list drags the pooled value onto a different quantity.

    This is the defect: a single 2-item list among 10-item lists makes the
    pooled estimator report the diversity of everyone's top 2. The per-list
    estimator is unaffected because each list keeps its own length.
    """
    # A ranked list is not an exchangeable bag: a ranker puts the items it
    # thinks alike at the top, so a list's first two entries are more similar to
    # each other than its ten entries are on average. That structure is exactly
    # what truncation destroys, so the fixture has to carry it -- random vectors
    # in high dimensions are near-orthogonal and would hide the bug.
    dim = 8
    emb = np.zeros((44, dim), dtype=np.float32)
    for li in range(4):                      # four lists of ten
        base = np.zeros(dim, dtype=np.float32)
        base[li % dim] = 1.0
        for pos in range(10):
            v = base.copy()
            if pos >= 2:                     # the tail of the list fans out
                v[(li + pos) % dim] += 1.0
            emb[li * 10 + pos] = v
    emb[40:42] = emb[0:2]                    # the short list mirrors a top-2

    long_lists = [np.arange(li * 10, li * 10 + 10) for li in range(4)]
    per_long = np.nanmean(intra_list_diversity_per_list(long_lists, emb))

    ragged = long_lists + [np.array([40, 41])]
    pooled = intra_list_diversity(ragged, emb)
    per = np.nanmean(intra_list_diversity_per_list(ragged, emb))

    # the per-list mean barely moves -- one more list among five
    assert abs(per - per_long) < 0.15
    # the pooled one now reports the diversity of everyone's top 2 instead
    assert pooled < per - 0.2


def test_diversity_bounds_are_the_cosine_bounds():
    """Identical items have zero diversity; orthogonal items have one."""
    same = np.tile(np.array([[1.0, 0.0, 0.0]], dtype=np.float32), (4, 1))
    assert np.allclose(intra_list_diversity_per_list([np.arange(4)], same), 0.0,
                       atol=1e-5)
    orth = np.eye(4, dtype=np.float32)
    assert np.allclose(intra_list_diversity_per_list([np.arange(4)], orth), 1.0,
                       atol=1e-5)


def test_lists_too_short_to_have_a_pair_are_undefined_not_zero():
    """A one-item list is not minimally diverse -- it has no pair at all."""
    emb = _emb(10)
    out = intra_list_diversity_per_list([np.array([0]), np.array([]), np.arange(3)], emb)
    assert np.isnan(out[0]) and np.isnan(out[1]) and not np.isnan(out[2])


def test_novelty_per_list_mean_reproduces_the_scalar():
    pop = np.full(20, 0.05, dtype=np.float64)
    pop[0] = 0.9
    rec = [np.array([0, 1, 2]), np.array([3, 4]), np.array([5])]
    assert np.isclose(novelty(rec, pop), np.nanmean(novelty_per_list(rec, pop)),
                      atol=1e-9)


def test_novelty_rewards_the_rarer_item():
    """Self-information is monotone in rarity, which is the whole point."""
    pop = np.array([0.5, 0.5, 0.001, 0.001], dtype=np.float64)
    common = novelty_per_list([np.array([0, 1])], pop)[0]
    rare = novelty_per_list([np.array([2, 3])], pop)[0]
    assert rare > common


def test_coverage_ci_is_an_interval_inside_the_unit_range():
    rec = [np.array([i, i + 1, i + 2]) for i in range(0, 60, 3)]
    point = coverage(rec, catalogue=100)
    lo, hi = coverage_ci(rec, catalogue=100, n_boot=80, seed=0)
    assert 0.0 <= lo <= hi <= 1.0
    # a resample draws duplicates, so the union of a resample cannot exceed the
    # union of the full sample -- the interval sits at or below the point
    assert hi <= point + 1e-9


def test_coverage_ci_is_deterministic_for_a_fixed_seed():
    rec = [np.array([i, i + 1]) for i in range(0, 40, 2)]
    a = coverage_ci(rec, catalogue=100, n_boot=50, seed=7)
    b = coverage_ci(rec, catalogue=100, n_boot=50, seed=7)
    assert a == b


def test_empty_input_does_not_raise():
    assert all(np.isnan(x) for x in coverage_ci([], catalogue=100))
    assert len(intra_list_diversity_per_list([], _emb(5))) == 0
