"""Non-lexical, non-semantic retrieval baselines.

These exist because a retrieval score is meaningless without them. Retrieving
200 articles from a 7-day universe of ~800 already captures a quarter of clicks
by chance, so "BM25 gets recall@200 = 0.28" is only interesting next to what
random, popularity and recency get on the same universe.

Popularity doubles as the Codabench submission baseline (`newsrec.submit`).
"""
from __future__ import annotations

import numpy as np
import polars as pl

from .ids import stable_uniform
from .store import FeatureStore


def random_topk(n_users: int, universe: np.ndarray, k: int, seed: int = 0,
                user_ids: np.ndarray | None = None,
                max_cells: int = 2 ** 26) -> np.ndarray:
    """Uniform sample without replacement -- the floor any retriever must clear.

    Drawn a block of users at a time rather than one `rng.choice` per user: at
    EB-NeRD large's 791K users the per-user loop was thirteen minutes for a
    baseline. Ranking one key per candidate and taking the k smallest is the same
    distribution, vectorised. `max_cells` caps the block so the intermediate stays
    bounded regardless of universe size.

    The keys come from hashing (user_idx, article_idx, seed) rather than from a
    sequential RNG, so a user's draw depends on *who they are* and not on where
    they landed in the table. With a stream, re-ordering the history table -- which
    a store rebuild can do -- silently re-rolled every baseline and moved the
    denominator of every lift-over-random figure. Falls back to row position when
    `user_ids` is not supplied, which is still deterministic, just not stable
    across a reordering.
    """
    k = min(k, len(universe))
    uni = np.asarray(universe, dtype=np.int32)
    ub = uni.astype(np.uint64)
    rows = (np.arange(n_users, dtype=np.uint64) if user_ids is None
            else np.asarray(user_ids, dtype=np.uint64))
    out = np.empty((n_users, k), dtype=np.int32)
    block = max(1, min(n_users, max_cells // max(len(uni), 1)))
    for s in range(0, n_users, block):
        n = min(block, n_users - s)
        keys = stable_uniform(rows[s:s + n, None], ub[None, :], seed)
        part = np.argpartition(keys, k - 1, axis=1)[:, :k]
        out[s:s + n] = uni[part]
    return out


def popularity_ranking(fs: FeatureStore, split: str, universe: np.ndarray | None,
                       decayed: bool = True) -> np.ndarray:
    """Articles ordered by clicks in the window strictly before `split`.

    Same table the submission baseline scores with, so the retrieval number and
    the leaderboard number come from one definition of popularity.
    """
    col = "clicks_decayed" if decayed else "clicks"
    pop = (
        fs.article_features(split)
        .select("article_idx", col)
        .filter(pl.col(col) > 0)
        .sort(col, descending=True)
        .collect()
    )
    order = pop["article_idx"].to_numpy().astype(np.int32)
    if universe is not None:
        order = order[np.isin(order, universe)]
    return order


def recency_ranking(fs: FeatureStore, split: str, universe: np.ndarray | None) -> np.ndarray:
    """Newest articles first -- news decays fast, so this is a real contender."""
    t0, _ = fs.bounds(split)
    stats = (
        fs.article_stats()
        .join(fs.articles().with_row_index("idx").select("idx", "published_time"),
              on="idx", how="left")
        .with_columns(pl.coalesce(pl.col("published_time"), pl.col("first_seen_time")).alias("_t"))
        .filter(pl.col("_t") < pl.lit(t0))
        .sort("_t", descending=True)
        .collect()
    )
    order = stats["idx"].to_numpy().astype(np.int32)
    if universe is not None:
        order = order[np.isin(order, universe)]
    return order


def broadcast(order: np.ndarray, n_users: int, k: int) -> np.ndarray:
    """A global ranking, served identically to every user."""
    k = min(k, len(order))
    return np.tile(order[:k], (n_users, 1)).astype(np.int32)
