"""Shared retrieval evaluation for the lexical (Q2) and semantic (Q3) systems.

Retrieval is per *user*, not per impression: the query is the user's click
history, so every impression by the same user shares one retrieved set. Recall
still varies across those impressions because the clicked articles differ.

recall@K follows the multi-click convention fixed in Q1:

    recall@K = |clicked ∩ topK| / |clicked|

not a hit/miss indicator -- 28.8% of MIND impressions have more than one click,
so an indicator would silently mean different things on the two datasets.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from .store import FeatureStore


def user_histories(fs: FeatureStore, split: str, mode: str = "shipped") -> tuple[np.ndarray, list[np.ndarray]]:
    h = fs.history(split, mode).select("user_idx", "article_idx").collect()
    return h["user_idx"].to_numpy(), [np.asarray(x, dtype=np.int64) for x in h["article_idx"].to_list()]


def drop_seen(topk: np.ndarray, histories: list[np.ndarray], k: int) -> np.ndarray:
    """Remove articles the user has already read from their retrieved list.

    Both retrievers otherwise rank a user's own history at the top -- BM25
    especially, since an article's own terms match its own text perfectly. Those
    slots are wasted: a news reader is not served a story they already clicked.
    Retrieve k + |history| candidates, drop the seen ones, keep the first k.
    """
    out = np.full((len(histories), k), -1, dtype=np.int32)
    for i, h in enumerate(histories):
        seen = set(np.asarray(h).tolist())
        keep = [d for d in topk[i].tolist() if d not in seen][:k]
        out[i, :len(keep)] = keep
    return out


def evaluate_recall(
    fs: FeatureStore,
    split: str,
    user_ids: np.ndarray,
    topk: np.ndarray,
    ks: tuple[int, ...] = (50, 100, 200),
    cold_threshold: int | None = None,
    only_users: bool = False,
) -> pl.DataFrame:
    """Per-impression recall@K for a user->top-K retrieval.

    Vectorised with list set-intersection rather than a Python loop over
    impressions: EB-NeRD's test split has 12.5M of them, and the loop version
    was the slowest thing in the harness by two orders of magnitude.

    Impressions whose user has no retrieved set (no history) score 0 rather than
    being dropped -- they are exactly the cold-start case the slice exposes.

    `only_users` restricts scoring to impressions belonging to `user_ids`. It is
    off by default because the full-split mean is the number Q2/Q3 report; turn
    it on when `user_ids` is a *subsample*, where the default would otherwise
    score every unsampled user as a zero and deflate the mean by the sampling
    ratio rather than measuring anything.
    """
    kmax = topk.shape[1] if topk.size else 0
    retrieved = pl.DataFrame({
        "user_idx": pl.Series(user_ids, dtype=pl.UInt32),
        "_topk": pl.Series([row.tolist() for row in topk], dtype=pl.List(pl.UInt32))
                 if kmax else pl.Series([[]] * len(user_ids), dtype=pl.List(pl.UInt32)),
    })
    hist = fs.history(split).select("user_idx", "n_hist")
    imp = fs.impressions(split).select("user_idx", "clicked")
    if only_users:
        imp = imp.filter(pl.col("user_idx").is_in(
            pl.Series(user_ids, dtype=pl.UInt32).implode()))
    df = (
        imp
        .filter(pl.col("clicked").list.len() > 0)
        .join(retrieved.lazy(), on="user_idx", how="left")
        .join(hist, on="user_idx", how="left")
        .with_columns(
            pl.col("_topk").fill_null(pl.lit([], dtype=pl.List(pl.UInt32))),
            pl.col("n_hist").fill_null(0),
            pl.col("clicked").list.len().alias("n_clicked"),
        )
        .with_columns([
            (pl.col("clicked").list.set_intersection(pl.col("_topk").list.head(k)).list.len()
             / pl.col("n_clicked")).alias(f"recall@{k}")
            for k in ks
        ])
        .drop("_topk")
        .collect(engine="streaming")
    )
    if cold_threshold is not None:
        df = df.with_columns((pl.col("n_hist") < cold_threshold).alias("is_cold"))
    return df


def summarise(df: pl.DataFrame, ks: tuple[int, ...] = (50, 100, 200)) -> dict:
    out = {"n_impressions": df.height}
    for k in ks:
        out[f"recall@{k}"] = float(df[f"recall@{k}"].mean())
    if "is_cold" in df.columns:
        for label, sub in [("cold", df.filter(pl.col("is_cold"))), ("warm", df.filter(~pl.col("is_cold")))]:
            out[label] = {"n": sub.height,
                          **{f"recall@{k}": (float(sub[f"recall@{k}"].mean()) if sub.height else None) for k in ks}}
    return out


def bootstrap_ci(values: np.ndarray, n_boot: int = 400, alpha: float = 0.05,
                 seed: int = 0, max_cells: int = 2 ** 25) -> tuple[float, float, float]:
    """Percentile bootstrap CI over impressions (Q4 asks for 95% CIs).

    Blocked for the same reason as `evaluate.bootstrap`: EB-NeRD large's test
    split has 12.5M impressions, and the naive (n_boot x n) draw does not fit.
    """
    rng = np.random.default_rng(seed)
    v = np.asarray(values, dtype=np.float64)
    n = v.size
    if n == 0:
        return (float("nan"),) * 3
    means = np.empty(n_boot, dtype=np.float64)
    block = max(1, min(n_boot, max_cells // n))
    for s in range(0, n_boot, block):
        b = min(block, n_boot - s)
        means[s:s + b] = v[rng.integers(0, n, size=(b, n))].mean(axis=1)
    return float(v.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def candidate_universe_for_split(fs: FeatureStore, split: str, days: int | None) -> np.ndarray | None:
    """Articles live in the `days` before the split ends.

    Without this, recall@200 on EB-NeRD is measured against a catalogue that
    includes articles published in 1993 -- retrievable in principle, impossible
    to have been clicked.
    """
    if not days:
        return None
    start, end = fs.bounds(split)
    return np.sort(fs.candidate_universe(start, end, days=days))


def universe_ceiling(fs: FeatureStore, split: str, universe: np.ndarray | None) -> dict:
    """How much recall is even reachable given the candidate universe.

    Restricting retrieval to recently-live articles is what makes corpus-wide
    recall meaningful, but it also caps it: a click on an article outside the
    window can never be retrieved. Reporting recall without this ceiling makes a
    universe that is too narrow look like a retrieval failure.
    """
    imp = fs.impressions(split).select("clicked").collect()
    total = reachable = 0
    per_imp = []
    uset = None if universe is None else set(universe.tolist())
    for (clicked,) in imp.iter_rows():
        c = np.asarray(clicked, dtype=np.int64)
        if c.size == 0:
            continue
        total += c.size
        if uset is None:
            hit = c.size
        else:
            hit = sum(1 for x in c.tolist() if x in uset)
        reachable += hit
        per_imp.append(hit / c.size)
    return {
        "universe_size": int(len(universe)) if universe is not None else None,
        "clicks_total": total,
        "clicks_reachable": reachable,
        "ceiling": float(np.mean(per_imp)) if per_imp else 0.0,
    }
