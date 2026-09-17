"""Q4: the offline evaluation harness.

Q2/Q3 measured *retrieval* -- does the clicked article appear anywhere in a
top-K pulled from the whole candidate universe. This module measures *ranking*,
which is what both leaderboards actually score: given the candidate list the
platform really showed in one impression, how well are the clicked articles
ordered within it. The two questions have different answers, and conflating them
is the easiest way to report a number that does not mean what it says.

Conventions, chosen to match the official `evaluate.py` in the EB-NeRD starter:

  * every clicked article in an impression is relevant (28.8% of MIND
    impressions have more than one click, so a single-click assumption would
    silently mean different things on the two datasets);
  * MRR uses the rank of the *first* relevant item;
  * nDCG uses binary gains, ideal-ordered over min(n_clicked, k);
  * AUC is the rank-sum form, which is exact and needs no threshold sweep.

Impressions with no positives, or with nothing but positives, are undefined for
AUC and are dropped from that metric only -- never silently scored as 0.5.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from .ids import stable_uniform

DISCOUNTS = 1.0 / np.log2(np.arange(2, 4098))          # 1/log2(rank+1), rank >= 1


def _idcg_map(k: int, max_pos: int) -> dict[int, float]:
    """IDCG@k for every possible positive count -- a small lookup, not a join."""
    cum = np.concatenate([[0.0], np.cumsum(DISCOUNTS[:k])])
    return {p: float(cum[min(p, k)]) for p in range(max_pos + 1)}


def _tie_jitter(pairs: pl.DataFrame, s: np.ndarray, seed: int) -> np.ndarray:
    """Noise below the resolution of any real score difference, keyed on the row.

    Keyed on (imp, article_idx) rather than drawn from a stream indexed by row
    position: a sequential draw makes the result depend on how the table happens
    to be ordered, so rebuilding the feature store re-resolved every tie and moved
    the constant-scoring baselines in the fourth decimal. See ids.stable_uniform.
    """
    u = stable_uniform(pairs["imp"].to_numpy(), pairs["article_idx"].to_numpy(), seed)
    return u * 1e-9 * (np.abs(s).mean() + 1.0)


def rank_metrics(pairs: pl.DataFrame, score_col: str = "score",
                 ks: tuple[int, ...] = (5, 10), seed: int = 0) -> pl.DataFrame:
    """Per-impression AUC, MRR and nDCG@k from flat (impression, candidate) rows.

    `pairs` needs columns: imp, label (bool), and `score_col`.

    Ties are broken by noise keyed on (impression, article) rather than by row
    order. Candidate
    lists are not shuffled -- EB-NeRD lists them in the order they were rendered
    -- so an ordinal rank would hand every constant-scoring baseline (popularity
    scores thousands of candidates 0) the platform's own layout as a free
    signal, which reads as model quality and is not.
    """
    s = pairs[score_col].to_numpy().astype(np.float64)
    d = pairs.with_columns(pl.Series("_raw", s),
                           pl.Series("_s", s + _tie_jitter(pairs, s, seed))).with_columns([
        # AUC uses midranks on the *unjittered* score, which is the standard tie
        # treatment and reproduces sklearn.roc_auc_score exactly. Breaking ties at
        # random instead leaves AUC unbiased but noisy -- a constant-scoring
        # baseline scored 0.5065 +/- 0.5 per impression where the truth is exactly
        # 0.5, and that noise moved with the row order of the input table.
        pl.col("_raw").rank("average").over("imp").alias("_r_asc"),
        # Top-k metrics have to commit to an order, so they use the jittered rank:
        # EB-NeRD lists candidates as rendered, and an ordinal rank on tied scores
        # would hand every constant-scoring baseline the platform's own layout as
        # a free signal.
        pl.col("_s").rank("ordinal", descending=True).over("imp").alias("_r_desc"),
    ])
    kmax = max(ks)
    disc = pl.Series("_d", DISCOUNTS)
    agg = [
        pl.len().alias("n"),
        pl.col("label").sum().alias("n_pos"),
        pl.when(pl.col("label")).then(pl.col("_r_asc").cast(pl.Float64))
          .otherwise(0.0).sum().alias("_rank_sum"),
        pl.when(pl.col("label")).then(pl.col("_r_desc")).otherwise(None)
          .min().alias("_first_rel"),
        # official MRR: the mean of 1/rank over *every* relevant item, not the
        # reciprocal rank of the first one. Verified against the graders' own
        # ebrec.evaluation.metrics._ranking.mrr_score in tests/test_official_metrics.py.
        pl.when(pl.col("label")).then(1.0 / pl.col("_r_desc").cast(pl.Float64))
          .otherwise(0.0).sum().alias("_rr_sum"),
    ]
    for k in ks:
        agg.append(
            pl.when(pl.col("label") & (pl.col("_r_desc") <= k))
              .then(pl.col("_r_desc").cast(pl.Int64) - 1)
              .otherwise(None).alias(f"_hit{k}")
        )
    g = d.group_by("imp").agg(agg)

    # DCG: sum of 1/log2(rank+1) over the relevant items inside the cutoff
    out = g.with_columns([
        pl.col(f"_hit{k}").list.drop_nulls().list.eval(
            pl.element().replace_strict(list(range(kmax)), list(DISCOUNTS[:kmax]),
                                        default=0.0, return_dtype=pl.Float64)
        ).list.sum().alias(f"_dcg{k}") for k in ks
    ])
    max_pos = int(g["n_pos"].max() or 0)
    for k in ks:
        m = _idcg_map(k, max_pos)
        out = out.with_columns(
            pl.col("n_pos").replace_strict(list(m), list(m.values()),
                                           default=0.0, return_dtype=pl.Float64).alias(f"_idcg{k}")
        )
    out = out.with_columns([
        # recall inside the impression at each cutoff -- the "rank penalty" with
        # no discount at all, kept alongside nDCG so the cutoff ablation can show
        # how much of a metric's movement is the discount and how much is the cut
        *[(pl.col(f"_hit{k}").list.drop_nulls().list.len() / pl.col("n_pos"))
          .alias(f"hit@{k}") for k in ks],
        pl.when((pl.col("n_pos") > 0) & (pl.col("n_pos") < pl.col("n")))
          .then((pl.col("_rank_sum") - pl.col("n_pos") * (pl.col("n_pos") + 1) / 2.0)
                / (pl.col("n_pos") * (pl.col("n") - pl.col("n_pos"))))
          .otherwise(None).alias("auc"),
        (pl.col("_rr_sum") / pl.col("n_pos")).alias("mrr"),
        # the single-best-hit variant, kept because it answers a different
        # question ("how far down is the first thing they wanted?") and is what
        # EB-NeRD's 0.51%-multi-click data effectively measures anyway
        (1.0 / pl.col("_first_rel")).fill_null(0.0).alias("mrr_first"),
        *[(pl.col(f"_dcg{k}") / pl.col(f"_idcg{k}")).fill_nan(0.0).fill_null(0.0)
          .alias(f"ndcg@{k}") for k in ks],
    ])
    return out.select(["imp", "n", "n_pos", "auc", "mrr", "mrr_first",
                       pl.col("_first_rel").alias("first_rel"),
                       *[f"ndcg@{k}" for k in ks], *[f"hit@{k}" for k in ks]])


def topk_lists(pairs: pl.DataFrame, k: int, score_col: str = "score",
               seed: int = 0) -> pl.DataFrame:
    """The k highest-scoring candidates per impression, in rank order."""
    s = pairs[score_col].to_numpy().astype(np.float64)
    return (
        pairs.with_columns(pl.Series("_s", s + _tie_jitter(pairs, s, seed)))
        .with_columns(pl.col("_s").rank("ordinal", descending=True).over("imp").alias("_r"))
        .filter(pl.col("_r") <= k)
        .sort("imp", "_r")
        .group_by("imp", maintain_order=True)
        .agg(pl.col("article_idx").alias("rec"))
    )


# ------------------------------------------------------------ beyond accuracy

def intra_list_diversity(rec: list[np.ndarray], emb: np.ndarray, chunk: int = 8192) -> float:
    """1 - mean pairwise cosine inside each recommended list, averaged.

    Content diversity, not category diversity: MIND's category taxonomy is 18
    labels wide and EB-NeRD's is not the same taxonomy, so a category-based
    Simpson index would not be comparable across the two datasets, while the
    embedding space is the same object the semantic retriever ranks in.
    """
    lens = np.array([len(r) for r in rec])
    keep = np.where(lens >= 2)[0]
    if not len(keep):
        return float("nan")
    kmin = int(lens[keep].min())
    if kmin < 2:
        keep = keep[lens[keep] >= 2]
    mat = np.stack([np.asarray(rec[i])[:kmin] for i in keep])       # (m, kmin)
    tot, n = 0.0, 0
    for s in range(0, len(mat), chunk):
        v = np.asarray(emb[mat[s:s + chunk].ravel()], dtype=np.float32)
        v = v.reshape(-1, kmin, v.shape[-1])
        v /= np.maximum(np.linalg.norm(v, axis=2, keepdims=True), 1e-12)
        gram = np.einsum("bkd,bjd->bkj", v, v)
        iu = np.triu_indices(kmin, k=1)
        tot += float(gram[:, iu[0], iu[1]].sum()); n += len(v) * len(iu[0])
    return float(1.0 - tot / max(n, 1))


def novelty(rec: list[np.ndarray], pop_share: np.ndarray) -> float:
    """Mean self-information -log2 p(i) of the recommended items.

    p(i) is the article's share of clicks in the window *before* the scored
    split, so novelty is measured against what a serving system could have known
    -- using in-window popularity here would make a recommender look novel for
    surfacing articles that later turned out to be hits.
    """
    logp = -np.log2(np.maximum(pop_share, 1e-12))
    vals = [logp[np.asarray(r)].mean() for r in rec if len(r)]
    return float(np.mean(vals)) if vals else float("nan")


def coverage(rec: list[np.ndarray], catalogue: int) -> float:
    """Share of the catalogue that appears in at least one recommendation."""
    seen = set()
    for r in rec:
        seen.update(np.asarray(r).tolist())
    return float(len(seen) / max(catalogue, 1))


# ------------------------------------------------------------------ summaries

def bootstrap(values: np.ndarray, n_boot: int = 500, alpha: float = 0.05,
              seed: int = 0, max_cells: int = 2 ** 25) -> tuple[float, float, float]:
    """Percentile bootstrap of the mean, in memory-bounded blocks.

    The obvious one-liner draws an (n_boot x n) index matrix. At EB-NeRD large's
    12.5M impressions and 1000 resamples that is 100 GB, so resamples are drawn
    a block at a time and only the block's means are kept.

    `values` is sorted first. Resampling with replacement is order-independent in
    distribution, so this changes nothing statistically -- but it pins the drawn
    resample, which otherwise depends on the row order of whatever table produced
    the column, and that order is not stable across a store rebuild.
    """
    v = np.asarray(values, dtype=np.float64)
    v = np.sort(v[~np.isnan(v)])
    n = v.size
    if n == 0:
        return (float("nan"),) * 3
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=np.float64)
    block = max(1, min(n_boot, max_cells // n))
    for s in range(0, n_boot, block):
        b = min(block, n_boot - s)
        means[s:s + b] = v[rng.integers(0, n, size=(b, n))].mean(axis=1)
    return float(v.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


METRICS = ("auc", "mrr", "mrr_first", "ndcg@5", "ndcg@10")


def summarise_ranking(df: pl.DataFrame, n_boot: int = 500,
                      metrics: tuple[str, ...] | None = None) -> dict:
    out = {"n_impressions": df.height}
    for m in (metrics or METRICS):
        if m not in df.columns:
            continue
        mean, lo, hi = bootstrap(df[m].to_numpy(), n_boot=n_boot)
        out[m] = round(mean, 5)
        out[f"{m}_ci95"] = [round(lo, 5), round(hi, 5)]
    return out


def rank_profile(first_rel: np.ndarray, cap: int = 40) -> list[int]:
    """Histogram of the rank of the first relevant item, 1..cap plus overflow.

    This is the empirical object every rank-discounted metric integrates over:
    nDCG and MRR differ only in the weight they put on this distribution, so
    plotting it next to the discount curves shows where a metric's opinion
    actually comes from.
    """
    v = np.asarray(first_rel, dtype=np.float64)
    v = v[~np.isnan(v)].astype(np.int64)
    h = np.bincount(np.clip(v, 1, cap + 1), minlength=cap + 2)[1:cap + 2]
    return [int(x) for x in h]
