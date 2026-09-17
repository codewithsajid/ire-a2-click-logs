"""Q2: the cascade -- retrieve top-K from the live universe, then re-rank.

Two different questions live under the phrase "two-stage", and conflating them
is how a re-ranker gets a number that does not mean what it says.

  * **In-impression.** Re-order the candidate list the platform actually showed.
    This is what both Codabench graders score, and what A1's Q4 table reports.
  * **Corpus-wide cascade.** Stage one retrieves K articles out of everything
    live; stage two re-ranks those K. This is what a serving system does, and
    what Q2's wording describes.

They are not comparable and both are reported. The cascade has a ceiling the
in-impression framing does not: A1 measured recall@100 at 0.1069 on EB-NeRD, so
stage one loses most clicks before stage two is consulted at all. Each stage has
one duty -- recall upstream, precision downstream -- and the end-to-end number is
the product.

**Training on retrieved sets.** The re-ranker here is trained on what stage one
retrieves rather than on what the platform showed, so that the distribution it
learns matches the one it scores. That choice has a cost this module handles
explicitly rather than hiding:

1. *Positives go missing.* At recall@100 = 0.1069, roughly nine impressions in
   ten have no clicked article anywhere in their retrieved set. A ranking
   objective silently drops a group with no positive, so training on the raw
   retrieved sets would throw away ~89% of the data and call it training. The
   clicked articles are therefore unioned back in -- standard practice for
   two-stage pipelines, and the only way the stage-two model sees a positive at
   all.
2. *Negatives are unexamined.* A retrieved article that was never shown to
   anyone was not rejected by the user; nobody looked at it. Treating it as a
   negative is the assumption the whole approach rests on, and it is the one the
   lecture warns about. `was_shown` records which negatives were genuinely
   examined, so the ablation can price the assumption instead of inheriting it.
   It is a diagnostic column, never a model feature -- a ranker given it would
   simply learn to predict the platform's own retrieval.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from .config import N_RECENT_DEFAULT
from .embeddings import article_embeddings
from .lexical import BM25Index
from .behaviour import CONTEXT_COLS, _present
from .retrieval import drop_seen, user_histories
from .semantic import ANNIndex, l2_normalise, user_vectors
from .store import FeatureStore


def stage1(fs: FeatureStore, split: str, k: int, method: str = "hybrid",
           emb_name: str = "contrastive", k1: float = 2.0, b: float = 1.0,
           history_mode: str = "shipped", n_recent: int = N_RECENT_DEFAULT,
           universe_days: int = 7, rrf_k: int = 60,
           ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Retrieve top-k per user from the articles live during this split.

    Retrieval is per *user*, not per impression: the query is the click history,
    so every impression by one user shares a retrieved set. That is also what
    makes the cascade affordable -- EB-NeRD's test split has 244,647 impressions
    but only 15,342 users.

    `method` is one of `bm25`, `emb`, or `hybrid` (reciprocal-rank fusion of the
    two, which A1 measured as competitive with the better single retriever on
    both datasets without needing to know in advance which one that is).

    Already-read articles are dropped: A1 measured that BM25's top hit was
    frequently an article from the user's own history, and excluding them lifted
    recall@50 by 38%.

    Returns (user_ids, topk_article_ids, topk_scores).
    """
    from .retrieval import candidate_universe_for_split

    universe = candidate_universe_for_split(fs, split, universe_days)
    user_ids, hists = user_histories(fs, split, history_mode)
    over = k + max((len(h) for h in hists), default=0)   # room to drop seen
    over = min(over, len(universe) if universe is not None else over)

    ranks: list[np.ndarray] = []
    scores_of: list[np.ndarray] = []

    if method in ("bm25", "hybrid"):
        idx = BM25Index.build(fs.texts(), lang=fs.lang(), k1=k1, b=b)
        Q = idx.queries_from_history(hists, n_recent=n_recent)
        s, ids = idx.search_sparse(Q, top_k=over, universe=universe)
        ranks.append(ids); scores_of.append(s)

    if method in ("emb", "hybrid"):
        embn = l2_normalise(np.asarray(article_embeddings(fs, emb_name), dtype=np.float32))
        uv = user_vectors(hists, embn, n_recent=n_recent)
        sub = universe if universe is not None else np.arange(embn.shape[0])
        ann = ANNIndex(embn[sub], ids=sub, kind="flat")
        s, ids = ann.search(uv, top_k=over)
        ranks.append(ids); scores_of.append(s)

    if len(ranks) == 1:
        ids, sc = ranks[0], scores_of[0]
    else:
        ids, sc = _rrf(ranks, over, rrf_k)

    kept = drop_seen(ids, hists, k)
    # re-read the fused score for whatever survived the seen-filter
    lookup = {}
    out_s = np.zeros_like(kept, dtype=np.float32)
    for i in range(kept.shape[0]):
        lookup = dict(zip(ids[i].tolist(), sc[i].tolist()))
        row = kept[i]
        out_s[i] = [lookup.get(int(x), 0.0) for x in row]
    return user_ids, kept, out_s


def _rrf(ranked: list[np.ndarray], k: int, rrf_k: int = 60
         ) -> tuple[np.ndarray, np.ndarray]:
    """Reciprocal-rank fusion: sum 1/(rrf_k + rank) across retrievers.

    Rank-based rather than score-based because BM25 scores and cosines are not on
    a common scale and no monotone rescaling makes them one. A1 shipped this as
    `hybrid_rrf` and it landed within a point of the better single retriever on
    both datasets.
    """
    n = ranked[0].shape[0]
    out_ids = np.full((n, k), -1, dtype=np.int32)
    out_sc = np.zeros((n, k), dtype=np.float32)
    for i in range(n):
        acc: dict[int, float] = {}
        for r in ranked:
            for pos, a in enumerate(r[i].tolist()):
                if a < 0:
                    continue
                acc[a] = acc.get(a, 0.0) + 1.0 / (rrf_k + pos + 1)
        if not acc:
            continue
        items = sorted(acc.items(), key=lambda kv: -kv[1])[:k]
        out_ids[i, :len(items)] = [a for a, _ in items]
        out_sc[i, :len(items)] = [s for _, s in items]
    return out_ids, out_sc


def retrieved_pairs(fs: FeatureStore, split: str, user_ids: np.ndarray,
                    topk: np.ndarray, scores: np.ndarray,
                    union_clicked: bool = True,
                    max_impressions: int = 0, seed: int = 0) -> pl.DataFrame:
    """One row per (impression, retrieved candidate), labelled and flagged.

    `union_clicked` adds each impression's clicked articles to its retrieved set.
    Without it roughly nine impressions in ten carry no positive at all -- stage
    one's recall@100 is 0.1069 on EB-NeRD -- and a ranking objective drops those
    groups entirely, so the model would train on the tenth of the data where
    retrieval already succeeded. That is a biased sample of exactly the wrong
    kind: it is the subset where stage two is needed least.

    **Measured consequence of the union.** Injecting positives makes "was
    injected" perfectly predictable from any feature that records whether an
    article appeared in a real impression -- and `roll_age_hours` is exactly
    that. Trained this way on EB-NeRD the model reaches 0.9896 AUC on its own
    retrieved sets and 0.4637 in-impression, i.e. below random: it learned the
    injection, not relevance. So `union_clicked=False` is also offered, which
    keeps only impressions whose positive stage one genuinely retrieved. That
    discards 70-80% of impressions and biases towards the cases retrieval already
    handles, but every row in it arrived by the same mechanism.

    `was_shown` marks candidates that were in the impression's real candidate
    list. It separates a negative the user saw and passed from one nobody ever
    looked at, and exists so the difference can be measured. It is never a model
    feature.

    `stage1_score` is the fused retrieval score, carried through so the cascade's
    "before" ordering is the one stage one actually produced.
    """
    ret = pl.DataFrame({
        "user_idx": pl.Series(user_ids, dtype=pl.UInt32),
        "_cand": pl.Series([r[r >= 0].tolist() for r in topk], dtype=pl.List(pl.UInt32)),
        "_score": pl.Series([s[r >= 0].tolist() for r, s in zip(topk, scores)],
                            dtype=pl.List(pl.Float32)),
    })

    src = fs.impressions(split)
    # Carry the EB-NeRD context block through. Without it the retrieved matrix
    # would be missing the whole `context` family while the in-impression matrix
    # has it, and the same model would be trained and scored on different
    # columns -- the training-serving skew this framing exists to avoid.
    imp = (
        src.filter(pl.col("clicked").list.len() > 0)
        .with_row_index("imp")
        .select("imp", "src_row", "user_idx", "time", "candidates", "clicked",
                *_present(src, CONTEXT_COLS))
    )
    if max_impressions:
        n = imp.select(pl.len()).collect().item()
        if n > max_impressions:
            sel = np.sort(np.random.default_rng(seed).choice(n, max_impressions, replace=False))
            imp = imp.filter(pl.col("imp").is_in(pl.Series(sel, dtype=pl.UInt32).implode()))

    joined = imp.join(ret.lazy(), on="user_idx", how="left").with_columns(
        pl.col("_cand").fill_null(pl.lit([], dtype=pl.List(pl.UInt32))),
        pl.col("_score").fill_null(pl.lit([], dtype=pl.List(pl.Float32))),
    )

    if union_clicked:
        # clicked articles not already retrieved, carrying a sentinel stage-1
        # score of 0 so the cascade's "before" ranking puts them where stage one
        # actually left them: nowhere
        joined = joined.with_columns(
            pl.col("clicked").list.set_difference(pl.col("_cand")).alias("_missed")
        ).with_columns(
            pl.col("_cand").list.concat(pl.col("_missed")).alias("_cand"),
            pl.col("_score").list.concat(
                pl.col("_missed").list.eval(pl.element().cast(pl.Float32) * 0.0)
            ).alias("_score"),
        ).drop("_missed")

    return (
        joined
        .explode(["_cand", "_score"])
        .drop_nulls("_cand")
        .rename({"_cand": "article_idx", "_score": "stage1_score"})
        .with_columns(
            pl.col("article_idx").is_in(pl.col("clicked")).alias("label"),
            pl.col("article_idx").is_in(pl.col("candidates")).alias("was_shown"),
        )
        .with_columns(pl.col("article_idx").len().over("imp").cast(pl.UInt16).alias("n_candidates"))
        .drop("candidates", "clicked")
        .collect(engine="streaming")
    )


def stage1_recall(pairs: pl.DataFrame, fs: FeatureStore, split: str) -> dict:
    """How many of the impression's real clicks stage one put in front of stage two.

    The cascade's upstream duty, and the ceiling on everything downstream: stage
    two cannot rank what stage one never retrieved.

    Counted against the impression's true clicked list read from the store, not
    against the candidate table. The candidate table cannot answer this once
    `union_clicked` is off, because then every positive in it arrived by
    retrieval by construction and the recall computes to exactly 1.0 -- which is
    what an earlier version of this function reported.
    """
    got = (
        pairs.lazy()
        # rows that stage one genuinely retrieved: with union on, an injected
        # positive carries a sentinel score of 0
        .filter((pl.col("stage1_score") > 0) & pl.col("label"))
        .group_by("imp").agg(pl.len().alias("hit"))
    )
    truth = (
        pairs.lazy().select("imp", "src_row").unique()
        .join(fs.impressions(split).select("src_row", "clicked"), on="src_row", how="left")
        .select("imp", pl.col("clicked").list.len().alias("pos"))
    )
    d = truth.join(got, on="imp", how="left").with_columns(pl.col("hit").fill_null(0)).collect()
    ok = d.filter(pl.col("pos") > 0)
    return {
        "n_impressions": d.height,
        "n_with_positive": ok.height,
        "recall": float((ok["hit"] / ok["pos"].cast(pl.Float64)).mean()) if ok.height else None,
        "impressions_with_no_positive_retrieved": float((ok["hit"] == 0).mean()) if ok.height else None,
    }
