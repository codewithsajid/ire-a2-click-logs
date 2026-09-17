"""Q3 ablation: how the user is built out of their history.

Both retrievers take the same two decisions and both were frozen without
evidence: how many recent clicks to read (`n_recent = 30`), and whether to decay
older ones. The semantic side mean-pools article vectors; the lexical side sums
term-frequency rows. They are the same choice in two representations, so they
are swept together and reported side by side -- if the optimum differs, that is
a fact about the representation rather than about the user.

Recency weighting is the only recency signal MIND admits: its history carries no
timestamps, only order, so the decay runs over rank position on both datasets to
keep the sweep comparable.
"""
from __future__ import annotations

import argparse, json, time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

import sys; sys.path.insert(0, str(Path(__file__).parent))
from q4_eval import tuned_bm25
from newsrec.embeddings import article_embeddings
from newsrec.lexical import BM25Index
from newsrec.retrieval import (candidate_universe_for_split, evaluate_recall, summarise,
                               user_histories)
from newsrec.semantic import ANNIndex, l2_normalise, user_vectors
from newsrec.store import FeatureStore

KS = (50, 100, 200)
TOPK = max(KS)


def weighted_query_matrix(idx: BM25Index, hists, n_recent: int,
                          halflife: float | None) -> sp.csr_matrix:
    """History -> sparse term row, optionally decaying older clicks by position.

    `queries_from_history` builds an unweighted incidence matrix; this is the
    same product with a per-click weight, so the two paths differ only in H.
    """
    rows, cols, vals = [], [], []
    for i, h in enumerate(hists):
        h = np.asarray(h, dtype=np.int64)
        if n_recent:
            h = h[-n_recent:]
        h = h[h < idx.n_docs]
        if h.size == 0:
            continue
        if halflife:
            age = np.arange(len(h))[::-1]
            w = 0.5 ** (age / halflife)
        else:
            w = np.ones(len(h))
        rows.extend([i] * len(h)); cols.extend(h.tolist()); vals.extend(w.tolist())
    H = sp.csr_matrix((vals, (rows, cols)), shape=(len(hists), idx.n_docs), dtype=np.float32)
    return (H @ idx._TF).tocsr()


def main(dataset: str, variant: str, split: str, emb_name: str, out_dir: Path):
    fs = FeatureStore(dataset, variant)
    uni = candidate_universe_for_split(fs, split, 7)
    uids, hists = user_histories(fs, split)
    emb = np.asarray(article_embeddings(fs, emb_name))
    k1, b = tuned_bm25(dataset, variant, None, None)
    idx = BM25Index.build(fs.texts(), lang=fs.lang()).reweight(k1=k1, b=b)
    cold_thr = 5 if dataset == "mind" else 10
    hist_len = np.array([len(h) for h in hists])

    print(f"== {dataset}/{variant} {split}: {len(uids):,} users | universe {len(uni):,} "
          f"| history length median {int(np.median(hist_len))} p90 {int(np.percentile(hist_len,90))}")
    res = {"dataset": dataset, "variant": variant, "split": split, "embedding": emb_name,
           "universe": int(len(uni)), "n_users": int(len(uids)),
           "history_median": int(np.median(hist_len)), "runs": []}

    def score(kind: str, n_recent: int, halflife: float | None):
        t0 = time.perf_counter()
        if kind == "emb":
            uv = user_vectors(hists, emb, n_recent=n_recent,
                              recency_weighted=halflife is not None,
                              halflife=halflife or 10.0)
            _, topk = ANNIndex(emb[uni], ids=uni, kind="flat").search(uv, top_k=TOPK)
        else:
            Q = weighted_query_matrix(idx, hists, n_recent, halflife)
            _, topk = idx.search_sparse(Q, top_k=TOPK, universe=uni)
        s = summarise(evaluate_recall(fs, split, uids, topk, KS, cold_threshold=cold_thr), KS)
        row = {"retriever": kind, "n_recent": n_recent, "halflife": halflife,
               "seconds": round(time.perf_counter() - t0, 1),
               **{f"recall@{k}": round(s[f"recall@{k}"], 5) for k in KS},
               "cold_recall@100": round(s["cold"]["recall@100"], 5),
               "warm_recall@100": round(s["warm"]["recall@100"], 5)}
        res["runs"].append(row)
        hl = "none" if halflife is None else f"{halflife:g}"
        print(f"  {kind:5s} n_recent={str(n_recent or 'all'):>4s} decay={hl:>4s}   "
              + "  ".join(f"{row[f'recall@{k}']:.4f}" for k in KS)
              + f"   cold {row['cold_recall@100']:.4f} warm {row['warm_recall@100']:.4f}")

    print(f"\n  {'ret':5s} {'window':>16s}   " + "  ".join(f"r@{k:<4}" for k in KS) + "    slices")
    for kind in ("emb", "bm25"):
        for n in (5, 10, 20, 30, 50, 100, 0):
            score(kind, n, None)
        for hl in (5.0, 10.0, 20.0):
            score(kind, 0, hl)

    out_dir.mkdir(parents=True, exist_ok=True)
    f = out_dir / f"userrep_{dataset}_{variant}.json"
    f.write_text(json.dumps(res, indent=2))
    for kind in ("emb", "bm25"):
        rows = [r for r in res["runs"] if r["retriever"] == kind]
        best = max(rows, key=lambda r: r["recall@100"])
        flat = next(r for r in rows if r["n_recent"] == 30 and r["halflife"] is None)
        print(f"\n  best {kind}: n_recent={best['n_recent'] or 'all'} "
              f"halflife={best['halflife']} -> {best['recall@100']:.4f} "
              f"({100*(best['recall@100']/flat['recall@100']-1):+.1f}% vs the shipped n_recent=30)")
    print(f"== wrote {f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--variant", default="small")
    ap.add_argument("--split", default="test")
    ap.add_argument("--embedding", default="contrastive")
    ap.add_argument("--out", default="reports/q3")
    a = ap.parse_args()
    main(a.dataset, a.variant, a.split, a.embedding, Path(a.out))
