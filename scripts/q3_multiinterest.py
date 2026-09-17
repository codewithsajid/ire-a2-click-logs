"""Q3 ablation: is one vector enough to describe a user?

The semantic user representation is a single mean-pooled vector over the whole
click history. That is the standard cheap choice, and it has an obvious failure
mode: a reader who follows football and politics is represented by a point
between the two, which may be near neither. News recommenders usually answer this
with a multi-interest representation, so it is worth measuring rather than
assuming -- especially now that the history window is unbounded, which gives the
mean more topics to average over.

Two ways of splitting one user into k queries, retrieving top-(K/k) from each and
merging:

  chunks  -- k contiguous slices of the history in time order. Cheap, and tests
             whether interests are *sequential* (the user moved on).
  kmeans  -- k centroids from Lloyd's algorithm on the user's own vectors. Tests
             whether interests are *concurrent* (the user holds several at once).

k=1 is the shipped representation and must reproduce it exactly, which is the
control that the merge logic is not itself doing the work.
"""
from __future__ import annotations

import argparse, json, time
from pathlib import Path

import numpy as np

from newsrec.embeddings import article_embeddings
from newsrec.retrieval import (candidate_universe_for_split, evaluate_recall,
                               summarise, user_histories)
from newsrec.semantic import ANNIndex, l2_normalise
from newsrec.store import FeatureStore

KS = (50, 100, 200)


def _kmeans(v: np.ndarray, k: int, iters: int = 5, seed: int = 0) -> np.ndarray:
    """Spherical k-means on one user's vectors; k-means++-ish seeding."""
    n = len(v)
    if n <= k:
        return v
    rng = np.random.default_rng(seed)
    c = v[rng.choice(n, k, replace=False)]
    for _ in range(iters):
        a = np.argmax(v @ c.T, axis=1)
        for j in range(k):
            m = a == j
            if m.any():
                c[j] = v[m].mean(0)
        c = l2_normalise(c)
    return c


def user_queries(hists, emb, k: int, mode: str) -> tuple[np.ndarray, np.ndarray]:
    """Flat query matrix plus the user each query row belongs to."""
    rows, owners = [], []
    for u, h in enumerate(hists):
        h = np.asarray(h)
        h = h[h < emb.shape[0]]
        if h.size == 0:
            rows.append(np.zeros(emb.shape[1], dtype=np.float32)); owners.append(u); continue
        v = l2_normalise(emb[h].astype(np.float32))
        if k == 1 or len(v) <= 1:
            cent = v.mean(0, keepdims=True)
        elif mode == "chunks":
            cent = np.stack([c.mean(0) for c in np.array_split(v, min(k, len(v)))])
        else:
            cent = _kmeans(v, k)
        for c in cent:
            rows.append(c); owners.append(u)
    return l2_normalise(np.asarray(rows, dtype=np.float32)), np.asarray(owners)


def merge(scores, ids, owners, n_users: int, k_out: int) -> np.ndarray:
    """Best score per (user, article) across that user's query vectors."""
    out = np.full((n_users, k_out), -1, dtype=np.int64)
    order = np.argsort(owners, kind="stable")
    owners, scores, ids = owners[order], scores[order], ids[order]
    bounds = np.searchsorted(owners, np.arange(n_users + 1))
    for u in range(n_users):
        lo, hi = bounds[u], bounds[u + 1]
        if lo == hi:
            continue
        a, sc = ids[lo:hi].ravel(), scores[lo:hi].ravel()
        uniq, inv = np.unique(a, return_inverse=True)
        best = np.full(len(uniq), -np.inf, dtype=np.float32)
        np.maximum.at(best, inv, sc)
        top = uniq[np.argsort(-best)[:k_out]]
        out[u, :len(top)] = top
    return out


def main(dataset: str, variant: str, split: str, emb_name: str, out_dir: Path):
    fs = FeatureStore(dataset, variant)
    uni = candidate_universe_for_split(fs, split, 7)
    uids, hists = user_histories(fs, split)
    emb = np.asarray(article_embeddings(fs, emb_name))
    ix = ANNIndex(emb[uni], ids=uni, kind="flat")
    cold = 5 if dataset == "mind" else 10
    hl = np.array([len(h) for h in hists])

    res = {"dataset": dataset, "variant": variant, "split": split, "embedding": emb_name,
           "n_users": len(uids), "universe": int(len(uni)),
           "history_median": int(np.median(hl)), "runs": []}
    print(f"== Q3 multi-interest {dataset}/{variant} {split}: {len(uids):,} users, "
          f"universe {len(uni):,}, history median {res['history_median']}")

    for mode in ("chunks", "kmeans"):
        for k in (1, 2, 3, 5, 8):
            if k > 1 and mode == "chunks" and False:
                continue
            t0 = time.perf_counter()
            Q, owners = user_queries(hists, emb, k, mode)
            # each sub-query retrieves the full depth; merging by best score then
            # trims to K, so a user is never handed fewer than K candidates
            sc, ids = ix.search(Q, top_k=max(KS))
            topk = merge(sc, ids, owners, len(uids), max(KS))
            s = summarise(evaluate_recall(fs, split, uids, topk, KS, cold_threshold=cold), KS)
            row = {"mode": mode, "k": k, "seconds": round(time.perf_counter() - t0, 1),
                   "queries": int(len(Q)),
                   **{f"recall@{kk}": round(s[f"recall@{kk}"], 5) for kk in KS},
                   "cold_recall@100": round(s["cold"]["recall@100"], 5),
                   "warm_recall@100": round(s["warm"]["recall@100"], 5)}
            res["runs"].append(row)
            print(f"  {mode:7s} k={k}  {row['recall@50']:.4f} {row['recall@100']:.4f} "
                  f"{row['recall@200']:.4f}   cold {row['cold_recall@100']:.4f} "
                  f"warm {row['warm_recall@100']:.4f}   {row['queries']:,} queries "
                  f"({row['seconds']}s)")
            if k == 1 and mode == "chunks":
                res["single_vector"] = row["recall@100"]

    base = res["single_vector"]
    best = max(res["runs"], key=lambda r: r["recall@100"])
    res["best"] = {k: best[k] for k in ("mode", "k", "recall@100")}
    res["gain_vs_single"] = round(best["recall@100"] / base - 1, 4)
    print(f"\n  single vector {base:.4f} -> best {best['mode']} k={best['k']} "
          f"{best['recall@100']:.4f} ({100 * res['gain_vs_single']:+.1f}%)")
    out_dir.mkdir(parents=True, exist_ok=True)
    f = out_dir / f"multiinterest_{dataset}_{variant}.json"
    f.write_text(json.dumps(res, indent=2))
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
