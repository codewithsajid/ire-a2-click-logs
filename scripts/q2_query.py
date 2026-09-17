"""Q2 ablation: does the query need term saturation?

BM25 saturates *document* term frequency through k1, and classic BM25 also
saturates *query* term frequency through k3. This pipeline never did: the query
is `H @ TF`, the raw summed term counts of every article the user clicked. That
was defensible when the query was 30 clicks. It is not obviously defensible now
that the userrep sweep moved the default to the whole history -- EB-NeRD's median
user has 81 clicks, its 90th percentile 377, so a common term can appear hundreds
of times in one query and dominate the dot product linearly.

k3 = inf is the shipped behaviour (no saturation); k3 = 0 is pure binary presence.
"""
from __future__ import annotations

import argparse, json, time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from newsrec.lexical import BM25Index
from newsrec.retrieval import (candidate_universe_for_split, evaluate_recall,
                               summarise, user_histories)
from newsrec.store import FeatureStore

KS = (50, 100, 200)
K3_GRID = [("inf (shipped)", None), ("1000", 1000.0), ("32", 32.0),
           ("8", 8.0), ("2", 2.0), ("0 (binary)", 0.0)]


def saturate(Q: sp.csr_matrix, k3: float | None) -> sp.csr_matrix:
    """(k3+1)*qtf / (k3+qtf), elementwise on the query weights."""
    if k3 is None:
        return Q
    R = Q.copy()
    if k3 == 0.0:
        R.data = np.ones_like(R.data)
    else:
        R.data = (k3 + 1.0) * R.data / (k3 + R.data)
    return R


def main(dataset: str, variant: str, split: str, k1: float | None, b: float | None,
         out_dir: Path):
    import sys; sys.path.insert(0, str(Path(__file__).parent))
    from q4_eval import tuned_bm25
    k1, b = tuned_bm25(dataset, variant, k1, b)

    fs = FeatureStore(dataset, variant)
    uni = candidate_universe_for_split(fs, split, 7)
    uids, hists = user_histories(fs, split)
    idx = BM25Index.build(fs.texts(), lang=fs.lang()).reweight(k1=k1, b=b)
    Q = idx.queries_from_history(hists)
    hl = np.array([len(h) for h in hists])
    cold = 5 if dataset == "mind" else 10

    res = {"dataset": dataset, "variant": variant, "split": split, "k1": k1, "b": b,
           "n_users": len(uids), "universe": int(len(uni)),
           "history_median": int(np.median(hl)), "history_p90": int(np.percentile(hl, 90)),
           "query_nnz_per_user": round(Q.nnz / max(Q.shape[0], 1), 1), "runs": []}
    print(f"== Q2 query saturation {dataset}/{variant} {split}: {len(uids):,} users, "
          f"history median {res['history_median']} p90 {res['history_p90']}, "
          f"{res['query_nnz_per_user']:.0f} distinct query terms/user")

    for label, k3 in K3_GRID:
        t0 = time.perf_counter()
        _, topk = idx.search_sparse(saturate(Q, k3), top_k=max(KS), universe=uni)
        s = summarise(evaluate_recall(fs, split, uids, topk, KS, cold_threshold=cold), KS)
        row = {"k3": label, "seconds": round(time.perf_counter() - t0, 1),
               **{f"recall@{k}": round(s[f"recall@{k}"], 5) for k in KS},
               "cold_recall@100": round(s["cold"]["recall@100"], 5),
               "warm_recall@100": round(s["warm"]["recall@100"], 5)}
        res["runs"].append(row)
        print(f"  k3={label:14s} {row['recall@50']:.4f} {row['recall@100']:.4f} "
              f"{row['recall@200']:.4f}   cold {row['cold_recall@100']:.4f} "
              f"warm {row['warm_recall@100']:.4f}  ({row['seconds']}s)")

    base = next(r for r in res["runs"] if r["k3"].startswith("inf"))
    best = max(res["runs"], key=lambda r: r["recall@100"])
    res["best"], res["gain_vs_shipped"] = best["k3"], round(
        best["recall@100"] / base["recall@100"] - 1, 4)
    print(f"\n  best k3={best['best'] if False else best['k3']} -> {best['recall@100']:.4f} "
          f"({100 * res['gain_vs_shipped']:+.1f}% vs no saturation)")
    out_dir.mkdir(parents=True, exist_ok=True)
    f = out_dir / f"query_{dataset}_{variant}.json"
    f.write_text(json.dumps(res, indent=2))
    print(f"== wrote {f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--variant", default="small")
    ap.add_argument("--split", default="test")
    ap.add_argument("--k1", type=float, default=None)
    ap.add_argument("--b", type=float, default=None)
    ap.add_argument("--out", default="reports/q2")
    a = ap.parse_args()
    main(a.dataset, a.variant, a.split, a.k1, a.b, Path(a.out))
