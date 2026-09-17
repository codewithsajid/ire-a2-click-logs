"""Q1/Q3 ablation: how much of "retrieval quality" is the candidate window?

Retrieval is restricted to articles alive in the last N days, anchored at the
start of the scored period. N=7 was picked once, on the reasoning that EB-NeRD
carries articles published in 1993 and corpus-wide recall against them is
meaningless. That reasoning is sound and the parameter was never swept -- which
matters more than it looks, because the window sets the difficulty of every
number in Q2 and Q3 and it is wildly asymmetric between the datasets: 2,063 live
articles on EB-NeRD large against 29,309 on MIND.

It also quietly decides the Q3 ANN conclusion. "Exact search beats every
approximate index at the operating point" is a statement about a 2,000-vector
corpus, and the operating point is this parameter.

Reported per N: the live universe, the recall ceiling it imposes (clicks on
articles outside the window can never be retrieved), and what BM25, embeddings
and random actually get -- so the window's cost in reachable clicks can be read
against its benefit in precision.
"""
from __future__ import annotations

import argparse, json, time
from pathlib import Path

import numpy as np

from newsrec.baselines import random_topk
from newsrec.embeddings import article_embeddings
from newsrec.lexical import BM25Index
from newsrec.retrieval import (candidate_universe_for_split, evaluate_recall,
                               summarise, universe_ceiling, user_histories)
from newsrec.semantic import ANNIndex, user_vectors
from newsrec.store import FeatureStore

KS = (50, 100, 200)
DAYS = [1, 3, 7, 14, 30, None]          # None = the whole catalogue


def main(dataset: str, variant: str, split: str, emb_name: str,
         k1: float | None, b: float | None, out_dir: Path):
    import sys; sys.path.insert(0, str(Path(__file__).parent))
    from q4_eval import tuned_bm25
    k1, b = tuned_bm25(dataset, variant, k1, b)

    fs = FeatureStore(dataset, variant)
    uids, hists = user_histories(fs, split)
    idx = BM25Index.build(fs.texts(), lang=fs.lang()).reweight(k1=k1, b=b)
    Q = idx.queries_from_history(hists)
    emb = np.asarray(article_embeddings(fs, emb_name))
    uv = user_vectors(hists, emb)
    cold = 5 if dataset == "mind" else 10

    res = {"dataset": dataset, "variant": variant, "split": split, "embedding": emb_name,
           "k1": k1, "b": b, "n_users": len(uids), "catalogue": int(fs.n_articles), "runs": []}
    print(f"== Q1 candidate window {dataset}/{variant} {split}: catalogue {fs.n_articles:,}, "
          f"{len(uids):,} users")

    for days in DAYS:
        t0 = time.perf_counter()
        uni = (candidate_universe_for_split(fs, split, days) if days is not None
               else np.arange(fs.n_articles, dtype=np.int64))
        ceil = universe_ceiling(fs, split, uni)["ceiling"]
        row = {"days": days, "universe": int(len(uni)), "ceiling": round(ceil, 5)}
        for name, topk in (
            ("bm25", idx.search_sparse(Q, top_k=max(KS), universe=uni)[1]),
            ("emb", ANNIndex(emb[uni], ids=uni, kind="flat").search(uv, top_k=max(KS))[1]),
            ("random", random_topk(len(uids), uni, max(KS), user_ids=uids)),
        ):
            s = summarise(evaluate_recall(fs, split, uids, topk, KS, cold_threshold=cold), KS)
            row[name] = {f"recall@{k}": round(s[f"recall@{k}"], 5) for k in KS}
            row[name]["cold_recall@100"] = round(s["cold"]["recall@100"], 5)
        # what the retrievers earn above chance, which is the only number
        # comparable across windows of different sizes
        for name in ("bm25", "emb"):
            row[f"{name}_lift"] = round(
                row[name]["recall@100"] / max(row["random"]["recall@100"], 1e-9), 2)
        row["seconds"] = round(time.perf_counter() - t0, 1)
        res["runs"].append(row)
        lbl = "all" if days is None else str(days)
        print(f"  days={lbl:>3s}  universe {row['universe']:>7,}  ceiling {row['ceiling']:.4f}  "
              f"bm25 {row['bm25']['recall@100']:.4f} (x{row['bm25_lift']:.1f})  "
              f"emb {row['emb']['recall@100']:.4f} (x{row['emb_lift']:.1f})  "
              f"random {row['random']['recall@100']:.4f}   ({row['seconds']}s)")

    ship = next(r for r in res["runs"] if r["days"] == 7)
    best_lift = max(res["runs"], key=lambda r: r["emb_lift"])
    res["shipped"] = {"days": 7, "universe": ship["universe"], "ceiling": ship["ceiling"]}
    res["best_lift"] = {"days": best_lift["days"], "emb_lift": best_lift["emb_lift"]}
    print(f"\n  shipped days=7: universe {ship['universe']:,}, ceiling {ship['ceiling']:.4f}")
    print(f"  best lift over random: days={best_lift['days']} at x{best_lift['emb_lift']:.1f}")
    out_dir.mkdir(parents=True, exist_ok=True)
    f = out_dir / f"universe_{dataset}_{variant}.json"
    f.write_text(json.dumps(res, indent=2))
    print(f"== wrote {f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--variant", default="small")
    ap.add_argument("--split", default="test")
    ap.add_argument("--embedding", default="contrastive")
    ap.add_argument("--k1", type=float, default=None)
    ap.add_argument("--b", type=float, default=None)
    ap.add_argument("--out", default="reports/q3")
    a = ap.parse_args()
    main(a.dataset, a.variant, a.split, a.embedding, a.k1, a.b, Path(a.out))
