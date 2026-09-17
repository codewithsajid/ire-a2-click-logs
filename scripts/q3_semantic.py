"""Q3: semantic retrieval, and the lexical-vs-semantic comparison it feeds."""
from __future__ import annotations

import argparse, json, time
from pathlib import Path

import numpy as np
import polars as pl

from newsrec.embeddings import article_embeddings
import sys; sys.path.insert(0, str(Path(__file__).parent))
from q4_eval import tuned_bm25
from newsrec.baselines import broadcast, popularity_ranking, random_topk, recency_ranking
from newsrec.config import N_RECENT_DEFAULT
from newsrec.lexical import BM25Index
from newsrec.retrieval import (bootstrap_ci, candidate_universe_for_split, evaluate_recall,
                               summarise, universe_ceiling, user_histories)
from newsrec.semantic import ANNIndex, encode_texts, user_vectors
from newsrec.store import FeatureStore

KS = (50, 100, 200)


def main(dataset: str, variant: str, split: str, emb_names: list[str], out_dir: Path,
         n_recent: int = N_RECENT_DEFAULT):
    # the same operating point Q2 selected for this corpus -- comparing semantic
    # retrieval against a BM25 that was never tuned would flatter it
    k1, b = tuned_bm25(dataset, variant, None, None)
    out_dir.mkdir(parents=True, exist_ok=True)   # topk arrays are written before the json
    fs = FeatureStore(dataset, variant)
    uni = candidate_universe_for_split(fs, split, 7)
    ceil = universe_ceiling(fs, split, uni)
    uids, hists = user_histories(fs, split)
    n = len(uids)
    nimp = fs.impressions(split).select(pl.len()).collect().item()
    print(f"== {dataset}/{variant} {split}: {nimp:,} impressions | {n:,} users "
          f"| universe {len(uni):,} | ceiling {ceil['ceiling']:.4f}")

    res = {"dataset": dataset, "variant": variant, "split": split,
           "n_impressions": nimp, "n_users": n, "universe": int(len(uni)),
           "n_recent": n_recent, "bm25": {"k1": k1, "b": b},
           "ceiling": ceil["ceiling"], "methods": {}}
    cold_thr = 5 if dataset == "mind" else int(np.quantile(
        fs.history(split).select("n_hist").collect()["n_hist"].to_numpy(), 0.25))

    def record(name, topk, seconds):
        df = evaluate_recall(fs, split, uids, topk, KS, cold_threshold=cold_thr)
        s = summarise(df, KS)
        s["seconds"] = round(seconds, 2)
        for k in KS:
            m, lo, hi = bootstrap_ci(df[f"recall@{k}"].to_numpy(), n_boot=400)
            s[f"recall@{k}_ci95"] = [round(lo, 5), round(hi, 5)]
            s[f"lift_vs_random@{k}"] = None
        res["methods"][name] = s
        print(f"  {name:24s} " + "  ".join(f"{s[f'recall@{k}']:.4f}" for k in KS)
              + f"   cold {s['cold']['recall@100']:.4f} warm {s['warm']['recall@100']:.4f}"
              + f"   ({seconds:.1f}s)")
        return s

    print(f"  {'method':24s} " + "  ".join(f"r@{k:<5}" for k in KS) + "     slices (r@100)")
    t0 = time.time(); record("random", random_topk(n, uni, max(KS), user_ids=uids), time.time() - t0)
    t0 = time.time(); record("popularity", broadcast(popularity_ranking(fs, split, uni), n, max(KS)), time.time() - t0)
    t0 = time.time(); record("recency", broadcast(recency_ranking(fs, split, uni), n, max(KS)), time.time() - t0)

    t0 = time.time()
    idx = BM25Index.build(fs.texts(), lang=fs.lang()).reweight(k1=k1, b=b)
    Q = idx.queries_from_history(hists, n_recent=n_recent)
    _, topk_bm25 = idx.search_sparse(Q, top_k=max(KS), universe=uni)
    record("bm25", topk_bm25, time.time() - t0)

    for name in emb_names:
        t0 = time.time()
        emb = np.asarray(article_embeddings(fs, name))
        sub = ANNIndex(emb[uni], ids=uni, kind="flat")
        uv = user_vectors(hists, emb, n_recent=n_recent)
        _, topk = sub.search(uv, top_k=max(KS))
        record(f"emb:{name.split('/')[-1]}", topk, time.time() - t0)
        if name == emb_names[0]:
            np.save(out_dir / f"topk_emb_{dataset}_{variant}.npy", topk)

    # random is the reference point: recall@K against a universe of size U is
    # roughly K/U by chance, so lift is the only comparable number across datasets
    rnd = res["methods"]["random"]
    for m in res["methods"].values():
        for k in KS:
            m[f"lift_vs_random@{k}"] = round(m[f"recall@{k}"] / max(rnd[f"recall@{k}"], 1e-9), 3)

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"topk_bm25_{dataset}_{variant}.npy", topk_bm25)
    np.save(out_dir / f"users_{dataset}_{variant}.npy", uids)
    (out_dir / f"q3_{dataset}_{variant}.json").write_text(json.dumps(res, indent=2))
    print(f"\n  lift over random @100: " + ", ".join(
        f"{k}={v['lift_vs_random@100']}" for k, v in res["methods"].items()))
    print(f"== wrote {out_dir}/q3_{dataset}_{variant}.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--embeddings", default="")
    ap.add_argument("--n-recent", type=int, default=N_RECENT_DEFAULT,
                    help="clicks read per user (0 = the whole history, the setting scripts/q3_userrep.py picked)")
    ap.add_argument("--out", default="reports/q3")
    a = ap.parse_args()
    names = [x for x in a.embeddings.split(",") if x]
    main(a.dataset, a.variant, a.split, names, Path(a.out), a.n_recent)
