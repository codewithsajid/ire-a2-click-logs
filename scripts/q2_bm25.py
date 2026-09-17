"""Q2: BM25 lexical candidate generation, with a (k1, b) ablation.

    python scripts/q2_bm25.py --dataset mind --variant small [--universe-days 7]

Reports recall@{50,100,200} on val for the parameter grid, then re-runs the best
setting on test with a cold/warm slice and bootstrap CIs. Because (k1, b) only
change the document weights, the grid reuses one tokenisation and one postings
table -- `reweight` is the whole cost per grid point.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import polars as pl

from newsrec.config import N_RECENT_DEFAULT
from newsrec.lexical import BM25Index
from newsrec.retrieval import (bootstrap_ci, candidate_universe_for_split,
                               evaluate_recall, summarise, universe_ceiling,
                               user_histories)
from newsrec.store import FeatureStore

KS = (50, 100, 200)


def run(dataset: str, variant: str, universe_days: int | None, n_recent: int,
        out_dir: Path, quick: int | None = None,
        grid_users: int | None = None) -> dict:
    fs = FeatureStore(dataset, variant)
    lang = fs.lang()
    print(f"== {fs}  lang={lang}")

    t0 = time.perf_counter()
    idx = BM25Index.build(fs.texts(), lang=lang)
    build_s = time.perf_counter() - t0
    print(f"  index: {idx.n_docs:,} docs | vocab {idx.vocab_size:,} | postings {len(idx.tf):,} "
          f"| avgdl {idx.avgdl:.1f} | {build_s:.1f}s")

    results = {"dataset": dataset, "variant": variant, "lang": lang,
               "index": {"n_docs": idx.n_docs, "vocab": idx.vocab_size,
                         "postings": int(len(idx.tf)), "avgdl": idx.avgdl,
                         "build_seconds": round(build_s, 2)},
               "n_recent": n_recent, "universe_days": universe_days, "grid": []}

    def retrieve(split: str, k1: float, b: float, variant_name: str = "bm25",
                 limit: int | None = None):
        uids, hists = user_histories(fs, split)
        limit = limit or quick
        if limit and len(uids) > limit:
            # a fixed random subsample, not a prefix: user_idx is assigned in
            # sorted src-id order, so uids[:n] is a biased slice of the catalogue
            keep = np.sort(np.random.default_rng(0).choice(len(uids), limit, replace=False))
            uids, hists = uids[keep], [hists[i] for i in keep]
        idx.reweight(k1=k1, b=b, variant=variant_name)
        universe = candidate_universe_for_split(fs, split, universe_days)
        Q = idx.queries_from_history(hists, n_recent=n_recent)
        _, topk = idx.search_sparse(Q, top_k=max(KS), universe=universe)
        return uids, topk, universe

    # ---- how much recall is reachable at all --------------------------------
    for split in ("val", "test"):
        uni = candidate_universe_for_split(fs, split, universe_days)
        ceil = universe_ceiling(fs, split, uni)
        results[f"ceiling_{split}"] = ceil
        print(f"  universe[{split}]: {ceil['universe_size'] or idx.n_docs:,} articles | "
              f"reachable clicks {ceil['clicks_reachable']:,}/{ceil['clicks_total']:,} "
              f"-> recall ceiling {ceil['ceiling']:.4f}")

    # ---- (k1, b) grid on val -------------------------------------------------
    print(f"\n  (k1, b) grid on val   [recall@50 / @100 / @200]")
    for k1 in (0.9, 1.2, 1.5, 2.0):
        for b in (0.0, 0.3, 0.75, 1.0):
            t0 = time.perf_counter()
            uids, topk, uni = retrieve("val", k1, b, limit=grid_users)
            df = evaluate_recall(fs, "val", uids, topk, KS,
                                 only_users=bool(grid_users or quick))
            s = summarise(df, KS)
            s.update({"k1": k1, "b": b, "seconds": round(time.perf_counter() - t0, 2),
                      "universe": int(len(uni)) if uni is not None else idx.n_docs})
            results["grid"].append(s)
            print(f"    k1={k1:<4} b={b:<5} "
                  + " / ".join(f"{s[f'recall@{k}']:.4f}" for k in KS)
                  + f"   ({s['seconds']:.1f}s)")

    best = max(results["grid"], key=lambda r: r["recall@100"])
    results["best"] = {"k1": best["k1"], "b": best["b"], "recall@100": best["recall@100"]}
    print(f"\n  best on val: k1={best['k1']} b={best['b']} (recall@100 {best['recall@100']:.4f})")

    # ---- BM25L variant at the best (k1, b) ----------------------------------
    uids, topk, _ = retrieve("val", best["k1"], best["b"], "bm25l", limit=grid_users)
    bm25l = summarise(evaluate_recall(fs, "val", uids, topk, KS,
                                      only_users=bool(grid_users or quick)), KS)
    results["bm25l_val"] = bm25l
    print(f"  BM25L (delta=0.5) on val: " + " / ".join(f"{bm25l[f'recall@{k}']:.4f}" for k in KS))

    # ---- best setting on test, with slices and CIs --------------------------
    uids, topk, uni = retrieve("test", best["k1"], best["b"])
    cold_thr = 5 if dataset == "mind" else None
    if cold_thr is None:   # EB-NeRD guarantees >=5 history items, so use a quantile
        hl = fs.history("test").select("n_hist").collect()["n_hist"]
        cold_thr = int(np.quantile(hl.to_numpy(), 0.25))
    df = evaluate_recall(fs, "test", uids, topk, KS, cold_threshold=cold_thr)
    test = summarise(df, KS)
    test["cold_threshold"] = cold_thr
    for k in KS:
        m, lo, hi = bootstrap_ci(df[f"recall@{k}"].to_numpy())
        test[f"recall@{k}_ci95"] = [round(lo, 5), round(hi, 5)]
    results["test"] = test
    print(f"\n  test @ k1={best['k1']} b={best['b']}  (cold threshold n_hist<{cold_thr})")
    for k in KS:
        lo, hi = test[f"recall@{k}_ci95"]
        ceil = results["ceiling_test"]["ceiling"] or 1.0
        print(f"    recall@{k:<4} {test[f'recall@{k}']:.4f}  ({test[f'recall@{k}']/ceil:.1%} of ceiling)"
              f"  95% CI [{lo:.4f}, {hi:.4f}]"
              f"   cold {test['cold'][f'recall@{k}']:.4f} (n={test['cold']['n']:,})"
              f"   warm {test['warm'][f'recall@{k}']:.4f} (n={test['warm']['n']:,})")

    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"q2_bm25_{dataset}_{variant}.json"
    out.write_text(json.dumps(results, indent=2))
    np.save(out_dir / f"topk_bm25_{dataset}_{variant}.npy", topk)
    np.save(out_dir / f"topk_bm25_{dataset}_{variant}_users.npy", uids)
    print(f"\n== wrote {out}")
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--universe-days", type=int, default=7)
    ap.add_argument("--n-recent", type=int, default=N_RECENT_DEFAULT,
                    help="clicks read per user (0 = the whole history, the setting scripts/q3_userrep.py picked)")
    ap.add_argument("--quick", type=int, default=None, help="limit users everywhere, for smoke tests")
    ap.add_argument("--grid-users", type=int, default=None,
                    help="limit users during the (k1,b) sweep only; test still uses all")
    ap.add_argument("--out", default="reports/q2")
    a = ap.parse_args()
    run(a.dataset, a.variant, a.universe_days, a.n_recent, Path(a.out), a.quick,
        a.grid_users)
