"""Q2 ablation: does the ranking objective matter, and by how much.

The lecture's claim is that the pointwise -> pairwise -> listwise progression is
a real ordering and that today it is three lines of config. Both halves are
testable here on one flag:

    binary        pointwise  -- regress the click; ranking is relative and this
                               objective does not know that, so capacity goes
                               into getting the absolute scale right
    rank_xendcg   listwise   -- a cross-entropy surrogate over the whole list
    lambdarank    listwise   -- pair gradients scaled by the nDCG change a swap
                               would cause, so the gradient concentrates where
                               the metric actually moves

The claim predicts lambdarank >= rank_xendcg > binary on nDCG. Whether that holds
on a near-single-click news log with ~11 candidates per impression is not
something the slide can answer, and it is cheap to measure.

Every comparison ships a **paired** bootstrap 95% CI on the per-impression
difference. Two separate CIs would overstate the uncertainty of a difference
measured on the same impressions, because the impression-to-impression variance
is common to both systems and cancels only when the resample is shared.

Writes reports/q2/objective_<dataset>_<variant>.json.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import polars as pl

from newsrec.evaluate import (METRICS, paired_report, rank_metrics,
                              summarise_ranking)
from newsrec.rerank import SHIPPED, feature_names, predict, train
from newsrec.store import FeatureStore

OBJECTIVES = ("binary", "rank_xendcg", "lambdarank")


def per_impression(df: pl.DataFrame, scores: np.ndarray) -> pl.DataFrame:
    pairs = df.select("imp", "article_idx", "label").with_columns(
        pl.Series("score", np.asarray(scores, dtype=np.float64)))
    return rank_metrics(pairs, ks=(5, 10))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["ebnerd", "mind"])
    ap.add_argument("--variant", default="small")
    ap.add_argument("--rounds", type=int, default=600)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--truncation-sweep", default="",
                    help="comma-separated lambdarank_truncation_level values to "
                         "sweep, e.g. 10,30,100,300")
    ap.add_argument("--baseline", default="binary",
                    help="objective every other one is compared against")
    ap.add_argument("--out", type=Path, default=Path("reports/q2"))
    a = ap.parse_args()

    root = Path("artifacts/features") / a.dataset / a.variant
    need = [root / f"{s}.parquet" for s in ("train", "val", "test")]
    if not all(p.exists() for p in need):
        raise SystemExit(f"run scripts/q2_rerank.py --dataset {a.dataset} first")
    tr, va, te = (pl.read_parquet(p).sort("imp") for p in need)

    feats = feature_names(tr, SHIPPED)
    print(f"[{a.dataset}/{a.variant}] {len(feats)} features, "
          f"{tr.height:,} train rows over {tr['imp'].n_unique():,} impressions")

    fitted, per_imp, summary = {}, {}, {}
    for obj in OBJECTIVES:
        t0 = time.perf_counter()
        m = train(tr, va, feats, objective=obj, num_boost_round=a.rounds,
                  verbose_eval=0)
        s = predict(m, te, feats)
        fitted[obj] = {"trees": int(m.best_iteration),
                       "fit_seconds": round(time.perf_counter() - t0, 1)}
        per_imp[obj] = per_impression(te, s)
        summary[obj] = summarise_ranking(per_imp[obj], n_boot=300)
        print(f"   {obj:<12} {fitted[obj]['trees']:>4} trees "
              f"{fitted[obj]['fit_seconds']:>7.1f}s   "
              f"AUC {summary[obj]['auc']:.4f}  nDCG@10 {summary[obj]['ndcg@10']:.4f}")

    base = a.baseline
    if a.truncation_sweep:
        print(f"\n   lambdarank truncation sweep "
              f"(candidates/impression: median "
              f"{int(te.group_by('imp').len()['len'].median())}, "
              f"max {int(te.group_by('imp').len()['len'].max())})")
        trunc = {}
        for tl in [int(x) for x in a.truncation_sweep.split(",")]:
            m = train(tr, va, feats, objective="lambdarank",
                      num_boost_round=a.rounds, verbose_eval=0, truncation=tl)
            pi = per_impression(te, predict(m, te, feats))
            s = summarise_ranking(pi, n_boot=300)
            vs = paired_report(per_imp[base], pi, n_boot=a.n_boot)
            trunc[tl] = {"trees": int(m.best_iteration), "summary": s,
                         "paired_vs_baseline": vs}
            nd = vs.get("ndcg@10", {})
            print(f"     truncation {tl:>4}: {m.best_iteration:>4} trees  "
                  f"AUC {s['auc']:.4f}  nDCG@10 {s['ndcg@10']:.4f}  "
                  f"vs {base} {nd.get('delta', float('nan')):+.4f} "
                  f"[{nd.get('ci95', [float('nan')]*2)[0]:+.4f}, "
                  f"{nd.get('ci95', [float('nan')]*2)[1]:+.4f}]"
                  f"{'' if nd.get('excludes_zero') else '  (spans 0)'}")
        out_trunc = trunc
    else:
        out_trunc = {}

    paired = {obj: paired_report(per_imp[base], per_imp[obj], n_boot=a.n_boot)
              for obj in OBJECTIVES if obj != base}

    print(f"\n   paired bootstrap vs {base} (95% CI on the per-impression delta)")
    print(f"   {'objective':<13} {'metric':<9} {'delta':>9} {'CI95':>22} {'excl. 0':>8}")
    for obj, rep in paired.items():
        for m in METRICS:
            r = rep.get(m)
            if not r:
                continue
            print(f"   {obj:<13} {m:<9} {r['delta']:>+9.4f} "
                  f"[{r['ci95'][0]:>+8.4f}, {r['ci95'][1]:>+8.4f}] "
                  f"{'yes' if r['excludes_zero'] else 'NO':>8}")

    out = {"dataset": a.dataset, "variant": a.variant,
           "n_features": len(feats), "features": feats,
           "baseline_objective": base,
           "fitted": fitted, "summary": summary, "paired_vs_baseline": paired,
           "truncation_sweep": out_trunc}
    a.out.mkdir(parents=True, exist_ok=True)
    f = a.out / f"objective_{a.dataset}_{a.variant}.json"
    f.write_text(json.dumps(out, indent=2))
    print(f"\n   wrote {f}")


if __name__ == "__main__":
    main()
