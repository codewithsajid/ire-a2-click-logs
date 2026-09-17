"""Q3.3 + Q9: isolate the improvement, and price the features we are not allowed.

Three sweeps over one model, all against the same shipped baseline and all with
a paired bootstrap CI on the per-impression delta:

  **leave-one-family-out** -- drop `user`, `article`, `match` or `context` in
  turn. This is what isolates *which kind* of signal the ranker actually runs on,
  and it is the only form of the question that survives features being
  correlated: dropping one column when three others carry the same information
  measures nothing.

  **the improvement (Q3.2/3.3)** -- add the strictly-causal rolling counters.
  A1's popularity is frozen at the split boundary, which leaves 84.2% (EB-NeRD)
  and 54.9% (MIND) of candidate slots at a prior click count of exactly zero;
  rolling brings both to 3.7%. This is the same change measured on NRMS in
  `q3_nrms.py`, so the improvement is shown on two architectures rather than
  one, and the architecture and feature axes stay separable.

  **the grey zone (Q9)** -- add features that exist in the logs but that a
  ranker choosing what to show has not yet observed: dwell and scroll for the
  visit in progress, and the rendered position. Reported precisely because they
  must not ship.

A note on what "isolation" costs. Each variant is retrained rather than having
its feature zeroed at scoring time, because a GBDT's splits are chosen with the
feature present and zeroing it at inference measures a crippled model rather
than a model that never had it.

Writes reports/q3/ablation_<dataset>_<variant>.json.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import polars as pl

from newsrec.evaluate import METRICS, paired_report, rank_metrics, summarise_ranking
from newsrec.rerank import (FAMILIES, PRODUCTION, SHIPPED, feature_names,
                            importances, predict, train)


def per_impression(df: pl.DataFrame, scores: np.ndarray) -> pl.DataFrame:
    return rank_metrics(
        df.select("imp", "article_idx", "label").with_columns(
            pl.Series("score", np.asarray(scores, dtype=np.float64))), ks=(5, 10))


def fit_and_score(tr, va, te, feats, args):
    t0 = time.perf_counter()
    m = train(tr, va, feats, objective=args.objective,
              num_boost_round=args.rounds, verbose_eval=0)
    s = predict(m, te, feats)
    return m, per_impression(te, s), round(time.perf_counter() - t0, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["ebnerd", "mind"])
    ap.add_argument("--variant", default="small")
    ap.add_argument("--objective", default="lambdarank")
    ap.add_argument("--rounds", type=int, default=600)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--out", type=Path, default=Path("reports/q3"))
    a = ap.parse_args()

    root = Path("artifacts/features") / a.dataset / a.variant
    need = [root / f"{s}.parquet" for s in ("train", "val", "test")]
    if not all(p.exists() for p in need):
        raise SystemExit(f"run scripts/q2_rerank.py --dataset {a.dataset} first")
    tr, va, te = (pl.read_parquet(p).sort("imp") for p in need)

    base_feats = feature_names(tr, SHIPPED)
    print(f"[{a.dataset}/{a.variant}] baseline = {len(base_feats)} features "
          f"over {SHIPPED}")

    variants: dict[str, list[str]] = {"shipped (baseline)": base_feats}
    for fam in SHIPPED:
        keep = tuple(f for f in SHIPPED if f != fam)
        fs_ = feature_names(tr, keep)
        if len(fs_) < len(base_feats):
            variants[f"- {fam}"] = fs_
    variants["+ rolling (the improvement)"] = feature_names(tr, PRODUCTION)
    grey = feature_names(tr, SHIPPED + ("grey",))
    if len(grey) > len(base_feats):
        variants["+ grey (Q9, must not ship)"] = grey

    results, per_imp, models = {}, {}, {}
    for name, feats in variants.items():
        m, pi, secs = fit_and_score(tr, va, te, feats, a)
        models[name], per_imp[name] = m, pi
        results[name] = {"n_features": len(feats),
                         "trees": int(m.best_iteration),
                         "fit_seconds": secs,
                         **summarise_ranking(pi, n_boot=300)}
        print(f"   {name:<30} {len(feats):>3} feats  {m.best_iteration:>4} trees  "
              f"AUC {results[name]['auc']:.4f}  nDCG@10 {results[name]['ndcg@10']:.4f}")

    base = "shipped (baseline)"
    paired = {n: paired_report(per_imp[base], per_imp[n], n_boot=a.n_boot)
              for n in variants if n != base}

    print(f"\n   paired bootstrap vs the shipped baseline "
          f"(95% CI on the per-impression delta)")
    print(f"   {'variant':<30} {'ΔAUC':>9} {'CI95':>22} {'Δn@10':>9} {'excl 0':>7}")
    for n, rep in paired.items():
        ra, rn = rep.get("auc", {}), rep.get("ndcg@10", {})
        print(f"   {n:<30} {ra.get('delta', float('nan')):>+9.4f} "
              f"[{ra.get('ci95', [0, 0])[0]:>+8.4f},{ra.get('ci95', [0, 0])[1]:>+8.4f}] "
              f"{rn.get('delta', float('nan')):>+9.4f} "
              f"{'yes' if ra.get('excludes_zero') else 'NO':>7}")

    imp = importances(models[base], base_feats)
    by_fam = (imp.group_by("family").agg(pl.col("share").sum())
              .sort("share", descending=True).to_dicts())

    out = {"dataset": a.dataset, "variant": a.variant, "objective": a.objective,
           "baseline_families": list(SHIPPED),
           "variants": {n: v for n, v in variants.items()},
           "results": results, "paired_vs_baseline": paired,
           "importances": imp.to_dicts(), "importance_by_family": by_fam}
    a.out.mkdir(parents=True, exist_ok=True)
    f = a.out / f"ablation_{a.dataset}_{a.variant}.json"
    f.write_text(json.dumps(out, indent=2))
    print(f"\n   wrote {f}")


if __name__ == "__main__":
    main()
