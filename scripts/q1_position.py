"""Q1: is position bias actually in this data, and how much of it is attention?

The lecture asserts a shape -- click rate collapses with rendered position, and
the collapse survives holding the item fixed, so it is attention rather than
quality. Before correcting for it, measure it: the correction is only principled
if the confound is there, and it is not obviously there on both datasets.

The thing that decides it is whether the stored candidate list preserves the
order the user saw. EB-NeRD's `article_ids_inview` is documented as the rendered
list, and A1 already relied on that (its tie-breaking jitter exists precisely so
a constant-scoring baseline cannot cash in the platform's layout). MIND's
`impressions` column is a list of `N12345-0` / `N12345-1` tokens whose order the
dataset paper does not promise. So this script reports the curve per dataset and
lets the numbers say which log carries the signal.

Writes reports/q1/position_<dataset>_<variant>.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl

from newsrec.behaviour import pairs
from newsrec.bias import position_bias_report
from newsrec.store import FeatureStore


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["ebnerd", "mind"])
    ap.add_argument("--variant", default="small")
    ap.add_argument("--split", default="train")
    ap.add_argument("--max-pos", type=int, default=0,
                    help="0 = the 99th percentile of candidate-list length")
    ap.add_argument("--min-shown", type=int, default=50)
    ap.add_argument("--max-impressions", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("reports/q1"))
    a = ap.parse_args()

    fs = FeatureStore(a.dataset, a.variant)
    print(f"[{a.dataset}/{a.variant}:{a.split}] {fs}")

    p = pairs(fs, a.split, max_impressions=a.max_impressions)
    print(f"   {p.height:,} (impression, candidate) rows "
          f"over {p['imp'].n_unique():,} impressions")

    max_pos = a.max_pos
    if not max_pos:
        # a long candidate list is rare and its deep positions are estimated from
        # a handful of impressions; cutting at p99 keeps the curve supported
        max_pos = int(p.select(pl.col("n_candidates").quantile(0.99)).item())
        print(f"   candidate lists: p99 = {max_pos}, "
              f"max = {p['n_candidates'].max()}")

    rep = position_bias_report(p, max_pos=max_pos, min_shown=a.min_shown)
    rep |= {"dataset": a.dataset, "variant": a.variant, "split": a.split,
            "n_pairs": p.height, "n_impressions": int(p["imp"].n_unique())}

    obs = rep["observed_ctr"]
    strat = {d["position"]: d for d in rep["stratified"]}
    prop = {d["position"]: d for d in rep["propensity"]}
    print(f"\n   {'pos':>4} {'shown':>10} {'rawCTR':>8} | {'ratio|L':>8} {'lens':>5} "
          f"| {'prop':>7} {'articles':>8}")
    print("   " + "-" * 62)
    for row in obs[:16]:
        k = row["position"]
        s, pr = strat.get(k, {}), prop.get(k, {})
        print(f"   {k:>4} {row['n_shown']:>10,} {row['ctr']:>8.4f} | "
              f"{s.get('ratio', float('nan')):>8.4f} {s.get('n_lengths', 0):>5} | "
              f"{pr.get('propensity', float('nan')):>7.4f} "
              f"{pr.get('n_articles', 0):>8,}")

    print(f"\n   raw CTR decay (last/first):            {rep.get('observed_decay'):.4f}"
          if rep.get("observed_decay") is not None else "")
    print(f"   length-stratified ratio range:         "
          f"[{rep.get('stratified_ratio_min'):.4f}, {rep.get('stratified_ratio_max'):.4f}]")
    print(f"   largest deviation from 1.0, in SEs:    {rep.get('stratified_max_z'):.1f}")
    print()
    print("   Reading: the raw column mixes position with list length -- these are")
    print("   near-single-click logs, so per-slot CTR is ~1/L before any behaviour.")
    print("   `ratio|L` holds L fixed; under 'stored order is not rendered order'")
    print("   it is 1.0 everywhere. That is the hypothesis this number tests.")

    a.out.mkdir(parents=True, exist_ok=True)
    f = a.out / f"position_{a.dataset}_{a.variant}.json"
    f.write_text(json.dumps(rep, indent=2))
    print(f"\n   wrote {f}")


if __name__ == "__main__":
    main()
