"""Q1: how much of each feature is actually populated, per dataset and split.

A feature that is null or zero on most rows is not a feature, and a re-ranker
that appears to ignore a family may be ignoring an empty column rather than an
uninformative one. Those two diagnoses call for opposite responses -- one is a
finding about the dataset, the other is a bug in the join -- so the coverage
table is produced before either is claimed.

Writes reports/q1/coverage_<dataset>_<variant>.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl

from newsrec.rerank import FAMILIES


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["ebnerd", "mind"])
    ap.add_argument("--variant", default="small")
    ap.add_argument("--splits", default="train,test")
    ap.add_argument("--out", type=Path, default=Path("reports/q1"))
    a = ap.parse_args()

    fam_of = {c: f for f, cs in FAMILIES.items() for c in cs}
    report = {"dataset": a.dataset, "variant": a.variant, "splits": {}}

    for split in a.splits.split(","):
        p = Path("artifacts/features") / a.dataset / a.variant / f"{split}.parquet"
        if not p.exists():
            print(f"   !! {p} missing -- run q2_rerank.py first")
            continue
        df = pl.read_parquet(p)
        n = df.height
        print(f"\n[{a.dataset}/{a.variant}:{split}] {n:,} rows, "
              f"{df['imp'].n_unique():,} impressions, "
              f"{df['user_idx'].n_unique():,} users")
        print(f"   {'feature':<24} {'family':<8} {'null%':>7} {'zero%':>7} {'distinct':>9}")
        print("   " + "-" * 60)

        rows = []
        for c in df.columns:
            if c not in fam_of:
                continue
            s = df[c]
            nulls = s.null_count() / n
            try:
                zeros = float((s.fill_null(0) == 0).sum()) / n
            except Exception:
                zeros = float("nan")
            d = s.n_unique()
            rows.append({"feature": c, "family": fam_of[c],
                         "null_frac": round(nulls, 4), "zero_frac": round(zeros, 4),
                         "n_distinct": int(d)})
            flag = "  <-- dead" if (nulls + zeros) > 0.98 or d <= 1 else ""
            print(f"   {c:<24} {fam_of[c]:<8} {nulls:>6.1%} {zeros:>7.1%} {d:>9,}{flag}")

        # how many rows have *any* prior behavioural evidence about the user
        cold = df.select(
            (pl.col("n_hist").is_null() | (pl.col("n_hist") == 0)).mean().alias("no_history"),
            (pl.col("hours_since_last_click").is_null()).mean().alias("no_click_time"),
            (pl.col("clicks_decayed") == 0).mean().alias("article_unclicked_before"),
            (pl.col("prior_inview") == 0).mean().alias("article_unseen_before"),
        ).to_dicts()[0]
        print(f"\n   rows whose user has no history:          {cold['no_history']:.1%}")
        print(f"   rows whose user has no dated prior click: {cold['no_click_time']:.1%}")
        print(f"   rows whose article had 0 prior clicks:    {cold['article_unclicked_before']:.1%}")
        print(f"   rows whose article had 0 prior exposures: {cold['article_unseen_before']:.1%}")

        report["splits"][split] = {"n_rows": n, "features": rows, "emptiness": cold}

    a.out.mkdir(parents=True, exist_ok=True)
    f = a.out / f"coverage_{a.dataset}_{a.variant}.json"
    f.write_text(json.dumps(report, indent=2))
    print(f"\n   wrote {f}")


if __name__ == "__main__":
    main()
