"""Q2: train the re-ranker, and report the metrics before and after it runs.

"Before" is A1's best single signal on the same rows -- the embedding ranker,
which won both datasets in A1's Q4 table. That is the honest baseline: the
question Q2 asks is what *learning a combination* buys over the best hand-chosen
signal, not what it buys over nothing. `bm25` and the popularity prior are
reported alongside for the same reason.

Both framings of "two-stage" are measured, because the assignment's wording and
the leaderboards' scoring are not the same question:

  * **in-impression** -- re-order the candidate list the platform actually
    showed. This is what both Codabench graders score and what A1's Q4 table
    reports, so it is the number that carries over.
  * **corpus-wide** -- retrieve top-K from the live candidate universe with A1's
    generator, then re-rank those. This is Q2's literal text. It is reported by
    `scripts/q2_twostage.py`, which shares this model.

Writes reports/q2/rerank_<dataset>_<variant>.json and a feature-importance table.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import polars as pl

from newsrec.behaviour import assemble
from newsrec.evaluate import rank_metrics, summarise_ranking
from newsrec.rerank import (FAMILIES, SHIPPED, add_matching_features,
                            feature_names, importances, predict, train)
from newsrec.store import FeatureStore


def tuned_bm25(dataset: str, variant: str, q2_dir: Path = Path("reports/q2")) -> tuple[float, float]:
    """The (k1, b) A1 selected for this corpus, so the two assignments agree."""
    f = q2_dir / f"q2_bm25_{dataset}_{variant}.json"
    if not f.exists():
        print(f"   !! {f} missing -- BM25 defaults k1=1.5 b=1.0")
        return 1.5, 1.0
    best = json.loads(f.read_text())["best"]
    print(f"   BM25 at A1's operating point: k1={best['k1']}, b={best['b']}")
    return float(best["k1"]), float(best["b"])


def build_split(fs: FeatureStore, split: str, emb: str, k1: float, b: float,
                cache: Path, max_impressions: int = 0, rebuild: bool = False,
                history_mode: str = "shipped") -> pl.DataFrame:
    """Assemble (and cache) the design matrix for one split.

    Cached because every ablation in Q3 re-reads these frames and the assembly is
    the expensive half: BM25 over the whole corpus and a max-cosine per user are
    minutes, while fitting the model is seconds.
    """
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists() and not rebuild:
        df = pl.read_parquet(cache)
        print(f"   [{split}] {df.height:,} rows from cache {cache}")
        return df
    t0 = time.perf_counter()
    df = assemble(fs, split, max_impressions=max_impressions, history_mode=history_mode)
    df = add_matching_features(fs, split, df, emb, k1, b, history_mode)
    # LightGBM's ranking objectives take group *sizes*, so the frame has to be
    # grouped by impression before it is ever handed over
    df = df.sort("imp", "position")
    df.write_parquet(cache)
    print(f"   [{split}] {df.height:,} rows over {df['imp'].n_unique():,} impressions "
          f"in {time.perf_counter() - t0:.1f}s -> {cache}")
    return df


def evaluate(df: pl.DataFrame, scores: np.ndarray, n_boot: int) -> dict:
    pairs = df.select("imp", "article_idx", "label").with_columns(
        pl.Series("score", scores))
    per_imp = rank_metrics(pairs, ks=(5, 10))
    return summarise_ranking(per_imp, n_boot=n_boot), per_imp


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["ebnerd", "mind"])
    ap.add_argument("--variant", default="small")
    ap.add_argument("--embedding", default=None,
                    help="default: contrastive (EB-NeRD) / MiniLM (MIND)")
    ap.add_argument("--objective", default="lambdarank",
                    choices=["lambdarank", "rank_xendcg", "binary"])
    ap.add_argument("--families", default=",".join(SHIPPED),
                    help="comma-separated subset of " + ",".join(FAMILIES))
    ap.add_argument("--rounds", type=int, default=600)
    ap.add_argument("--max-impressions", type=int, default=0)
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", type=Path, default=Path("reports/q2"))
    a = ap.parse_args()

    emb = a.embedding or ("contrastive" if a.dataset == "ebnerd"
                          else "sentence-transformers/all-MiniLM-L6-v2")
    fs = FeatureStore(a.dataset, a.variant)
    print(f"[{a.dataset}/{a.variant}] {fs}\n   embedding: {emb}")
    k1, b = tuned_bm25(a.dataset, a.variant)

    root = Path("artifacts/features") / a.dataset / a.variant
    frames = {
        s: build_split(fs, s, emb, k1, b, root / f"{s}.parquet",
                       a.max_impressions if s == "train" else 0, a.rebuild)
        for s in ("train", "val", "test")
    }

    fams = tuple(x for x in a.families.split(",") if x)
    feats = feature_names(frames["train"], fams)
    print(f"\n   {len(feats)} features over families {fams}:")
    for fam in fams:
        got = [c for c in FAMILIES[fam] if c in feats]
        missing = [c for c in FAMILIES[fam] if c not in feats]
        print(f"     {fam:8s} {len(got):2d} used" +
              (f"   ({len(missing)} absent on this dataset: {','.join(missing)})"
               if missing else ""))

    t0 = time.perf_counter()
    model = train(frames["train"], frames["val"], feats,
                  objective=a.objective, num_boost_round=a.rounds)
    fit_s = time.perf_counter() - t0
    print(f"\n   fitted {model.best_iteration} trees in {fit_s:.1f}s "
          f"(best val score {model.best_score['val']})")

    test = frames["test"]
    results = {}

    # --- before: the best single hand-chosen signal on identical rows
    for name, col in [("emb", "emb_cos"), ("bm25", "bm25"),
                      ("pop_prior", "clicks_decayed"), ("ctr_prior", "ctr_smoothed")]:
        s = test[col].to_numpy().astype(np.float64)
        results[name], _ = evaluate(test, np.nan_to_num(s), a.n_boot)

    # --- after
    t0 = time.perf_counter()
    s_rr = predict(model, test, feats)
    score_s = time.perf_counter() - t0
    results["rerank"], per_imp_rr = evaluate(test, s_rr, a.n_boot)

    imp = importances(model, feats)
    print("\n   top features by gain:")
    for r in imp.head(15).iter_rows(named=True):
        print(f"     {r['share']:6.1%}  {r['family']:8s} {r['feature']}")
    print("\n   gain by family:")
    by_fam = imp.group_by("family").agg(pl.col("share").sum()).sort("share", descending=True)
    for r in by_fam.iter_rows(named=True):
        print(f"     {r['share']:6.1%}  {r['family']}")

    print(f"\n   {'ranker':<12} {'AUC':>8} {'MRR':>8} {'nDCG@5':>8} {'nDCG@10':>8}")
    print("   " + "-" * 48)
    for name in ("pop_prior", "ctr_prior", "bm25", "emb", "rerank"):
        r = results[name]
        print(f"   {name:<12} {r['auc']:>8.4f} {r['mrr']:>8.4f} "
              f"{r['ndcg@5']:>8.4f} {r['ndcg@10']:>8.4f}")

    out = {
        "dataset": a.dataset, "variant": a.variant, "embedding": emb,
        "objective": a.objective, "families": list(fams),
        "n_features": len(feats), "features": feats,
        "bm25": {"k1": k1, "b": b},
        "n_train_rows": frames["train"].height,
        "n_test_rows": test.height,
        "n_test_impressions": int(test["imp"].n_unique()),
        "best_iteration": int(model.best_iteration),
        "fit_seconds": round(fit_s, 1),
        "score_seconds": round(score_s, 2),
        "results": results,
        "importances": imp.to_dicts(),
        "importance_by_family": by_fam.to_dicts(),
    }
    a.out.mkdir(parents=True, exist_ok=True)
    tag = f"_{a.tag}" if a.tag else ""
    f = a.out / f"rerank_{a.dataset}_{a.variant}{tag}.json"
    f.write_text(json.dumps(out, indent=2))
    model.save_model(str(a.out / f"model_{a.dataset}_{a.variant}{tag}.txt"))
    print(f"\n   wrote {f}")


if __name__ == "__main__":
    main()
