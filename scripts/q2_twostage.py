"""Q2: the full two-stage pipeline, reported in both framings.

Runs stage one over the live candidate universe, trains the re-ranker on what it
retrieves, and then scores two different things with the same model:

  * the **retrieved** sets -- the cascade a serving system actually runs, where
    "before" is stage one's own fused ranking;
  * the **shown** candidate lists -- the question both leaderboards score, where
    "before" is A1's best single signal on identical rows.

K is swept without re-retrieving: top-200 contains top-100 contains top-50 under
one ranking, so the deeper run is sliced rather than recomputed.

Writes reports/q2/twostage_<dataset>_<variant>.json.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import polars as pl

from newsrec.behaviour import attach_features
from newsrec.evaluate import rank_metrics, summarise_ranking
from newsrec.rerank import (FAMILIES, SHIPPED, add_matching_features,
                            feature_names, importances, predict, train)
from newsrec.store import FeatureStore
from newsrec.twostage import retrieved_pairs, stage1, stage1_recall

# `n_candidates` is the size of the list being scored. In the retrieved framing
# that is K for every row, so it carries no information at training time and a
# completely different meaning at in-impression scoring time. Excluding it is
# cheaper than explaining a feature that means two things.
DROP = ("n_candidates",)


def build(fs, split, k, emb, k1, b, cache: Path, rebuild: bool,
          method: str, max_impressions: int = 0,
          union_clicked: bool = True, drop_empty: bool = True) -> tuple[pl.DataFrame, dict]:
    if cache.exists() and not rebuild:
        df = pl.read_parquet(cache)
        meta = json.loads(cache.with_suffix(".json").read_text())
        print(f"   [{split}] {df.height:,} rows from cache")
        return df, meta

    t0 = time.perf_counter()
    uids, topk, sc = stage1(fs, split, k, method=method, emb_name=emb, k1=k1, b=b)
    t_ret = time.perf_counter() - t0
    print(f"   [{split}] stage1 {method} top-{k} for {len(uids):,} users in {t_ret:.1f}s")

    p = retrieved_pairs(fs, split, uids, topk, sc, max_impressions=max_impressions,
                        union_clicked=union_clicked)
    rec = stage1_recall(p, fs, split)
    if not union_clicked and drop_empty:
        # For TRAINING: a ranking objective silently drops a group with no
        # positive, so dropping them here makes the discard visible in the row
        # count instead of invisible inside LightGBM.
        # For EVALUATION `drop_empty` is False -- those impressions are exactly
        # the ones stage one failed on, and deleting them would report the
        # cascade's precision while hiding its recall.
        keep = p.group_by("imp").agg(pl.col("label").sum().alias("_p")).filter(pl.col("_p") > 0)
        p = p.join(keep.select("imp"), on="imp", how="inner")
        rec["impressions_kept"] = int(p["imp"].n_unique())
    print(f"   [{split}] {p.height:,} rows; stage-1 recall {rec['recall']:.4f}; "
          f"{rec['impressions_with_no_positive_retrieved']:.1%} of impressions "
          f"had no positive retrieved (clicked unioned back in)")

    df = attach_features(fs, split, p)
    df = add_matching_features(fs, split, df, emb, k1, b)
    df = df.sort("imp")
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(cache)
    meta = {"stage1_seconds": round(t_ret, 2), "k": k, "method": method,
            "union_clicked": union_clicked, **rec}
    cache.with_suffix(".json").write_text(json.dumps(meta))
    print(f"   [{split}] features in {time.perf_counter() - t0:.1f}s -> {cache}")
    return df, meta


def ev(df, scores, n_boot):
    pairs = df.select("imp", "article_idx", "label").with_columns(
        pl.Series("score", np.asarray(scores, dtype=np.float64)))
    return summarise_ranking(rank_metrics(pairs, ks=(5, 10)), n_boot=n_boot)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["ebnerd", "mind"])
    ap.add_argument("--variant", default="small")
    ap.add_argument("--embedding", default=None)
    ap.add_argument("--k", type=int, default=200)
    # Q2.1 states K ~ 100-200, so that is the range swept. 50 is available as an
    # extra point but is not part of the reported sweep unless asked for.
    ap.add_argument("--k-sweep", default="100,200")
    ap.add_argument("--method", default="hybrid", choices=["bm25", "emb", "hybrid"])
    ap.add_argument("--objective", default="lambdarank")
    ap.add_argument("--rounds", type=int, default=600)
    ap.add_argument("--n-boot", type=int, default=300)
    ap.add_argument("--max-impressions", type=int, default=60_000,
                    help="subsample impressions (never candidates within one). "
                         "K=200 over EB-NeRD's 200k train impressions is 40M rows "
                         "per split before features; the cascade's conclusions do "
                         "not need every impression and its CIs are already "
                         "narrower than the differences being measured.")
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--compare-training-sets", action="store_true", default=True,
                    help="also train on retrieved-without-union and on the shown "
                         "lists, and score all three on both framings")
    ap.add_argument("--out", type=Path, default=Path("reports/q2"))
    a = ap.parse_args()

    emb = a.embedding or ("contrastive" if a.dataset == "ebnerd"
                          else "sentence-transformers/all-MiniLM-L6-v2")
    fs = FeatureStore(a.dataset, a.variant)
    bm = json.loads((Path("reports/q2") / f"q2_bm25_{a.dataset}_{a.variant}.json").read_text())["best"]
    k1, b = float(bm["k1"]), float(bm["b"])
    print(f"[{a.dataset}/{a.variant}] {fs}\n   embedding {emb}; BM25 k1={k1} b={b}; "
          f"stage1={a.method} K={a.k}")

    root = Path("artifacts/twostage") / a.dataset / a.variant
    frames, metas = {}, {}
    for s in ("train", "val", "test"):
        frames[s], metas[s] = build(fs, s, a.k, emb, k1, b,
                                    root / f"{s}_k{a.k}_{a.method}_n{a.max_impressions}.parquet",
                                    a.rebuild, a.method, a.max_impressions)

    feats = [c for c in feature_names(frames["train"], SHIPPED) if c not in DROP]
    print(f"\n   {len(feats)} features (excluding {', '.join(DROP)})")

    # Three candidate sets to train the SAME model on, so that "which candidate
    # set should stage two learn from" becomes a measurement rather than a
    # preference. All three are scored on both framings below.
    models: dict[str, object] = {}

    t0 = time.perf_counter()
    models["retrieved+union"] = train(frames["train"], frames["val"], feats,
                                      objective=a.objective, num_boost_round=a.rounds)
    print(f"   [retrieved+union] {models['retrieved+union'].best_iteration} trees "
          f"in {time.perf_counter() - t0:.1f}s")

    if a.compare_training_sets:
        nou = {}
        for s in ("train", "val"):
            nou[s], _ = build(fs, s, a.k, emb, k1, b,
                              root / f"{s}_k{a.k}_{a.method}_n{a.max_impressions}_nounion.parquet",
                              a.rebuild, a.method, a.max_impressions, union_clicked=False)
        t0 = time.perf_counter()
        models["retrieved-only"] = train(nou["train"], nou["val"], feats,
                                         objective=a.objective, num_boost_round=a.rounds)
        print(f"   [retrieved-only] {models['retrieved-only'].best_iteration} trees "
              f"in {time.perf_counter() - t0:.1f}s "
              f"({nou['train'].height:,} rows, impressions whose positive was "
              f"genuinely retrieved)")

        shown_tr = Path("artifacts/features") / a.dataset / a.variant
        if (shown_tr / "train.parquet").exists():
            st = pl.read_parquet(shown_tr / "train.parquet").sort("imp")
            sv = pl.read_parquet(shown_tr / "val.parquet").sort("imp")
            t0 = time.perf_counter()
            models["shown"] = train(st, sv, feats, objective=a.objective,
                                    num_boost_round=a.rounds)
            print(f"   [shown] {models['shown'].best_iteration} trees "
                  f"in {time.perf_counter() - t0:.1f}s ({st.height:,} rows)")

    out = {"dataset": a.dataset, "variant": a.variant, "embedding": emb,
           "stage1": {"method": a.method, "k": a.k, **metas["test"]},
           "objective": a.objective, "features": feats,
           "trained_on": "stage1-retrieved sets (clicked unioned in)",
           "best_iteration": {n: int(m.best_iteration) for n, m in models.items()}}

    # ---------------- framing A: the cascade, end to end
    #
    # Scored on stage one's real top-K with nothing injected, and with every
    # impression kept -- including the ~79% where retrieval returned no clicked
    # article at all, which score 0. That is what the cascade actually delivers:
    # stage two cannot rank what stage one did not retrieve, so the end-to-end
    # number is bounded by stage-1 recall and says so.
    #
    # The union'd test set is kept only as a diagnostic. Its positives are there
    # because they were clicked, so any feature recording "appeared in a real
    # impression" predicts the label directly and every model scores absurdly
    # well on it. It is reported to show the size of that illusion, never as a
    # result.
    honest, honest_meta = build(fs, "test", a.k, emb, k1, b,
                                root / f"test_k{a.k}_{a.method}_n{a.max_impressions}_nounion.parquet",
                                a.rebuild, a.method, a.max_impressions,
                                union_clicked=False, drop_empty=False)
    out["stage1"]["recall_on_test"] = honest_meta.get("recall")
    cascade = {"stage1_order": ev(honest, honest["stage1_score"].to_numpy(), a.n_boot)}
    for name, m in models.items():
        cascade[f"rerank[{name}]"] = ev(honest, predict(m, honest, feats), a.n_boot)

    test = frames["test"]
    diagnostic = {"stage1_order": ev(test, test["stage1_score"].to_numpy(), a.n_boot)}
    for name, m in models.items():
        diagnostic[f"rerank[{name}]"] = ev(test, predict(m, test, feats), a.n_boot)
    out["cascade_union_diagnostic"] = diagnostic

    # K sweep by slicing the stage-1 ranking rather than re-retrieving
    sweep = {}
    ranked = honest.with_columns(
        pl.col("stage1_score").rank("ordinal", descending=True).over("imp").alias("_r"))
    # The sweep is a cascade measurement, so it uses the model trained on the
    # cascade's own candidate distribution. Scoring it with the shown-trained
    # model would measure the train/serve mismatch, not the effect of K.
    best = ("retrieved-only" if "retrieved-only" in models
            else next(iter(models)))
    for k in [int(x) for x in a.k_sweep.split(",") if int(x) <= a.k]:
        sub = ranked.filter(pl.col("_r") <= k)
        # recall@K: the funnel's upstream duty, and the ceiling on every
        # downstream number at this K
        rec = stage1_recall(sub, fs, "test")
        sweep[k] = {
            "n_rows": sub.height,
            "trained_on": best,
            "stage1_recall": rec["recall"],
            "impressions_missed": rec["impressions_with_no_positive_retrieved"],
            "stage1_order": ev(sub, sub["stage1_score"].to_numpy(), a.n_boot),
            "rerank": ev(sub, predict(models[best], sub, feats), a.n_boot),
        }
    out["cascade"] = cascade
    out["k_sweep"] = sweep

    # ---------------- framing B: in-impression, the leaderboard's question
    shown = Path("artifacts/features") / a.dataset / a.variant / "test.parquet"
    if shown.exists():
        sh = pl.read_parquet(shown)
        inimp = {
            "emb": ev(sh, np.nan_to_num(sh["emb_cos"].to_numpy()), a.n_boot),
            "bm25": ev(sh, np.nan_to_num(sh["bm25"].to_numpy()), a.n_boot),
        }
        for name, m in models.items():
            inimp[f"rerank[{name}]"] = ev(sh, predict(m, sh, feats), a.n_boot)
        out["in_impression"] = inimp

    # ---------------- what the unexamined negatives are doing
    if "was_shown" in test.columns:
        neg = test.filter(~pl.col("label"))
        out["negatives"] = {
            "n": neg.height,
            "examined_frac": float(neg["was_shown"].mean()),
        }

    imp = importances(models["retrieved+union"], feats)
    out["importances"] = imp.to_dicts()
    out["importance_by_family"] = (
        imp.group_by("family").agg(pl.col("share").sum())
        .sort("share", descending=True).to_dicts())

    print(f"\n   CASCADE end-to-end (stage-1 top-{a.k}, nothing injected, "
          f"all impressions kept)")
    print(f"   {'':<26} {'AUC':>8} {'MRR':>8} {'nDCG@5':>8} {'nDCG@10':>8}")
    for n, r in cascade.items():
        print(f"   {n:<26} {r['auc']:>8.4f} {r['mrr']:>8.4f} {r['ndcg@5']:>8.4f} {r['ndcg@10']:>8.4f}")
    print(f"\n   [diagnostic] same models on the union'd set -- the illusion:")
    for n, r in diagnostic.items():
        print(f"   {n:<26} {r['auc']:>8.4f} {r['mrr']:>8.4f} {r['ndcg@5']:>8.4f} {r['ndcg@10']:>8.4f}")
    print(f"\n   K sweep -- Q2.1 range, cascade trained on {best}")
    print(f"   {'K':>5} {'recall@K':>10} {'missed':>8} {'nDCG@10 before':>15} {'after':>8}")
    for k, v in sweep.items():
        print(f"   {k:>5} {v['stage1_recall']:>10.4f} {v['impressions_missed']:>8.1%} "
              f"{v['stage1_order']['ndcg@10']:>15.4f} {v['rerank']['ndcg@10']:>8.4f}")
    if "in_impression" in out:
        print(f"\n   IN-IMPRESSION (shown lists -- the leaderboard's question)")
        print(f"   {'':<26} {'AUC':>8} {'MRR':>8} {'nDCG@5':>8} {'nDCG@10':>8}")
        for n, r in out["in_impression"].items():
            print(f"   {n:<26} {r['auc']:>8.4f} {r['mrr']:>8.4f} "
                  f"{r['ndcg@5']:>8.4f} {r['ndcg@10']:>8.4f}")
    if "negatives" in out:
        print(f"\n   negatives that were actually shown to someone: "
              f"{out['negatives']['examined_frac']:.1%} of {out['negatives']['n']:,}")

    a.out.mkdir(parents=True, exist_ok=True)
    f = a.out / f"twostage_{a.dataset}_{a.variant}.json"
    f.write_text(json.dumps(out, indent=2))
    for name, m in models.items():
        tag = name.replace("+", "_").replace("-", "_")
        m.save_model(str(a.out / f"model_twostage_{tag}_{a.dataset}_{a.variant}.txt"))
    print(f"\n   wrote {f}")


if __name__ == "__main__":
    main()
