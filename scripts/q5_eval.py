"""Q5: the full evaluation -- every metric, two slices, CIs on all of them.

Accuracy metrics come from A1's harness, which `tests/test_official_metrics.py`
validates against the graders' own `ebrec` implementation. The beyond-accuracy
three are A1's as well, and each answers a question no accuracy metric asks:

  * **diversity** -- intra-list distance in the embedding space, not over the
    category taxonomy. MIND's taxonomy is 18 labels wide and EB-NeRD's is a
    different taxonomy entirely, so a category-based index is not comparable
    across the two; the embedding space is the object the semantic retriever
    already ranks in.
  * **novelty** -- mean self-information of what was recommended, against the
    click distribution of the window *before* the scored split. Using in-window
    popularity would make a system look novel for surfacing what later turned
    out to be a hit.
  * **coverage** -- the share of the catalogue that appears in any list at all.
    A1 found the popularity prior reaching 34% of MIND's catalogue while
    scoring 0.430 on head impressions and 0.240 on tail ones; accuracy alone
    never says that.

**A confound this test cannot fully escape.** The head/tail slice is defined by
an article's exposure count, and the `article-log` arm is built from exposure
counts. Conditioning on a variable removes its variance inside the slice, so
within "head" every article already has high exposure and the feature has less
left to discriminate with. That is visible in the result -- article-log scores
*lower* on the head than on the tail, which is backwards from any causal story --
and it is a property of the slicing, not of the features. The comparison between
the two arms *within* a slice is still meaningful; the comparison of one arm
*across* slices is not.

**The slices are a test, not a table.** L7's head/torso/tail slide claims that
memorised behavioural statistics are unbeatable on the head and empty on the
tail, where only content survives. That is a prediction about *which family of
features wins where*, so this script trains a content-only and a
behaviour-only model on the same rows and compares them per slice. Reporting the
shipped model's metrics per slice would not test it.

Writes reports/q5/eval_<dataset>_<variant>.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl

from newsrec.embeddings import article_embeddings
from newsrec.evaluate import (METRICS, bootstrap, coverage, intra_list_diversity,
                              novelty, paired_report, rank_metrics,
                              summarise_ranking, topk_lists)
from newsrec.rerank import FAMILIES, SHIPPED, feature_names, predict, train
from newsrec.semantic import l2_normalise
from newsrec.store import FeatureStore

# L7 s.40's claim is about whether the *article* has logs, so the two arms are
# defined at column level rather than by family. Splitting by family muddles it:
# the `article` family holds `age_hours`, which is metadata available for an
# article with no clicks at all, and the `match` family holds `cat_*`, which is
# derived from the user's click log.
#
#   content     -- everything computable for an article nobody has ever clicked:
#                  lexical and semantic match, category affinity, and freshness.
#   article-log -- the memorised article statistics, which are exactly what a
#                  tail article does not have.
CONTENT_COLS = ("bm25", "emb_cos", "emb_max", "emb_recent",
                "cat_share", "cat_clicks", "cat_recency", "cat_recency_share",
                "age_hours", "roll_age_hours")
ARTICLE_LOG_COLS = ("ctr_smoothed", "clicks_decayed", "prior_clicks",
                    "prior_inview", "roll_inview")

COLD_THRESHOLD = 5      # A1's: keyed on history length, not "user seen in train"
HEAD_QUANTILE = 0.80    # an article is head if its prior exposure is top 20%


def beyond_accuracy(df: pl.DataFrame, scores: np.ndarray, emb: np.ndarray,
                    pop_share: np.ndarray, catalogue: int, k: int = 10) -> dict:
    pairs = df.select("imp", "article_idx").with_columns(
        pl.Series("score", np.asarray(scores, dtype=np.float64)))
    lists = topk_lists(pairs, k=k)
    rec = [np.asarray(r, dtype=np.int64) for r in lists["rec"].to_list()]
    return {
        f"ild@{k}": intra_list_diversity(rec, emb),
        f"novelty@{k}": novelty(rec, pop_share),
        f"coverage@{k}": coverage(rec, catalogue),
        "n_lists": len(rec),
    }


def summarise_with_ci(per_imp: pl.DataFrame, n_boot: int) -> dict:
    """Every accuracy metric with its 95% CI (Q5 requires CIs on all of them)."""
    return summarise_ranking(per_imp, n_boot=n_boot)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["ebnerd", "mind"])
    ap.add_argument("--variant", default="small")
    ap.add_argument("--embedding", default=None)
    ap.add_argument("--objective", default="lambdarank")
    ap.add_argument("--rounds", type=int, default=600)
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--out", type=Path, default=Path("reports/q5"))
    a = ap.parse_args()

    emb_name = a.embedding or ("contrastive" if a.dataset == "ebnerd"
                               else "sentence-transformers/all-MiniLM-L6-v2")
    fs = FeatureStore(a.dataset, a.variant)
    root = Path("artifacts/features") / a.dataset / a.variant
    tr, va, te = (pl.read_parquet(root / f"{s}.parquet").sort("imp")
                  for s in ("train", "val", "test"))

    emb = l2_normalise(np.asarray(article_embeddings(fs, emb_name), dtype=np.float32))
    # novelty's reference distribution: clicks strictly before the scored split
    af = fs.article_features("test").select("article_idx", "clicks").collect()
    pop = np.zeros(fs.n_articles, dtype=np.float64)
    pop[af["article_idx"].to_numpy()] = af["clicks"].to_numpy()
    pop_share = pop / max(pop.sum(), 1.0)

    print(f"[{a.dataset}/{a.variant}] {te.height:,} test rows over "
          f"{te['imp'].n_unique():,} impressions")

    # ---------------------------------------------------------------- models
    models = {}
    arms = {"shipped": feature_names(tr, SHIPPED),
            "content-only": [c for c in CONTENT_COLS if c in tr.columns],
            "article-log-only": [c for c in ARTICLE_LOG_COLS if c in tr.columns]}
    for name, f in arms.items():
        if not f:
            continue
        m = train(tr, va, f, objective=a.objective, num_boost_round=a.rounds,
                  verbose_eval=0)
        models[name] = (m, f, predict(m, te, f))
        print(f"   {name:<16} {len(f):>3} features, {m.best_iteration:>4} trees")

    # ------------------------------------------------------------- overall
    overall = {}
    for name, (m, f, s) in models.items():
        pi = rank_metrics(te.select("imp", "article_idx", "label").with_columns(
            pl.Series("score", s)), ks=(5, 10))
        overall[name] = summarise_with_ci(pi, a.n_boot)
        overall[name].update(beyond_accuracy(te, s, emb, pop_share, fs.n_articles))

    print(f"\n   Q5 all metrics, full pipeline (95% CI from bootstrap over impressions)")
    print(f"   {'model':<16} {'AUC':>8} {'MRR':>8} {'nDCG@5':>8} {'nDCG@10':>8} "
          f"{'ILD@10':>8} {'nov@10':>8} {'cov@10':>8}")
    for name, r in overall.items():
        print(f"   {name:<16} {r['auc']:>8.4f} {r['mrr']:>8.4f} {r['ndcg@5']:>8.4f} "
              f"{r['ndcg@10']:>8.4f} {r['ild@10']:>8.3f} {r['novelty@10']:>8.2f} "
              f"{r['coverage@10']:>8.3f}")

    # -------------------------------------------------------------- slices
    # cold vs warm, on history length -- A1's reason: MINDsmall's train and dev
    # user pools overlap by only 11.9%, so "seen in training" would label almost
    # every dev user cold and measure nothing.
    hist = fs.history("test").select("user_idx", pl.col("n_hist").alias("_nh")).collect()
    te2 = te.join(hist, on="user_idx", how="left").with_columns(
        pl.col("_nh").fill_null(0))
    nh = te2.group_by("imp").agg(pl.col("_nh").first())
    # A1's fixed threshold is the definition of "cold-start" worth reporting, but
    # it is degenerate on EB-NeRD, whose shipped history has a median of 93
    # clicks and where no test user falls under 5. A slice containing 100% of the
    # impressions measures nothing, so the *reported* split is the bottom decile
    # of history length, with the fixed-threshold count published next to it so
    # the reader can see which definition they are looking at.
    n_fixed_cold = int((nh["_nh"] < COLD_THRESHOLD).sum())
    q10 = float(np.quantile(nh["_nh"].to_numpy(), 0.10))
    cold_cut = max(COLD_THRESHOLD, q10) if n_fixed_cold < 0.01 * nh.height else COLD_THRESHOLD
    imp_cold = nh.with_columns((pl.col("_nh") <= cold_cut).alias("is_cold"))
    cold_def = {"fixed_threshold": COLD_THRESHOLD,
                "impressions_under_fixed_threshold": n_fixed_cold,
                "history_p10": q10, "threshold_used": cold_cut,
                "note": ("fixed threshold kept where it splits; widened to the "
                         "history p10 where it would leave the slice empty")}

    # Head vs tail on the article's popularity **as it stood at the impression**,
    # not on its pre-split exposure. 80.6% of EB-NeRD articles have zero prior
    # exposure because they did not exist when the window closed, so the pre-split
    # version splits on "did this article exist yet" and puts 97% of impressions
    # in the tail. `roll_inview` is the counter at request time and is populated
    # on 99.8% of rows.
    clicked = te.filter(pl.col("label")).select("imp", "article_idx", "roll_inview")
    ex_vals = clicked["roll_inview"].fill_null(0).to_numpy().astype(np.float64)
    thr = float(np.quantile(ex_vals, HEAD_QUANTILE)) if ex_vals.size else 0.0
    head_imp = (clicked.with_columns(pl.col("roll_inview").fill_null(0).alias("_ex"))
                .group_by("imp").agg(pl.col("_ex").max())
                .with_columns((pl.col("_ex") >= thr).alias("is_head")))

    slices = {
        "cold": imp_cold.filter(pl.col("is_cold"))["imp"],
        "warm": imp_cold.filter(~pl.col("is_cold"))["imp"],
        "head": head_imp.filter(pl.col("is_head"))["imp"],
        "tail": head_imp.filter(~pl.col("is_head"))["imp"],
    }

    sliced = {}
    print(f"\n   Q5 slices (cold: n_hist <= {cold_cut:.0f}, "
          f"{n_fixed_cold:,} impressions under the fixed threshold of "
          f"{COLD_THRESHOLD}; head: clicked article's exposure-at-request "
          f">= {thr:.0f}, top {100 * (1 - HEAD_QUANTILE):.0f}%)")
    print(f"   {'slice':<8} {'imps':>9} {'model':<16} {'AUC':>8} {'nDCG@10':>9}")
    for sname, imps in slices.items():
        sub_mask = te["imp"].is_in(imps.implode())
        sub = te.filter(sub_mask)
        if sub.height == 0:
            continue
        sliced[sname] = {"n_impressions": int(sub["imp"].n_unique())}
        for name, (m, f, s) in models.items():
            ss = np.asarray(s)[sub_mask.to_numpy()]
            pi = rank_metrics(sub.select("imp", "article_idx", "label").with_columns(
                pl.Series("score", ss)), ks=(5, 10))
            sliced[sname][name] = summarise_with_ci(pi, a.n_boot)
            print(f"   {sname:<8} {sliced[sname]['n_impressions']:>9,} {name:<16} "
                  f"{sliced[sname][name]['auc']:>8.4f} "
                  f"{sliced[sname][name]['ndcg@10']:>9.4f}")

    # the lecture's prediction, stated as a number
    verdict = {}
    for pair in (("head", "tail"),):
        for sname in pair:
            if sname in sliced and "content-only" in models and "article-log-only" in models:
                c = sliced[sname]["content-only"]["auc"]
                b = sliced[sname]["article-log-only"]["auc"]
                verdict[sname] = {"content_auc": c, "article_log_auc": b,
                                  "article_log_wins": bool(b > c)}
    if verdict:
        print(f"\n   L7 s.40 predicts: behaviour unbeatable on the head, "
              f"only content survives on the tail")
        for sname, v in verdict.items():
            print(f"     {sname:<6} content {v['content_auc']:.4f}  "
                  f"article-log {v['article_log_auc']:.4f}  -> "
                  f"{'article-log' if v['article_log_wins'] else 'content'} wins")

    out = {"dataset": a.dataset, "variant": a.variant, "embedding": emb_name,
           "cold_definition": cold_def, "head_quantile": HEAD_QUANTILE,
           "head_exposure_threshold": thr,
           "arms": {"content": list(CONTENT_COLS),
                    "article_log": list(ARTICLE_LOG_COLS)},
           "overall": overall, "slices": sliced, "head_tail_verdict": verdict}
    a.out.mkdir(parents=True, exist_ok=True)
    f = a.out / f"eval_{a.dataset}_{a.variant}.json"
    f.write_text(json.dumps(out, indent=2))
    print(f"\n   wrote {f}")


if __name__ == "__main__":
    main()
