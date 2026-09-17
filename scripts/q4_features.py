"""Q4 ablation: which user and article features actually help the ranking?

Q1 built a feature store -- per-user recency, activity and a category profile;
per-article smoothed CTR, decayed popularity, age, and content flags -- and then
Q2/Q3 ranked with none of it. This closes that loop: every feature is scored on
its own, and again as an addition to the embedding ranker, so the table reports
marginal contribution rather than a single fused number.

Features are z-scored *within each impression* before they are summed. That is
the only combination that is meaningful here: raw BM25 and cosine live on
different scales, and a user-level feature is constant inside an impression, so
un-normalised addition would just re-weight by whichever feature has the larger
variance.

`user_activity` is deliberately included and is expected to be useless: it is
constant across the candidates of one impression, so it cannot reorder them. It
is the negative control that shows the harness is measuring ranking rather than
anything about the user.
"""
from __future__ import annotations

import argparse, json, time
from pathlib import Path

import numpy as np
import polars as pl

from newsrec.embeddings import article_embeddings
from newsrec.evaluate import METRICS, rank_metrics, summarise_ranking
from newsrec.config import N_RECENT_DEFAULT
from newsrec.lexical import BM25Index
from newsrec.retrieval import user_histories
from newsrec.semantic import user_vectors
from newsrec.store import FeatureStore

from q4_eval import article_vector, build_pairs, tuned_bm25

KS = (5, 10)


def zscore_within(pairs: pl.DataFrame, name: str, x: np.ndarray) -> np.ndarray:
    """Standardise a feature inside each impression."""
    d = pairs.select("imp").with_columns(pl.Series("_x", x.astype(np.float64)))
    d = d.with_columns(
        ((pl.col("_x") - pl.col("_x").mean().over("imp"))
         / pl.col("_x").std().over("imp").fill_null(0.0).replace(0.0, 1.0)).alias("_z"))
    return np.nan_to_num(d["_z"].to_numpy(), nan=0.0)


def build_features(fs: FeatureStore, split: str, pairs: pl.DataFrame, emb_name: str,
                   k1: float, b: float,
                   n_recent: int = N_RECENT_DEFAULT) -> dict[str, np.ndarray]:
    n_art = fs.n_articles
    art = pairs["article_idx"].to_numpy()
    f: dict[str, np.ndarray] = {}

    # ---- article features from the store (prior window only)
    af = fs.article_features(split)
    f["pop_decayed"] = article_vector(af, "article_idx", "clicks_decayed", n_art)[art]
    f["ctr_smoothed"] = article_vector(af, "article_idx", "ctr_smoothed", n_art)[art]

    stats = (fs.article_stats()
             .join(fs.articles().with_row_index("idx").select("idx", "published_time"),
                   on="idx", how="left")
             .with_columns(pl.coalesce(pl.col("published_time"), pl.col("first_seen_time"))
                           .dt.epoch("s").cast(pl.Float64).alias("t")))
    art_t = article_vector(stats, "idx", "t", n_art, fill=np.nan)
    imp_t = pairs["time"].dt.epoch("s").to_numpy().astype(np.float64)
    f["freshness"] = -(imp_t - np.nan_to_num(art_t[art], nan=-1e12))

    # ---- article content features, whichever the dataset actually has.
    # EB-NeRD ships sentiment and a paywall flag; MIND ships neither. A feature a
    # dataset genuinely lacks is dropped from its table rather than faked as zero,
    # which would otherwise show up as a confidently useless feature.
    have = set(fs.articles().collect_schema().names())
    optional = [c for c in ("sentiment_score", "has_abstract", "premium") if c in have]
    a = fs.articles().with_row_index("idx").select(
        "idx", "category",
        *[pl.col(c).cast(pl.Float64).fill_null(0.0) for c in optional]).collect()
    for col in optional:
        v = np.zeros(n_art); v[a["idx"].to_numpy()] = a[col].to_numpy()
        if np.unique(v).size > 1:          # a constant column cannot rank anything
            f[col] = v[art]

    # ---- user x article: does the candidate sit in the user's category profile?
    uf = fs.user_features(split).select("user_idx", "top_categories").collect()
    cat = pairs.select("article_idx", "user_idx").join(
        a.select(pl.col("idx").alias("article_idx"), "category"), on="article_idx", how="left"
    ).join(uf, on="user_idx", how="left").with_columns(
        pl.col("top_categories").fill_null([]),
        pl.col("category").fill_null(""))
    f["category_match"] = cat.select(
        pl.col("category").is_in(pl.col("top_categories")).cast(pl.Float64)
    ).to_series().to_numpy()

    # ---- user-level activity: the negative control (constant within an impression)
    ua = fs.user_features(split).select("user_idx", "clicks_7d").collect()
    m = np.zeros(int(pairs["user_idx"].max()) + 1)
    m[ua["user_idx"].to_numpy()] = ua["clicks_7d"].to_numpy().astype(np.float64)
    f["user_activity"] = m[pairs["user_idx"].to_numpy()]

    # ---- the two content retrievers, on identical history
    uids, hists = user_histories(fs, split)
    urow = np.full(int(max(pairs["user_idx"].max(), uids.max())) + 1, -1, dtype=np.int64)
    urow[uids] = np.arange(len(uids))
    pu = urow[pairs["user_idx"].to_numpy()]
    known = pu >= 0

    idx = BM25Index.build(fs.texts(), lang=fs.lang()).reweight(k1=k1, b=b)
    Q = idx.queries_from_history(hists, n_recent=n_recent)
    s = np.zeros(len(art), dtype=np.float64)
    s[known] = idx.score_pairs(Q, pu[known], art[known])
    f["bm25"] = s

    emb = np.asarray(article_embeddings(fs, emb_name))
    uv = user_vectors(hists, emb, n_recent=n_recent)
    embn = emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12)
    s = np.zeros(len(art), dtype=np.float64)
    s[known] = np.einsum("ij,ij->i", uv[pu[known]], embn[art[known]])
    f["emb"] = s
    return f


def main(dataset: str, variant: str, split: str, emb_name: str, k1: float | None, b: float | None,
         out_dir: Path, n_boot: int, max_impressions: int,
         n_recent: int = N_RECENT_DEFAULT):
    k1, b = tuned_bm25(dataset, variant, k1, b)
    fs = FeatureStore(dataset, variant)
    pairs = build_pairs(fs, split, max_impressions=max_impressions)
    n_imp = pairs["imp"].n_unique()
    print(f"== Q4 features {dataset}/{variant} {split}: {n_imp:,} impressions, "
          f"{pairs.height:,} candidate rows")

    t0 = time.perf_counter()
    raw = build_features(fs, split, pairs, emb_name, k1, b, n_recent)
    z = {k: zscore_within(pairs, k, v) for k, v in raw.items()}
    print(f"   {len(raw)} features in {time.perf_counter()-t0:.0f}s: {', '.join(raw)}")

    res = {"dataset": dataset, "variant": variant, "split": split,
           "n_impressions": int(n_imp), "n_pairs": int(pairs.height), "n_recent": n_recent,
           "k1": k1, "b": b,
           "features": list(raw), "n_boot": n_boot, "alone": {}, "added_to_emb": {},
           "combined": {}}

    def evaluate(score: np.ndarray) -> dict:
        p = pairs.select("imp", "article_idx", "label").with_columns(pl.Series("score", score))
        return summarise_ranking(rank_metrics(p, ks=KS), n_boot=n_boot)

    print(f"\n  {'feature':16s} {'AUC alone':>10s} {'nDCG@10':>9s}   "
          f"{'+emb AUC':>9s} {'Δ vs emb':>9s}")
    base = evaluate(z["emb"])
    res["emb_baseline"] = base
    for name in raw:
        a = evaluate(z[name])
        res["alone"][name] = a
        if name == "emb":
            print(f"  {name:16s} {a['auc']:10.4f} {a['ndcg@10']:9.4f}   "
                  f"{'--':>9s} {'--':>9s}")
            continue
        c = evaluate(z["emb"] + z[name])
        res["added_to_emb"][name] = c
        print(f"  {name:16s} {a['auc']:10.4f} {a['ndcg@10']:9.4f}   "
              f"{c['auc']:9.4f} {c['auc']-base['auc']:+9.4f}")

    allf = sum(z[k] for k in raw)
    res["combined"]["all_features"] = evaluate(allf)
    helpful = [k for k in raw if k == "emb"
               or res["added_to_emb"][k]["auc"] > base["auc"]]
    res["combined"]["helpful_only"] = evaluate(sum(z[k] for k in helpful))
    res["combined"]["helpful_features"] = helpful
    print(f"\n  all {len(raw)} features   AUC {res['combined']['all_features']['auc']:.4f}")
    print(f"  the {len(helpful)} that each helped   AUC "
          f"{res['combined']['helpful_only']['auc']:.4f}  ({', '.join(helpful)})")

    out_dir.mkdir(parents=True, exist_ok=True)
    f = out_dir / f"features_{dataset}_{variant}.json"
    f.write_text(json.dumps(res, indent=2))
    print(f"== wrote {f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--variant", default="small")
    ap.add_argument("--split", default="test")
    ap.add_argument("--embedding", default="contrastive")
    ap.add_argument("--k1", type=float, default=None,
                    help="default: whatever Q2 selected for this corpus")
    ap.add_argument("--b", type=float, default=None,
                    help="default: whatever Q2 selected for this corpus")
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--max-impressions", type=int, default=0)
    ap.add_argument("--n-recent", type=int, default=N_RECENT_DEFAULT,
                    help="clicks read per user (0 = the whole history, the setting scripts/q3_userrep.py picked)")
    ap.add_argument("--out", default="reports/q4")
    a = ap.parse_args()
    main(a.dataset, a.variant, a.split, a.embedding, a.k1, a.b, Path(a.out),
         a.n_boot, a.max_impressions, a.n_recent)
