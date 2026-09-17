"""Q4: run the evaluation harness over every ranker, on both datasets.

Each ranker is a scoring function over the (impression, candidate) pairs the
platform actually showed. That is the leaderboard's question and it is not the
question Q2/Q3 answered -- a retriever that never surfaces an article can still
rank it correctly once the impression puts it in front of the user.

Q9 (anti-gaming) is answered here rather than in a separate script: the
`pop_oracle` ranker is deliberately illegal -- it counts clicks from inside the
scored split, i.e. from the future -- and is reported next to its serving-safe
twin so the size of the illusion is a measured number instead of a claim.
"""
from __future__ import annotations

import argparse, json, time
from pathlib import Path

import numpy as np
import polars as pl

from newsrec.embeddings import article_embeddings
from newsrec.evaluate import (METRICS, coverage, intra_list_diversity, novelty,
                              rank_metrics, rank_profile, summarise_ranking, topk_lists)
from newsrec.config import N_RECENT_DEFAULT
from newsrec.lexical import BM25Index
from newsrec.retrieval import user_histories
from newsrec.semantic import user_vectors
from newsrec.store import FeatureStore


def tuned_bm25(dataset: str, variant: str, k1: float | None, b: float | None,
               q2_dir: Path = Path("reports/q2")) -> tuple[float, float]:
    """Use the (k1, b) that Q2 selected for this corpus unless told otherwise.

    Q2 tunes BM25 per corpus and the design note reports those values, so the Q4
    harness reading a different pair would make the comparison table quietly
    disagree with the table above it. Explicit --k1/--b still win.
    """
    if k1 is not None and b is not None:
        return k1, b
    f = q2_dir / f"q2_bm25_{dataset}_{variant}.json"
    if not f.exists():
        print(f"   !! {f} missing -- falling back to BM25 defaults k1=1.5 b=1.0")
        return (k1 if k1 is not None else 1.5), (b if b is not None else 1.0)
    best = json.loads(f.read_text())["best"]
    k1 = best["k1"] if k1 is None else k1
    b = best["b"] if b is None else b
    print(f"   BM25 at the Q2-selected operating point: k1={k1}, b={b}  ({f})")
    return k1, b

KS = (1, 3, 5, 10, 20)            # cutoff ablation: where does the ranking stop paying?
TOPK_BA = 10                      # list length the beyond-accuracy metrics see
RANK_CAP = 40                     # rank-profile histogram depth


# ------------------------------------------------------------------ pair table

def build_pairs(fs: FeatureStore, split: str, max_impressions: int = 0,
                seed: int = 0) -> pl.DataFrame:
    """Flatten labelled impressions into one row per shown candidate.

    `max_impressions` subsamples *impressions* (never candidates within one), so
    every sampled impression keeps the full candidate list the platform showed.
    EB-NeRD large's test split is 12.5M impressions over 150M candidate rows;
    scoring eight rankers over all of it is hours of wall clock for confidence
    intervals that are already narrower than the differences being measured.
    """
    imp = (
        fs.impressions(split)
        .filter(pl.col("clicked").list.len() > 0)
        .with_row_index("imp")
        .select("imp", "user_idx", "time", "candidates", "clicked")
    )
    if max_impressions:
        n = imp.select(pl.len()).collect().item()
        if n > max_impressions:
            keep = np.sort(np.random.default_rng(seed).choice(n, max_impressions, replace=False))
            imp = imp.filter(pl.col("imp").is_in(
                pl.Series(keep, dtype=pl.UInt32).implode()))
    return (
        imp.explode("candidates")  # empty candidate lists stay null, then drop
        .drop_nulls("candidates")
        .rename({"candidates": "article_idx"})
        .with_columns(pl.col("article_idx").is_in(pl.col("clicked")).alias("label"))
        .drop("clicked")
        .collect(engine="streaming")
    )


def article_vector(lf: pl.LazyFrame, key: str, value: str, n: int,
                   fill: float = 0.0) -> np.ndarray:
    """Dense per-article array from a sparse (article_idx, value) table."""
    out = np.full(n, fill, dtype=np.float32)
    d = lf.select(key, value).collect()
    out[d[key].to_numpy()] = d[value].to_numpy().astype(np.float32)
    return out


# -------------------------------------------------------------------- rankers

def make_rankers(fs: FeatureStore, split: str, pairs: pl.DataFrame, emb_name: str,
                 k1: float, b: float, history_mode: str,
                 n_recent: int = N_RECENT_DEFAULT) -> dict:
    n_art = fs.n_articles
    art = pairs["article_idx"].to_numpy()
    rankers, timings = {}, {}

    def timed(name, fn):
        t0 = time.perf_counter()
        rankers[name] = np.asarray(fn(), dtype=np.float32)
        timings[name] = round(time.perf_counter() - t0, 2)

    # --- serving-safe: nothing here reads a timestamp at or after the split start
    timed("random", lambda: np.random.default_rng(0).random(len(art)))

    af = fs.article_features(split)
    pop = article_vector(af, "article_idx", "clicks_decayed", n_art)
    ctr = article_vector(af, "article_idx", "ctr_smoothed", n_art)
    timed("pop_prior", lambda: pop[art])
    timed("ctr_prior", lambda: ctr[art])

    stats = (fs.article_stats()
             .join(fs.articles().with_row_index("idx").select("idx", "published_time"),
                   on="idx", how="left")
             .with_columns(pl.coalesce(pl.col("published_time"), pl.col("first_seen_time"))
                           .dt.epoch("s").cast(pl.Float32).alias("t")))
    art_t = article_vector(stats, "idx", "t", n_art, fill=np.nan)
    imp_t = pairs["time"].dt.epoch("s").to_numpy().astype(np.float64)
    # newer is better; an article with no known date sorts last, never first
    age = imp_t - np.nan_to_num(art_t[art], nan=-1e12)
    timed("recency", lambda: -age)

    # --- content rankers, both fed the identical history
    uids, hists = user_histories(fs, split, mode=history_mode)
    urow = np.full(int(max(pairs["user_idx"].max(), uids.max())) + 1, -1, dtype=np.int64)
    urow[uids] = np.arange(len(uids))
    pu = urow[pairs["user_idx"].to_numpy()]
    known = pu >= 0

    idx = BM25Index.build(fs.texts(), lang=fs.lang()).reweight(k1=k1, b=b)
    Q = idx.queries_from_history(hists, n_recent=n_recent)

    def bm25():
        s = np.zeros(len(art), dtype=np.float32)
        s[known] = idx.score_pairs(Q, pu[known], art[known])
        return s
    timed("bm25", bm25)

    emb = np.asarray(article_embeddings(fs, emb_name))
    uv = user_vectors(hists, emb, n_recent=n_recent)
    embn = emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12)

    def semantic():
        s = np.zeros(len(art), dtype=np.float32)
        s[known] = np.einsum("ij,ij->i", uv[pu[known]], embn[art[known]]).astype(np.float32)
        return s
    timed("emb", semantic)

    # --- fusion: rank-based, so the two score scales never have to be reconciled
    def rrf(c: int = 60):
        d = pairs.select("imp").with_columns(
            pl.Series("_b", rankers["bm25"]), pl.Series("_e", rankers["emb"]))
        d = d.with_columns([
            pl.col("_b").rank("ordinal", descending=True).over("imp").alias("rb"),
            pl.col("_e").rank("ordinal", descending=True).over("imp").alias("re"),
        ])
        return (1.0 / (c + d["rb"].to_numpy()) + 1.0 / (c + d["re"].to_numpy()))
    timed("hybrid_rrf", rrf)

    # --- Q9: illegal at serving time. Popularity counted from inside the split.
    def oracle():
        cl = (fs.impressions(split).select("clicked").explode("clicked").drop_nulls()
              .group_by("clicked").agg(pl.len().alias("c"))
              .rename({"clicked": "article_idx"}))
        return article_vector(cl, "article_idx", "c", n_art)[art]
    timed("pop_oracle*", oracle)

    return rankers, timings, embn, pop


# -------------------------------------------------------------------- slicing

def slice_labels(fs: FeatureStore, split: str, pairs: pl.DataFrame,
                 pop: np.ndarray, cold_threshold: int) -> pl.DataFrame:
    """Per-impression cold/warm and head/tail labels."""
    hist = fs.history(split).select("user_idx", "n_hist").collect()
    live = pop[pop > 0]
    head_cut = float(np.quantile(live, 0.80)) if live.size else np.inf
    clicked_head = (
        pairs.filter(pl.col("label"))
        .with_columns(pl.Series("_p", pop[pairs.filter(pl.col("label"))["article_idx"].to_numpy()]))
        .group_by("imp").agg((pl.col("_p").max() >= head_cut).alias("is_head"))
    )
    return (
        pairs.group_by("imp").agg(pl.col("user_idx").first())
        .join(hist.lazy().collect(), on="user_idx", how="left")
        .with_columns(pl.col("n_hist").fill_null(0))
        .with_columns((pl.col("n_hist") < cold_threshold).alias("is_cold"))
        .join(clicked_head, on="imp", how="left")
        .with_columns(pl.col("is_head").fill_null(False))
        .select("imp", "is_cold", "is_head")
    )


def main(dataset: str, variant: str, split: str, emb_name: str, k1: float | None, b: float | None,
         history_mode: str, out_dir: Path, n_boot: int, max_impressions: int = 0,
         n_recent: int = N_RECENT_DEFAULT):
    k1, b = tuned_bm25(dataset, variant, k1, b)
    fs = FeatureStore(dataset, variant)
    t0 = time.perf_counter()
    pairs = build_pairs(fs, split, max_impressions=max_impressions)
    n_imp = pairs["imp"].n_unique()
    print(f"== Q4 {dataset}/{variant} {split}: {n_imp:,} labelled impressions, "
          f"{pairs.height:,} candidate rows ({pairs.height/n_imp:.1f} per impression), "
          f"{pairs['label'].sum():,} clicks  [{time.perf_counter()-t0:.1f}s]")

    rankers, timings, embn, pop = make_rankers(fs, split, pairs, emb_name, k1, b,
                                              history_mode, n_recent)
    cold_thr = 5 if dataset == "mind" else 10
    sl = slice_labels(fs, split, pairs, pop, cold_thr)
    catalogue = int(pairs["article_idx"].n_unique())
    pop_share = (pop + 1.0) / (pop.sum() + len(pop))

    res = {"dataset": dataset, "variant": variant, "split": split, "history_mode": history_mode,
           "n_impressions": int(n_imp), "n_pairs": int(pairs.height), "n_recent": n_recent,
           "n_clicks": int(pairs["label"].sum()), "catalogue": catalogue,
           "bm25": {"k1": k1, "b": b}, "embedding": emb_name,
           "cold_threshold": cold_thr, "n_boot": n_boot,
           "max_impressions": max_impressions, "rankers": {}}

    all_metrics = tuple(METRICS) + tuple(f"ndcg@{k}" for k in KS) + tuple(f"hit@{k}" for k in KS)
    res["cutoffs"] = list(KS)
    res["rank_cap"] = RANK_CAP
    hdr = f"  {'ranker':14s} " + "  ".join(f"{m:>7s}" for m in METRICS) + \
          "   ILD@10   nov@10   cov@10   cold/warm nDCG@10   head/tail nDCG@10"
    print(hdr)
    for name, score in rankers.items():
        p = pairs.select("imp", "article_idx", "label").with_columns(pl.Series("score", score))
        per = rank_metrics(p, ks=KS).join(sl, on="imp", how="left")
        s = summarise_ranking(per, n_boot=n_boot, metrics=all_metrics)
        s["seconds"] = timings[name]
        s["rank_profile"] = rank_profile(per["first_rel"].to_numpy(), cap=RANK_CAP)

        rec = topk_lists(p, TOPK_BA)["rec"].to_list()
        rec = [np.asarray(r, dtype=np.int64) for r in rec]
        s["ild@10"] = round(intra_list_diversity(rec, embn), 4)
        s["novelty@10"] = round(novelty(rec, pop_share), 4)
        s["coverage@10"] = round(coverage(rec, catalogue), 4)

        for col, a, bl in [("is_cold", "cold", "warm"), ("is_head", "head", "tail")]:
            for lab, sub in [(a, per.filter(pl.col(col))), (bl, per.filter(~pl.col(col)))]:
                s[lab] = {"n": sub.height,
                          **{m: (round(float(sub[m].mean()), 5) if sub.height else None)
                             for m in METRICS}}
        res["rankers"][name] = s
        print(f"  {name:14s} " + "  ".join(f"{s[m]:7.4f}" for m in METRICS) +
              f"   {s['ild@10']:6.4f}  {s['novelty@10']:7.3f}  {s['coverage@10']:6.4f}"
              f"     {s['cold']['ndcg@10'] or 0:.4f}/{s['warm']['ndcg@10'] or 0:.4f}"
              f"          {s['head']['ndcg@10'] or 0:.4f}/{s['tail']['ndcg@10'] or 0:.4f}")

    out_dir.mkdir(parents=True, exist_ok=True)
    f = out_dir / f"q4_{dataset}_{variant}_{split}_{history_mode}.json"
    f.write_text(json.dumps(res, indent=2))
    o = res["rankers"]["pop_oracle*"]["auc"]; l = res["rankers"]["pop_prior"]["auc"]
    print(f"\n  Q9: popularity AUC {l:.4f} serving-safe vs {o:.4f} with future clicks "
          f"({100*(o-l)/max(l,1e-9):+.1f}%)")
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
    ap.add_argument("--history-mode", default="shipped", choices=["shipped", "augmented"])
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--max-impressions", type=int, default=0,
                    help="subsample impressions (0 = all); needed at large scale")
    ap.add_argument("--n-recent", type=int, default=N_RECENT_DEFAULT,
                    help="clicks read per user (0 = the whole history, the setting scripts/q3_userrep.py picked)")
    ap.add_argument("--out", default="reports/q4")
    a = ap.parse_args()
    main(a.dataset, a.variant, a.split, a.embedding, a.k1, a.b, a.history_mode,
         Path(a.out), a.n_boot, a.max_impressions, a.n_recent)
