"""Q3 ablation: does a similarity threshold buy anything over plain top-K?

Everything in Q3 so far retrieves a fixed 200 candidates per user, however weak
the match. That is a defensible default and an unexamined one, and it is least
defensible on the small variants: EB-NeRD/small's live universe is 1,677
articles, so a top-200 list is 12% of the entire catalogue handed to every user,
while MIND/small's 22,771-article universe makes the same list 0.9%.

A threshold cannot raise recall -- it only removes candidates -- so the question
is not "is recall higher" but "how much of the candidate budget can a cutoff cut
before recall starts to go". Two forms are compared, because they answer
different questions:

  * a global cutoff, which assumes cosine is comparable across users;
  * a per-user relative cutoff (keep >= alpha * that user's best match), which
    does not.

The gap between them is a calibration measurement: if the relative cutoff
dominates, a single global threshold is meaningless no matter how it is tuned.
"""
from __future__ import annotations

import argparse, json
from pathlib import Path

import numpy as np
import polars as pl

from newsrec.embeddings import article_embeddings
from newsrec.retrieval import (candidate_universe_for_split, evaluate_recall, summarise,
                               user_histories)
from newsrec.semantic import ANNIndex, user_vectors
from newsrec.store import FeatureStore

KS = (50, 100, 200)
TOPK = max(KS)


def recall_of(fs, split, uids, ids, cold_thr):
    s = summarise(evaluate_recall(fs, split, uids, ids, KS, cold_threshold=cold_thr), KS)
    return {f"recall@{k}": round(s[f"recall@{k}"], 5) for k in KS} | {
        "cold_recall@100": round(s["cold"]["recall@100"], 5),
        "warm_recall@100": round(s["warm"]["recall@100"], 5)}


def main(dataset: str, variant: str, split: str, emb_name: str, out_dir: Path):
    fs = FeatureStore(dataset, variant)
    uni = candidate_universe_for_split(fs, split, 7)
    uids, hists = user_histories(fs, split)
    emb = np.asarray(article_embeddings(fs, emb_name))
    uv = user_vectors(hists, emb)

    ix = ANNIndex(emb[uni], ids=uni, kind="flat")
    scores, topk = ix.search(uv, top_k=TOPK)
    scores = np.asarray(scores, dtype=np.float32)
    # a sentinel that is a valid uint32 but can never be a real article index,
    # so a dropped slot simply fails every set-intersection
    SENT = np.uint32(fs.n_articles)
    cold_thr = 5 if dataset == "mind" else 10

    has_hist = np.array([len(h) > 0 for h in hists])
    print(f"== {dataset}/{variant} {split}: universe {len(uni):,} | {len(uids):,} users "
          f"({int((~has_hist).sum()):,} with no history) | top-{TOPK} is "
          f"{100*TOPK/len(uni):.1f}% of the universe")

    res = {"dataset": dataset, "variant": variant, "split": split, "embedding": emb_name,
           "universe": int(len(uni)), "n_users": int(len(uids)),
           "topk_share_of_universe": round(TOPK / len(uni), 4),
           "score_percentiles": {}, "global": [], "relative": []}

    # --- what do the similarities even look like?
    valid = scores[has_hist]
    for r in (1, 10, 50, 100, 200):
        col = valid[:, min(r, TOPK) - 1]
        res["score_percentiles"][f"rank{r}"] = {
            q: round(float(np.percentile(col, q)), 4) for q in (5, 25, 50, 75, 95)}
    p = res["score_percentiles"]
    print(f"  cosine at rank   1: median {p['rank1'][50]:.3f}  (p5 {p['rank1'][5]:.3f} "
          f"-> p95 {p['rank1'][95]:.3f})")
    print(f"  cosine at rank 200: median {p['rank200'][50]:.3f}  (p5 {p['rank200'][5]:.3f} "
          f"-> p95 {p['rank200'][95]:.3f})")
    spread = float(np.median(valid[:, 0] - valid[:, TOPK - 1]))
    print(f"  median within-user drop rank1->rank200: {spread:.3f}")

    base = recall_of(fs, split, uids, topk, cold_thr)
    res["baseline"] = base | {"mean_kept": float(TOPK)}

    # The control that decides the question. A threshold that keeps a mean of K'
    # candidates has to beat simply taking the top K' -- otherwise it is spending
    # the same budget worse, and "it saved 56% of the candidates" means nothing.
    fixed_cache: dict[int, float] = {}

    def fixed_topk_recall(kp: int) -> float:
        kp = max(1, min(TOPK, int(round(kp))))
        if kp not in fixed_cache:
            ids = np.where(np.arange(TOPK)[None, :] < kp, topk, SENT).astype(np.uint32)
            fixed_cache[kp] = recall_of(fs, split, uids, ids, cold_thr)["recall@200"]
        return fixed_cache[kp]

    print(f"\n  {'cutoff':>22s}  {'kept':>7s} {'empty':>6s}   "
          + "  ".join(f"r@{k:<4}" for k in KS) + "   retained  vs top-K'")

    def record(bucket, label, mask, extra):
        kept = mask.sum(1)
        ids = np.where(mask, topk, SENT).astype(np.uint32)
        # push kept candidates to the front so recall@50 sees the best 50 kept
        order = np.argsort(~mask, axis=1, kind="stable")
        ids = np.take_along_axis(ids, order, axis=1)
        r = recall_of(fs, split, uids, ids, cold_thr)
        mk = float(kept.mean())
        ctl = fixed_topk_recall(mk)
        row = {**extra, "mean_kept": round(mk, 2),
               "empty_users": round(float((kept == 0).mean()), 4),
               **r, "retained@200": round(r["recall@200"] / max(base["recall@200"], 1e-9), 4),
               "budget": round(mk / TOPK, 4),
               "fixed_topk_control": round(ctl, 5),
               "vs_control": round(r["recall@200"] - ctl, 5)}
        bucket.append(row)
        print(f"  {label:>22s}  {row['mean_kept']:7.1f} {row['empty_users']:6.1%}   "
              + "  ".join(f"{row[f'recall@{k}']:.4f}" for k in KS)
              + f"   {row['retained@200']:.4f}   {row['vs_control']:+.5f}")

    # the interesting band is narrow and dataset-specific, so the grid is anchored
    # on this run's own score distribution rather than a fixed ladder
    q = [float(np.percentile(valid, x)) for x in (1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99)]
    for tau in [0.0] + [round(x, 4) for x in q]:
        record(res["global"], f"global tau={tau:.2f}", scores >= tau, {"tau": tau})

    top1 = scores[:, :1]
    for alpha in (0.0, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99):
        record(res["relative"], f"relative a={alpha:.2f}", scores >= alpha * top1,
               {"alpha": alpha})

    out_dir.mkdir(parents=True, exist_ok=True)
    f = out_dir / f"threshold_{dataset}_{variant}.json"
    f.write_text(json.dumps(res, indent=2))
    print(f"== wrote {f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--embedding", default="contrastive")
    ap.add_argument("--out", default="reports/q3")
    a = ap.parse_args()
    main(a.dataset, a.variant, a.split, a.embedding, Path(a.out))
