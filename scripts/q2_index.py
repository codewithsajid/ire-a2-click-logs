"""Q2 ablation: what goes into the index, and what the tokeniser throws away.

Three choices were made once, early, on judgement and never measured:

  * indexed field -- title + abstract. EB-NeRD also ships a body; MIND does not,
    so the shipped choice was "the field both datasets have". That is an argument
    for comparability, not for retrieval quality, and the two are separable: this
    measures what the body is worth on EB-NeRD even though it cannot be used
    symmetrically.
  * stopwords -- a hand-written list per language. BM25's IDF already discounts
    frequent terms, so the list may be buying nothing but a smaller postings table.
  * stemming -- Snowball, run once per distinct token. Danish is more inflected
    than English, so the two corpora need not agree.

Index size and build time are reported next to recall: a choice that costs recall
but halves the postings is a different trade from one that costs nothing.
"""
from __future__ import annotations

import argparse, json, time
from pathlib import Path

import numpy as np
import polars as pl

import newsrec.lexical as lexical
from newsrec.lexical import BM25Index
from newsrec.retrieval import (candidate_universe_for_split, evaluate_recall,
                               summarise, user_histories)
from newsrec.store import FeatureStore

KS = (50, 100, 200)


def field_text(fs: FeatureStore, field: str) -> list[str]:
    cols = {"title": ["title"], "title+abstract": ["title", "abstract"],
            "title+abstract+body": ["title", "abstract", "body"]}[field]
    have = set(fs.articles().collect_schema().names())
    cols = [c for c in cols if c in have]
    return (fs.articles()
            .select(pl.concat_str([pl.col(c).fill_null("") for c in cols], separator=" ")
                    .alias("t"))
            .collect()["t"].to_list())


def build(texts, lang, stopwords: bool, stemming: bool, k1: float, b: float):
    """Toggle the tokeniser's two filters by patching the module's knobs."""
    saved_stop = lexical.STOPWORDS
    saved_langs = lexical.STEM_LANGS
    if not stopwords:
        lexical.STOPWORDS = {k: set() for k in saved_stop}
    # stemming is language-conditional by default now, so the ablation has to be
    # able to force it both ways rather than only off
    lexical.STEM_LANGS = ({lang} if stemming else set())
    try:
        t0 = time.perf_counter()
        idx = BM25Index.build(texts, lang=lang).reweight(k1=k1, b=b)
        return idx, time.perf_counter() - t0
    finally:
        lexical.STOPWORDS = saved_stop
        lexical.STEM_LANGS = saved_langs


def main(dataset: str, variant: str, split: str, k1: float | None, b: float | None,
         out_dir: Path):
    import sys; sys.path.insert(0, str(Path(__file__).parent))
    from q4_eval import tuned_bm25
    k1, b = tuned_bm25(dataset, variant, k1, b)

    fs = FeatureStore(dataset, variant)
    uni = candidate_universe_for_split(fs, split, 7)
    uids, hists = user_histories(fs, split)
    cold = 5 if dataset == "mind" else 10
    have_body = fs.articles().select(pl.col("body").is_not_null().any()).collect().item() \
        if "body" in set(fs.articles().collect_schema().names()) else False

    fields = ["title", "title+abstract"] + (["title+abstract+body"] if have_body else [])
    cases = [(f, True, True) for f in fields]
    cases += [("title+abstract", False, True), ("title+abstract", True, False),
              ("title+abstract", False, False)]

    res = {"dataset": dataset, "variant": variant, "split": split, "k1": k1, "b": b,
           "has_body": bool(have_body), "n_users": len(uids), "runs": []}
    print(f"== Q2 index pipeline {dataset}/{variant} {split}  (body available: {have_body})")

    for field, stop, stem in cases:
        texts = field_text(fs, field)
        idx, build_s = build(texts, fs.lang(), stop, stem, k1, b)
        Q = idx.queries_from_history(hists)
        _, topk = idx.search_sparse(Q, top_k=max(KS), universe=uni)
        s = summarise(evaluate_recall(fs, split, uids, topk, KS, cold_threshold=cold), KS)
        row = {"field": field, "stopwords": stop, "stemming": stem,
               "vocab": idx.vocab_size, "postings": int(len(idx.tf)),
               "avgdl": round(idx.avgdl, 1), "build_seconds": round(build_s, 2),
               **{f"recall@{k}": round(s[f"recall@{k}"], 5) for k in KS},
               "cold_recall@100": round(s["cold"]["recall@100"], 5)}
        res["runs"].append(row)
        tag = f"{field:20s} stop={str(stop):5s} stem={str(stem):5s}"
        print(f"  {tag}  vocab {row['vocab']:>7,} postings {row['postings']:>9,} "
              f"avgdl {row['avgdl']:5.1f} | {row['recall@50']:.4f} {row['recall@100']:.4f} "
              f"{row['recall@200']:.4f} | build {row['build_seconds']:5.2f}s")

    # the shipped configuration for *this* corpus, whatever STEM_LANGS says
    ship_stem = fs.lang() in lexical.STEM_LANGS
    base = next(r for r in res["runs"]
                if r["field"] == "title+abstract" and r["stopwords"]
                and r["stemming"] == ship_stem)
    for r in res["runs"]:
        r["vs_shipped"] = round(r["recall@100"] / base["recall@100"] - 1, 4)
    best = max(res["runs"], key=lambda r: r["recall@100"])
    res["shipped_recall@100"] = base["recall@100"]
    res["best"] = {k: best[k] for k in ("field", "stopwords", "stemming", "recall@100", "vs_shipped")}
    print(f"\n  shipped (title+abstract, stopwords, stem={ship_stem}) {base[chr(39)+chr(39)] if False else base['recall@100']:.4f}")
    print(f"  best {best['field']} stop={best['stopwords']} stem={best['stemming']} "
          f"-> {best['recall@100']:.4f} ({100 * best['vs_shipped']:+.1f}%)")
    out_dir.mkdir(parents=True, exist_ok=True)
    f = out_dir / f"index_{dataset}_{variant}.json"
    f.write_text(json.dumps(res, indent=2))
    print(f"== wrote {f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--variant", default="small")
    ap.add_argument("--split", default="test")
    ap.add_argument("--k1", type=float, default=None)
    ap.add_argument("--b", type=float, default=None)
    ap.add_argument("--out", default="reports/q2")
    a = ap.parse_args()
    main(a.dataset, a.variant, a.split, a.k1, a.b, Path(a.out))
