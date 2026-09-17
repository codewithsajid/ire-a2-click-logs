"""Qualitative lexical-vs-semantic comparison: what each retriever actually returns.

Aggregate recall says which is better; this says *why*. For a handful of users we
print the click history that formed the query, then the top BM25 and top embedding
results side by side, marking the articles the user really clicked.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl

from newsrec.embeddings import article_embeddings
from newsrec.lexical import BM25Index
from newsrec.retrieval import candidate_universe_for_split, user_histories
from newsrec.semantic import ANNIndex, user_vectors
from newsrec.store import FeatureStore
import sys; sys.path.insert(0, str(Path(__file__).parent))
from q4_eval import tuned_bm25


def short(s: str | None, n: int = 66) -> str:
    s = (s or "").replace("\n", " ").strip()
    return s if len(s) <= n else s[: n - 1] + "…"


def main(dataset: str, variant: str, emb_name: str, n_users: int, out: Path):
    fs = FeatureStore(dataset, variant)
    split = "test"
    uni = candidate_universe_for_split(fs, split, 7)
    uids, hists = user_histories(fs, split)
    arts = fs.articles().select("title", "category").collect()
    titles, cats = arts["title"], arts["category"]

    k1, b = tuned_bm25(dataset, variant, None, None)
    idx = BM25Index.build(fs.texts(), lang=fs.lang()).reweight(k1=k1, b=b)
    Q = idx.queries_from_history(hists)
    _, bm_top = idx.search_sparse(Q, top_k=10, universe=uni)

    emb = np.asarray(article_embeddings(fs, emb_name))
    ann = ANNIndex(emb[uni], ids=uni, kind="flat")
    _, em_top = ann.search(user_vectors(hists, emb), top_k=10)

    clicks = (
        fs.impressions(split).select("user_idx", "clicked")
        .filter(pl.col("clicked").list.len() > 0)
        .group_by("user_idx").agg(pl.col("clicked").flatten().unique().alias("clicked"))
        .collect()
    )
    click_map = {int(u): set(c) for u, c in clicks.iter_rows()}
    pos = {int(u): i for i, u in enumerate(uids)}

    # pick users where the two retrievers disagree most, that's where it's informative
    scored = []
    for u, cl in click_map.items():
        i = pos.get(u)
        if i is None or len(hists[i]) < 5 or not cl:
            continue
        b, e = set(bm_top[i, :10].tolist()), set(em_top[i, :10].tolist())
        scored.append((len(e & cl) - len(b & cl), len(b & e), u, i))
    scored.sort(key=lambda t: (-abs(t[0]), t[1]))

    lines = [f"# Lexical vs semantic — worked examples ({dataset}/{variant}, {split})", "",
             f"Query = terms/vectors of each user's last 30 clicked articles. "
             f"Universe = {len(uni):,} articles live in the 7 days around the split. "
             f"`*` marks an article the user actually clicked.", ""]
    for _, _, u, i in scored[:n_users]:
        cl = click_map[u]
        lines += [f"## user {u} — {len(hists[i])} clicks in history", "", "**History (most recent 6):**"]
        for d in list(hists[i])[-6:]:
            lines.append(f"- [{cats[int(d)]}] {short(titles[int(d)])}")
        lines += ["", "| # | BM25 | emb |", "|---|------|-----|"]
        for r in range(5):
            bd, ed = int(bm_top[i, r]), int(em_top[i, r])
            bm = ("*" if bd in cl else "") + f"[{cats[bd]}] {short(titles[bd], 48)}"
            em = ("*" if ed in cl else "") + f"[{cats[ed]}] {short(titles[ed], 48)}"
            lines.append(f"| {r+1} | {bm} | {em} |")
        overlap = len(set(bm_top[i, :10].tolist()) & set(em_top[i, :10].tolist()))
        lines += ["", f"top-10 overlap between the two retrievers: **{overlap}/10**", ""]

    # corpus-level agreement
    ov = np.mean([len(set(bm_top[i].tolist()) & set(em_top[i].tolist())) for i in range(len(uids))])
    lines += ["## Overall", "",
              f"- mean top-10 overlap between BM25 and embeddings: **{ov:.2f}/10** — "
              f"the two retrieve largely different articles, so their errors are not the same errors",
              f"- BM25 query terms per user (median): "
              f"{int(np.median(np.diff(Q.indptr)))}"]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines))
    print("\n".join(lines[:60]))
    print(f"\n== wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="ebnerd")
    ap.add_argument("--variant", default="small")
    ap.add_argument("--embedding", default="contrastive")
    ap.add_argument("--n-users", type=int, default=3)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    main(a.dataset, a.variant, a.embedding, a.n_users,
         Path(a.out or f"reports/q3/examples_{a.dataset}_{a.variant}.md"))
