"""Lexical vs semantic, with and without excluding already-read articles."""
import json, time
from pathlib import Path
import numpy as np, polars as pl
from newsrec.store import FeatureStore
import sys; sys.path.insert(0, str(Path(__file__).parent))
from q4_eval import tuned_bm25
from newsrec.lexical import BM25Index
from newsrec.semantic import ANNIndex, user_vectors
from newsrec.embeddings import article_embeddings
from newsrec.baselines import random_topk, popularity_ranking, recency_ranking, broadcast
from newsrec.retrieval import (user_histories, evaluate_recall, summarise, drop_seen,
                               candidate_universe_for_split, universe_ceiling, bootstrap_ci)
KS = (50, 100, 200); KMAX = 200
out = {}
for ds, var, embs in [("ebnerd", "small", ["contrastive", "xlm_roberta"]),
                      ("mind", "small", ["sentence-transformers/all-MiniLM-L6-v2"])]:
    fs = FeatureStore(ds, var); split = "test"
    uni = candidate_universe_for_split(fs, split, 7)
    ceil = universe_ceiling(fs, split, uni)["ceiling"]
    uids, hists = user_histories(fs, split); n = len(uids)
    maxh = int(np.percentile([len(h) for h in hists], 99))
    over = KMAX + maxh
    print(f"\n=== {ds}/{var}  universe {len(uni):,}  ceiling {ceil:.4f}  users {n:,}")
    res = {}

    def add(name, topk_raw, hists_for_drop=None):
        rows = {}
        for tag, tk in [("raw", topk_raw[:, :KMAX]),
                        ("no-seen", drop_seen(topk_raw, hists_for_drop, KMAX) if hists_for_drop is not None else None)]:
            if tk is None: continue
            s = summarise(evaluate_recall(fs, split, uids, tk, KS), KS)
            rows[tag] = {k: s[f"recall@{k}"] for k in KS}
        res[name] = rows
        line = f"  {name:22s}"
        for tag in ("raw", "no-seen"):
            if tag in rows:
                line += f"  {tag:8s}" + " ".join(f"{rows[tag][k]:.4f}" for k in KS)
        print(line)

    add("random", random_topk(n, uni, over, user_ids=uids), hists)
    add("popularity", broadcast(popularity_ranking(fs, split, uni), n, over), hists)
    add("recency", broadcast(recency_ranking(fs, split, uni), n, over), hists)
    k1, b = tuned_bm25(ds, var, None, None)   # whatever Q2 selected for this corpus
    idx = BM25Index.build(fs.texts(), lang=fs.lang()).reweight(k1=k1, b=b)
    Q = idx.queries_from_history(hists)
    _, bm = idx.search_sparse(Q, top_k=over, universe=uni)
    add("bm25", bm, hists)
    for e in embs:
        emb = np.asarray(article_embeddings(fs, e))
        ann = ANNIndex(emb[uni], ids=uni, kind="flat")
        _, em = ann.search(user_vectors(hists, emb), top_k=over)
        add(f"emb:{e.split('/')[-1]}", em, hists)
    out[f"{ds}/{var}"] = {"universe": int(len(uni)), "ceiling": ceil, "methods": res}
Path("reports/q3/compare_dropseen.json").write_text(json.dumps(out, indent=2))
print("\n== wrote reports/q3/compare_dropseen.json")
