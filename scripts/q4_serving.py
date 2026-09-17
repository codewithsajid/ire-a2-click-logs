"""Q4: serving and scale -- index memory, p99 latency, cost per 1000 queries, 10x.

A1 measured the *retrieval* service: D_total 9.19 ms with `ann_search` the
bottleneck at 8.96 ms, service CV 0.01 (so M/D/1, not M/M/1 -- M/M/1
over-predicts wait 1.9x), and 110 -> 467 q/s going from 1 to 48 leaf threads.
A2 adds the second stage, so the question is what re-ranking costs on top and
which stage the budget actually goes to.

The cascade's own argument (L6 s.21) is that the funnel exists because *feature*
cost, not model cost, is what forbids running the heavy ranker on everything.
That is a testable claim about this system: if it holds, the re-rank stage is
dominated by gathering features for K candidates rather than by evaluating the
trees over them.

**Measurement discipline.** A1's harness caught eight measurement bugs before
they became results -- a cold read served warm by a live memmap, a timer that
included its own setup. The same rules apply here:

  * every stage is timed separately *and* end to end, so the parts are checked
    against the whole rather than assumed to sum;
  * the first `warmup` requests are discarded (BLAS threadpools, page cache and
    lazy CUDA contexts all initialise on first use);
  * latency is reported as a distribution, never a mean -- the SLA is p99 and a
    mean hides exactly the tail it is about;
  * memory is measured as both the serialised size and the RSS delta, because
    the two differ and only one of them is what a box must hold.

Writes reports/q4/serving_<dataset>_<variant>.json.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path

import numpy as np
import polars as pl

from newsrec.config import DATA_ROOT
from newsrec.embeddings import article_embeddings
from newsrec.lexical import BM25Index
from newsrec.rerank import SHIPPED, feature_names
from newsrec.retrieval import candidate_universe_for_split, user_histories
from newsrec.semantic import ANNIndex, l2_normalise, user_vectors
from newsrec.store import FeatureStore


def rss_mb() -> float:
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * os.sysconf("SC_PAGESIZE") / 2 ** 20


def quantiles(x: list[float]) -> dict:
    a = np.asarray(x, dtype=np.float64) * 1000.0     # ms
    return {"n": len(a), "mean_ms": float(a.mean()),
            "p50_ms": float(np.percentile(a, 50)),
            "p95_ms": float(np.percentile(a, 95)),
            "p99_ms": float(np.percentile(a, 99)),
            "max_ms": float(a.max())}


def measure_index_memory(fs, emb_name, universe, texts, lang, k1, b) -> dict:
    """Q4.1 -- serialised bytes and resident cost of each served structure."""
    import faiss

    out = {}
    gc.collect(); base = rss_mb()
    emb = l2_normalise(np.asarray(article_embeddings(fs, emb_name), dtype=np.float32))
    out["article_vectors"] = {
        "n": int(emb.shape[0]), "dim": int(emb.shape[1]),
        "bytes": int(emb.nbytes), "mb": round(emb.nbytes / 2 ** 20, 2)}

    gc.collect(); before = rss_mb()
    ann = ANNIndex(emb[universe], ids=universe, kind="flat")
    ser = faiss.serialize_index(ann.index)
    out["ann_index"] = {
        "kind": "IndexFlatIP", "n_vectors": int(len(universe)),
        "serialized_mb": round(len(ser) / 2 ** 20, 2),
        "rss_delta_mb": round(rss_mb() - before, 2)}

    gc.collect(); before = rss_mb()
    t0 = time.perf_counter()
    bm = BM25Index.build(texts, lang=lang, k1=k1, b=b)
    out["bm25_index"] = {
        "n_docs": bm.n_docs, "vocab": bm.vocab_size,
        "postings": int(bm._W.nnz),
        "csr_mb": round((bm._W.data.nbytes + bm._W.indices.nbytes
                         + bm._W.indptr.nbytes) / 2 ** 20, 2),
        "rss_delta_mb": round(rss_mb() - before, 2),
        "build_seconds": round(time.perf_counter() - t0, 2)}

    # feature store: what a server must hold to answer a request, not the whole
    # parquet tree -- the per-article tables are the served half
    store_root = DATA_ROOT / "processed" / fs.dataset / fs.variant
    disk = sum(p.stat().st_size for p in store_root.rglob("*.parquet"))
    gc.collect(); before = rss_mb()
    af = fs.article_features("test").collect()
    st = fs.article_stats().collect()
    out["feature_store"] = {
        "parquet_on_disk_mb": round(disk / 2 ** 20, 2),
        "article_tables_rows": af.height + st.height,
        "article_tables_rss_mb": round(rss_mb() - before, 2),
        "estimated_bytes_per_article": round(
            (rss_mb() - before) * 2 ** 20 / max(fs.n_articles, 1), 1)}
    out["total_resident_mb"] = round(rss_mb() - base, 2)
    return out, bm, ann, emb


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["ebnerd", "mind"])
    ap.add_argument("--variant", default="small")
    ap.add_argument("--embedding", default=None)
    ap.add_argument("--k", type=int, default=200)
    ap.add_argument("--requests", type=int, default=500)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--sla-ms", type=float, default=100.0)
    ap.add_argument("--instance-usd-hr", type=float, default=0.50,
                    help="price of the box the benchmark ran on; stated as an "
                         "assumption because the cost answer is linear in it")
    ap.add_argument("--out", type=Path, default=Path("reports/q4"))
    a = ap.parse_args()

    emb_name = a.embedding or ("contrastive" if a.dataset == "ebnerd"
                               else "sentence-transformers/all-MiniLM-L6-v2")
    fs = FeatureStore(a.dataset, a.variant)
    bmj = json.loads((Path("reports/q2") /
                      f"q2_bm25_{a.dataset}_{a.variant}.json").read_text())["best"]
    k1, b = float(bmj["k1"]), float(bmj["b"])
    universe = candidate_universe_for_split(fs, "test", 7)
    texts, lang = fs.texts(), fs.lang()

    print(f"[{a.dataset}/{a.variant}] {fs}")
    print(f"   universe {len(universe):,} live articles; K={a.k}")

    mem, bm, ann, emb = measure_index_memory(fs, emb_name, universe, texts, lang, k1, b)
    print(f"\n   Q4.1 index + feature-store memory")
    for k, v in mem.items():
        if isinstance(v, dict):
            print(f"     {k:<18} " + "  ".join(f"{kk}={vv}" for kk, vv in v.items()))
    print(f"     total resident: {mem['total_resident_mb']} MB")

    # ---- the serving path for one user
    user_ids, hists = user_histories(fs, "test")
    Q = bm.queries_from_history(hists)
    uv = user_vectors(hists, emb)

    model_path = Path("reports/q2") / f"model_{a.dataset}_{a.variant}_shipped.txt"
    import lightgbm as lgb
    model = lgb.Booster(model_file=str(model_path)) if model_path.exists() else None
    te = pl.read_parquet(Path("artifacts/features") / a.dataset / a.variant / "test.parquet")
    feats = feature_names(te, SHIPPED)
    feat_pool = te.select(feats).to_numpy().astype(np.float32)
    print(f"   re-ranker: {'loaded' if model else 'MISSING -- run q2_rerank first'}"
          f", {len(feats)} features")

    rng = np.random.default_rng(0)
    picks = rng.integers(0, len(user_ids), size=a.requests + a.warmup)
    stage = {n: [] for n in ("bm25", "ann", "fuse", "feature_fetch", "rerank", "end_to_end")}

    for i, u in enumerate(picks):
        t_all = time.perf_counter()

        t = time.perf_counter()
        _, bm_ids = bm.search_sparse(Q[u:u + 1], top_k=a.k, universe=universe)
        t_bm = time.perf_counter() - t

        t = time.perf_counter()
        _, ann_ids = ann.search(uv[u:u + 1], top_k=a.k)
        t_ann = time.perf_counter() - t

        t = time.perf_counter()
        cand = np.unique(np.concatenate([bm_ids[0], ann_ids[0]]))[:a.k]
        t_fuse = time.perf_counter() - t

        # feature fetch: gather K rows of the design matrix, which is what a
        # feature store round-trip costs once it is in RAM
        t = time.perf_counter()
        rows = rng.integers(0, feat_pool.shape[0], size=len(cand))
        X = feat_pool[rows]
        t_feat = time.perf_counter() - t

        t = time.perf_counter()
        if model is not None:
            model.predict(X)
        t_rr = time.perf_counter() - t

        total = time.perf_counter() - t_all
        if i >= a.warmup:
            stage["bm25"].append(t_bm); stage["ann"].append(t_ann)
            stage["fuse"].append(t_fuse); stage["feature_fetch"].append(t_feat)
            stage["rerank"].append(t_rr); stage["end_to_end"].append(total)

    lat = {k: quantiles(v) for k, v in stage.items()}
    print(f"\n   Q4.2 latency for one user request ({a.requests} requests, "
          f"{a.warmup} warm-up discarded)")
    print(f"     {'stage':<15} {'p50':>8} {'p95':>8} {'p99':>8} {'mean':>8}")
    for k, v in lat.items():
        print(f"     {k:<15} {v['p50_ms']:>8.3f} {v['p95_ms']:>8.3f} "
              f"{v['p99_ms']:>8.3f} {v['mean_ms']:>8.3f}")

    e2e = lat["end_to_end"]
    gen = lat["bm25"]["mean_ms"] + lat["ann"]["mean_ms"] + lat["fuse"]["mean_ms"]
    rr = lat["feature_fetch"]["mean_ms"] + lat["rerank"]["mean_ms"]
    share_fetch = (lat["feature_fetch"]["mean_ms"] / max(rr, 1e-9))
    print(f"\n     candidate generation {gen:.3f} ms  |  re-ranking {rr:.3f} ms")
    print(f"     within re-ranking, feature fetch is {share_fetch:.1%} "
          f"-- the cascade's claim is that this dominates model evaluation")

    # ---- Q4.3 cost per 1000 queries at the SLA
    qps_1 = 1000.0 / max(e2e["mean_ms"], 1e-9)
    cores = os.cpu_count() or 1
    meets = e2e["p99_ms"] < a.sla_ms
    qps_box = qps_1 * cores          # embarrassingly parallel across requests
    cost = {
        "sla_ms": a.sla_ms,
        "p99_ms": e2e["p99_ms"],
        "meets_sla": bool(meets),
        "qps_single_threaded": round(qps_1, 1),
        "cores": cores,
        "qps_per_box_ideal": round(qps_box, 1),
        "instance_usd_per_hour": a.instance_usd_hr,
        "usd_per_1000_queries": round(a.instance_usd_hr / max(qps_box, 1e-9) / 3.6, 6),
        "note": ("linear scaling across cores is an upper bound: A1 measured "
                 "110 -> 467 q/s going 1 -> 48 leaves, i.e. 4.23x for 48x the "
                 "threads, so the realistic figure is several times this"),
    }
    print(f"\n   Q4.3 cost at p99 < {a.sla_ms} ms")
    print(f"     p99 {e2e['p99_ms']:.2f} ms -> SLA {'met' if meets else 'MISSED'}")
    print(f"     {qps_1:.1f} q/s single-threaded, {cores} cores")
    print(f"     ${cost['usd_per_1000_queries']:.6f} per 1000 queries "
          f"at ${a.instance_usd_hr}/hr (ideal scaling)")

    out = {"dataset": a.dataset, "variant": a.variant, "embedding": emb_name,
           "k": a.k, "universe": int(len(universe)),
           "n_articles": fs.n_articles, "n_users": int(len(user_ids)),
           "memory": mem, "latency": lat,
           "stage_split": {"candidate_generation_ms": round(gen, 3),
                           "reranking_ms": round(rr, 3),
                           "feature_fetch_share_of_rerank": round(share_fetch, 4)},
           "cost": cost}
    a.out.mkdir(parents=True, exist_ok=True)
    f = a.out / f"serving_{a.dataset}_{a.variant}.json"
    f.write_text(json.dumps(out, indent=2))
    print(f"\n   wrote {f}")


if __name__ == "__main__":
    main()
