"""Q3 ablation: what does approximate nearest-neighbour search actually cost?

Two questions, kept separate because they have different answers:

A. **Operating point.** At the scale this assignment actually retrieves over (a
   7-day candidate universe of 1.7K-23K articles), is an ANN index worth it at
   all? Measured against the exact index on the same vectors, so any recall drop
   is attributable to approximation and nothing else. Reports both *ANN recall*
   (overlap with the exact top-K -- the index's own fidelity) and *task recall*
   (the assignment metric), because a fidelity loss only matters if it moves the
   thing we are graded on.

B. **Scale sweep.** Index the full 125K-article EB-NeRD corpus at increasing N to
   find the crossover where the graph index starts beating brute force, which is
   the only honest way to justify -- or reject -- HNSW here.

Everything is timed single-threaded by default: this box has 48 cores, and
letting BLAS use all of them makes brute force look like an ANN index.
"""
from __future__ import annotations

import argparse, json, time
from pathlib import Path

import numpy as np

from newsrec.embeddings import article_embeddings
from newsrec.retrieval import (bootstrap_ci, candidate_universe_for_split,
                               evaluate_recall, summarise,
                               user_histories)
from newsrec.semantic import l2_normalise, user_vectors
from newsrec.store import FeatureStore

KS = (50, 100, 200)
TOPK = max(KS)


# ---------------------------------------------------------------- thread control

def limit_threads(n: int):
    """Pin BLAS *and* FAISS to n threads so the comparison is algorithmic."""
    import faiss
    faiss.omp_set_num_threads(n)
    try:
        from threadpoolctl import threadpool_limits
        return threadpool_limits(limits=n)
    except ImportError:
        return None


# ---------------------------------------------------------------- exact baseline

def numpy_topk(emb: np.ndarray, q: np.ndarray, k: int, chunk: int = 4096):
    """Brute force by chunked matmul -- the 'no index at all' baseline."""
    n = emb.shape[0]
    k = min(k, n)
    best_s = np.full((q.shape[0], 0), -np.inf, np.float32)
    best_i = np.zeros((q.shape[0], 0), np.int64)
    for s in range(0, n, chunk):
        blk = emb[s:s + chunk]
        sims = q @ blk.T
        kk = min(k, blk.shape[0])
        part = np.argpartition(-sims, kk - 1, axis=1)[:, :kk]
        best_s = np.hstack([best_s, np.take_along_axis(sims, part, 1)])
        best_i = np.hstack([best_i, part + s])
    order = np.argsort(-best_s, axis=1)[:, :k]
    return np.take_along_axis(best_s, order, 1), np.take_along_axis(best_i, order, 1)


# ---------------------------------------------------------------- index builders

def build_index(spec: dict, emb: np.ndarray):
    """Return (index, build_seconds, bytes). `spec` names one ablation point."""
    import faiss
    d = emb.shape[1]
    kind, metric = spec["kind"], faiss.METRIC_INNER_PRODUCT
    t0 = time.perf_counter()
    if kind == "flat":
        ix = faiss.IndexFlatIP(d)
    elif kind == "sq8":
        ix = faiss.IndexScalarQuantizer(d, faiss.ScalarQuantizer.QT_8bit, metric)
    elif kind == "hnsw":
        if spec.get("metric", "ip") == "l2":
            ix = faiss.IndexHNSWFlat(d, spec["M"])          # equivalent ranking on
        else:                                              # unit vectors
            ix = faiss.IndexHNSWFlat(d, spec["M"], metric)
        ix.hnsw.efConstruction = spec["efC"]
    elif kind == "ivf":
        ix = faiss.IndexIVFFlat(faiss.IndexFlatIP(d), d, spec["nlist"], metric)
    elif kind == "ivfpq":
        m = spec.get("m", 8)
        while d % m:
            m -= 1
        ix = faiss.IndexIVFPQ(faiss.IndexFlatIP(d), d, spec["nlist"], m, spec.get("nbits", 8))
    else:
        raise ValueError(kind)
    if not ix.is_trained:
        ix.train(emb)
    ix.add(emb)
    build_s = time.perf_counter() - t0
    return ix, build_s, len(faiss.serialize_index(ix))


def set_search_params(ix, spec: dict):
    import faiss
    if spec["kind"] == "hnsw":
        ix.hnsw.efSearch = spec["efS"]
    elif spec["kind"] in ("ivf", "ivfpq"):
        faiss.extract_index_ivf(ix).nprobe = spec["nprobe"]


def label(spec: dict) -> str:
    k = spec["kind"]
    if k == "hnsw":
        m = "" if spec.get("metric", "ip") == "ip" else "/L2"
        return f"hnsw M={spec['M']} efC={spec['efC']} efS={spec['efS']}{m}"
    if k == "ivf":
        return f"ivf nlist={spec['nlist']} nprobe={spec['nprobe']}"
    if k == "ivfpq":
        return f"ivfpq nlist={spec['nlist']} nprobe={spec['nprobe']} m={spec.get('m',8)}"
    return {"flat": "faiss flat (exact)", "sq8": "faiss SQ8 (exact scan, 8-bit)"}[k]


# ---------------------------------------------------------------- measurement

def ann_recall(approx: np.ndarray, gt_scores: np.ndarray, emb: np.ndarray, q: np.ndarray,
               k: int, eps: float = 1e-5, chunk: int = 512) -> float:
    """Fraction of a query's top-k that is *as good as* the exact top-k.

    Plain set overlap against the exact ids is wrong here: MIND carries many
    duplicate articles, whose embeddings are identical, so two exact indices
    disagree on which of a tied group to return and an exact index scores below
    1.0 for no reason. Scoring by value instead -- an id counts if its true
    similarity reaches the k-th exact score -- makes the exact index exactly 1.0
    by construction and leaves genuine approximation error intact.
    """
    thr = gt_scores[:, min(k, gt_scores.shape[1]) - 1] - eps
    hits = 0
    for s in range(0, len(q), chunk):
        ids = approx[s:s + chunk, :k]
        valid = ids >= 0
        sc = np.einsum("cd,ckd->ck", q[s:s + chunk], emb[np.where(valid, ids, 0)])
        hits += int(((sc >= thr[s:s + chunk, None]) & valid).sum())
    return float(hits / (len(q) * k))


def n_repeats(n_q: int, n_db: int) -> int:
    """Enough repeats to beat timer noise, few enough to finish: the cost of one
    exact pass is ~n_q*n_db, so budget on that product."""
    return int(max(1, min(3, 2e8 // max(n_q * n_db, 1))))


def timed_search(ix, q: np.ndarray, k: int, repeats: int = 3):
    ix.search(q[:64], k)                                   # warm caches
    best = np.inf
    for _ in range(repeats):
        t0 = time.perf_counter()
        s, i = ix.search(q, k)
        best = min(best, time.perf_counter() - t0)
    return s, i, best


def timed_numpy(emb: np.ndarray, q: np.ndarray, k: int, repeats: int = 3):
    """Same protocol as timed_search -- one warm pass, then best-of-repeats --
    so brute force is not penalised by a cold cache the FAISS rows never pay."""
    gs, gt = numpy_topk(emb, q, k)
    best = np.inf
    for _ in range(repeats):
        t0 = time.perf_counter()
        numpy_topk(emb, q, k)
        best = min(best, time.perf_counter() - t0)
    return gt, gs, best


# ---------------------------------------------------------------- part A

def part_a(fs: FeatureStore, split: str, emb: np.ndarray, threads: int, out: Path,
           tag: str, specs: list[dict], max_queries: int = 0, seed: int = 0):
    uni = candidate_universe_for_split(fs, split, 7)
    uids, hists = user_histories(fs, split)
    sub = np.ascontiguousarray(l2_normalise(emb[uni].astype(np.float32)))
    q = np.ascontiguousarray(user_vectors(hists, emb))
    n_all = len(q)
    if max_queries and n_all > max_queries:
        # a fixed random subsample: 10K users pins recall to ~+-0.4% and keeps the
        # single-threaded sweep to an hour instead of most of a day
        keep = np.sort(np.random.default_rng(seed).choice(n_all, max_queries, replace=False))
        q, uids, hists = q[keep], uids[keep], [hists[i] for i in keep]
    reps = n_repeats(len(q), len(uni))
    print(f"\n== A. operating point: {tag} {split} | universe {len(uni):,} x {sub.shape[1]}d "
          f"| {len(q):,} of {n_all:,} queries | {threads} thread(s) | {reps} timing rep(s)")

    ctx = limit_threads(threads)
    try:
        gt, gs, np_s = timed_numpy(sub, q, TOPK, repeats=reps)
        rows = [{"index": "numpy brute force", "build_s": 0.0, "bytes": int(sub.nbytes),
                 "query_s": round(np_s, 4), "qps": round(len(q) / np_s, 1),
                 "ann_recall@100": 1.0, "exact": True}]
        cold_thr = 5 if fs.dataset == "mind" else 10
        sampled = bool(max_queries and n_all > max_queries)

        def task_recall(topk_local, with_ci: bool = False):
            ids = uni[topk_local]
            df = evaluate_recall(fs, split, uids, ids, KS, cold_threshold=cold_thr,
                                 only_users=sampled)
            s = summarise(df, KS)
            out = {f"recall@{k}": round(s[f"recall@{k}"], 5) for k in KS}
            if with_ci:
                lo, hi = bootstrap_ci(df["recall@100"].to_numpy(), n_boot=500)[1:]
                out["recall@100_ci95"] = [round(lo, 5), round(hi, 5)]
            return out

        # An approximate index cannot beat exact search on the objective it
        # approximates, so any positive "task recall" delta is sampling noise.
        # The exact row carries a CI so the table states the size of that noise
        # instead of leaving every +2% looking like a result.
        rows[0].update(task_recall(gt, with_ci=True))
        ci = rows[0]["recall@100_ci95"]
        half = (ci[1] - ci[0]) / 2
        print(f"  exact task recall@100 = {rows[0]['recall@100']:.4f} "
              f"95% CI [{ci[0]:.4f}, {ci[1]:.4f}] -- differences under "
              f"{100 * half / max(rows[0]['recall@100'], 1e-9):.1f}% are noise")
        for spec in specs:
            ix, bs, nbytes = build_index(spec, sub)
            set_search_params(ix, spec)
            _, idx, qs = timed_search(ix, q, TOPK, repeats=reps)
            r = {"index": label(spec), "build_s": round(bs, 3), "bytes": int(nbytes),
                 "query_s": round(qs, 4), "qps": round(len(q) / qs, 1),
                 "ann_recall@100": round(ann_recall(idx, gs, sub, q, 100), 4),
                 "exact": spec["kind"] == "flat"}
            r.update(task_recall(idx))
            rows.append(r)
            print(f"  {r['index']:38s} build {r['build_s']:7.2f}s  "
                  f"{r['bytes']/2**20:7.1f}MB  {r['qps']:9.1f} q/s  "
                  f"ann_r@100 {r['ann_recall@100']:.4f}  task_r@100 {r['recall@100']:.4f}")
    finally:
        if ctx is not None:
            ctx.__exit__(None, None, None)
    (out / f"ann_operating_{tag}.json").write_text(json.dumps(
        {"tag": tag, "split": split, "universe": int(len(uni)), "dim": int(sub.shape[1]),
         "n_queries": int(len(q)), "n_users_total": int(n_all), "threads": threads,
         "repeats": reps, "rows": rows}, indent=2))
    return rows


# ---------------------------------------------------------------- part B

def part_b(emb: np.ndarray, sizes: list[int], n_queries: int, threads: int, out: Path,
           tag: str, specs: list[dict], seed: int = 0):
    rng = np.random.default_rng(seed)
    full = l2_normalise(emb.astype(np.float32))
    qi = rng.choice(full.shape[0], size=min(n_queries, full.shape[0]), replace=False)
    q = np.ascontiguousarray(full[qi] + rng.normal(0, 0.05, (len(qi), full.shape[1])).astype(np.float32))
    q = np.ascontiguousarray(l2_normalise(q))              # near-corpus, not identical
    print(f"\n== B. scale sweep: {tag} | dim {full.shape[1]} | {len(q):,} queries "
          f"| {threads} thread(s)")
    rows = []
    ctx = limit_threads(threads)
    try:
        for n in sizes:
            if n > full.shape[0]:
                continue
            sub = np.ascontiguousarray(full[rng.choice(full.shape[0], n, replace=False)])
            reps = n_repeats(len(q), n)
            gt, gs, np_s = timed_numpy(sub, q, TOPK, repeats=reps)
            rows.append({"n": n, "index": "numpy brute force", "build_s": 0.0,
                         "bytes": int(sub.nbytes), "query_s": round(np_s, 4),
                         "qps": round(len(q) / np_s, 1), "ann_recall@100": 1.0})
            print(f"  N={n:>7,}  {'numpy brute force':30s} {len(q)/np_s:9.1f} q/s")
            for spec in specs:
                s = dict(spec)
                if s["kind"] in ("ivf", "ivfpq"):
                    s["nlist"] = max(1, min(s["nlist"], n // 40))
                ix, bs, nbytes = build_index(s, sub)
                set_search_params(ix, s)
                _, idx, qs = timed_search(ix, q, TOPK, repeats=reps)
                rows.append({"n": n, "index": label(s), "build_s": round(bs, 3),
                             "bytes": int(nbytes), "query_s": round(qs, 4),
                             "qps": round(len(q) / qs, 1),
                             "ann_recall@100": round(ann_recall(idx, gs, sub, q, 100), 4)})
                print(f"  N={n:>7,}  {label(s):30s} {len(q)/qs:9.1f} q/s  "
                      f"build {bs:6.2f}s  {nbytes/2**20:6.1f}MB  r@100 {rows[-1]['ann_recall@100']:.4f}")
    finally:
        if ctx is not None:
            ctx.__exit__(None, None, None)
    (out / f"ann_scale_{tag}.json").write_text(json.dumps(
        {"tag": tag, "dim": int(full.shape[1]), "n_queries": int(len(q)),
         "threads": threads, "rows": rows}, indent=2))
    return rows


# ---------------------------------------------------------------- specs

def operating_specs() -> list[dict]:
    s = [{"kind": "flat"}, {"kind": "sq8"}]
    for M in (8, 16, 32, 64):
        for efS in (16, 64, 256):
            s.append({"kind": "hnsw", "M": M, "efC": 200, "efS": efS})
    for efC in (40, 500):                                  # efConstruction at fixed M
        for efS in (64, 256):
            s.append({"kind": "hnsw", "M": 32, "efC": efC, "efS": efS})
    s.append({"kind": "hnsw", "M": 32, "efC": 200, "efS": 256, "metric": "l2"})
    for nlist in (64, 256, 1024):
        for nprobe in (1, 8, 32):
            s.append({"kind": "ivf", "nlist": nlist, "nprobe": nprobe})
    s.append({"kind": "ivfpq", "nlist": 256, "nprobe": 32, "m": 8})
    return s


def scale_specs() -> list[dict]:
    return [{"kind": "flat"},
            {"kind": "hnsw", "M": 16, "efC": 200, "efS": 64},
            {"kind": "hnsw", "M": 32, "efC": 200, "efS": 64},
            {"kind": "hnsw", "M": 32, "efC": 200, "efS": 256},
            {"kind": "ivf", "nlist": 1024, "nprobe": 8}]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="ebnerd")
    ap.add_argument("--variant", default="small")
    ap.add_argument("--split", default="test")
    ap.add_argument("--embedding", default="contrastive")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--part", default="ab", choices=["a", "b", "ab"])
    ap.add_argument("--scale-variant", default="large")
    ap.add_argument("--sizes", default="2000,8000,32000,125000")
    ap.add_argument("--scale-queries", type=int, default=2000)
    ap.add_argument("--max-queries", type=int, default=0,
                    help="subsample user queries in part A (0 = use all)")
    ap.add_argument("--out", default="reports/q3")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    tag = f"{a.dataset}_{a.variant}_{a.embedding}"

    if "a" in a.part:
        fs = FeatureStore(a.dataset, a.variant)
        emb = np.asarray(article_embeddings(fs, a.embedding))
        part_a(fs, a.split, emb, a.threads, out, tag, operating_specs(),
               max_queries=a.max_queries)
    if "b" in a.part:
        fs2 = FeatureStore(a.dataset, a.scale_variant)
        emb2 = np.asarray(article_embeddings(fs2, a.embedding))
        part_b(emb2, [int(x) for x in a.sizes.split(",")], a.scale_queries, a.threads,
               out, f"{a.dataset}_{a.scale_variant}_{a.embedding}", scale_specs())
