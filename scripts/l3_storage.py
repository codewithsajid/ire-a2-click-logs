"""L3 ablation: storage, access patterns, and the RUM triangle on our own index.

L3's argument is that where data lives and how it is touched dominates everything
else — sequential beats random by two orders of magnitude, logs beat in-place
updates when writes dominate, and every access method is a point on the
read/update/memory triangle where lowering two raises the third. This repo has
been quietly assuming all of it: every vector lives in RAM, every index is built
once and never updated, and the RUM trade in the ANN ablation is measured but
never named.

Four parts, each testing a claim the slides make rather than illustrating it:

A. **The hierarchy, on our own artefacts.** Cold NVMe vs warm page cache vs mmap,
   for the 368 MB embedding matrix, and sequential slab reads against the random
   row gathers a user vector actually performs. The slide quotes ~100× for random
   4 KB on NVMe; a row here is 3,072 bytes, i.e. almost exactly one page, so this
   is that constant measured on the exact access pattern the service uses.

B. **Precision as a bandwidth trade.** The L2 harness found `ann_search` is
   memory-bandwidth bound, not compute bound. That is a falsifiable prediction:
   halve the bytes per vector and throughput should nearly double. fp32 → fp16 →
   int8 → int4 halves it four times, and FAISS's scalar quantiser will do it
   exactly, so the prediction can be checked and the recall cost priced beside it.

C. **The update axis — the one RUM corner nothing here has measured.** News turns
   over daily and every index in this repo is rebuilt wholesale. L3 says the escape
   is LSM-shaped: write sorted runs and merge them, which is what Lucene does with
   segments. So absorb a week of new articles three ways — full rebuild, in-place
   append, and base-plus-segments with periodic merge — and measure what each costs
   in update time, query throughput and bytes.

D. **Compression as compute traded for bandwidth.** The store is parquet+zstd and
   the vectors are raw float32, neither chosen by measurement. Codec sweep on both.

Every index here is exact, so nothing in parts A, C or D can move recall; part B is
the one place precision is traded and it reports the fidelity it costs.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import gc
import shutil
import time
from pathlib import Path

import numpy as np
import polars as pl

from newsrec.embeddings import article_embeddings
from newsrec.retrieval import candidate_universe_for_split
from newsrec.semantic import l2_normalise
from newsrec.store import FeatureStore

TOPK = 100


# ---------------------------------------------------------------- helpers

def drop_cache(path: Path | str) -> None:
    """Evict a file from the page cache without root.

    POSIX_FADV_DONTNEED drops clean pages for this file only, which is what we
    want: `echo 3 > /proc/sys/vm/drop_caches` needs root and would also throw away
    everyone else's cache on a shared box.
    """
    fd = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(fd)
    except OSError:
        pass
    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    os.close(fd)


def timed(fn, repeats: int = 3, warm: bool = True):
    """Best-of-N, because we are measuring a floor and not an average."""
    if warm:
        fn()
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def limit_faiss(n: int = 1):
    import faiss
    faiss.omp_set_num_threads(n)


# ---------------------------------------------------------------- part A

def part_a(emb_path: Path, gathers: list[int], repeats: int) -> dict:
    """Where the vectors live, and what touching them randomly costs.

    The comparison that matters holds the *volume* fixed and varies only the
    *pattern*: k contiguous rows against k scattered rows, same bytes either way.
    Comparing a full sequential read against a partial gather -- the obvious thing
    to do -- measures how much less data the gather read, not how much worse it
    read it.
    """
    size_mb = emb_path.stat().st_size / 2 ** 20
    print(f"\n== A. storage hierarchy | {emb_path.name} | {size_mb:,.0f} MB")

    def full_read():
        return np.load(emb_path)

    drop_cache(emb_path)
    t0 = time.perf_counter()
    arr = full_read()
    cold_s = time.perf_counter() - t0
    warm_s = timed(full_read, repeats)
    n_rows, dim = arr.shape
    row_bytes = dim * arr.dtype.itemsize
    del arr

    seq = [{"mode": "full read, cold (NVMe)", "seconds": round(cold_s, 4),
            "mb_per_s": round(size_mb / cold_s, 1)},
           {"mode": "full read, warm (page cache)", "seconds": round(warm_s, 4),
            "mb_per_s": round(size_mb / warm_s, 1)}]
    print(f"  full read cold   {cold_s*1e3:8.1f} ms  {size_mb/cold_s:9,.0f} MB/s")
    print(f"  full read warm   {warm_s*1e3:8.1f} ms  {size_mb/warm_s:9,.0f} MB/s"
          f"   ({cold_s/warm_s:.1f}× faster — the page cache is the whole gap)")

    rng = np.random.default_rng(0)
    rows = []
    for k in gathers:
        k = min(k, n_rows)
        touched_mb = k * row_bytes / 2 ** 20
        start = int(rng.integers(0, n_rows - k + 1))
        contig = np.arange(start, start + k)
        scatter = np.sort(rng.choice(n_rows, size=k, replace=False))

        def read(idx):
            drop_cache(emb_path)
            mm = np.load(emb_path, mmap_mode="r")
            t0 = time.perf_counter()
            _ = np.asarray(mm[idx])
            dt = time.perf_counter() - t0
            del mm
            return dt

        c_s = min(read(contig) for _ in range(repeats))
        r_s = min(read(scatter) for _ in range(repeats))
        rows.append({
            "rows": int(k), "share_of_corpus": round(k / n_rows, 4),
            "touched_mb": round(touched_mb, 2),
            "contiguous_s": round(c_s, 5), "scattered_s": round(r_s, 5),
            "contiguous_mb_per_s": round(touched_mb / c_s, 1),
            "scattered_mb_per_s": round(touched_mb / r_s, 1),
            "scattered_us_per_row": round(r_s / k * 1e6, 2),
            "random_penalty": round(r_s / c_s, 2)})
        print(f"  {k:>7,} rows ({touched_mb:7.1f} MB, {100*k/n_rows:5.1f}% of corpus)  "
              f"contiguous {c_s*1e3:8.2f} ms  scattered {r_s*1e3:8.2f} ms  "
              f"→ random costs {r_s/c_s:5.2f}×  ({r_s/k*1e6:6.1f} µs/row)")

    return {"file": emb_path.name, "size_mb": round(size_mb, 1),
            "rows": int(n_rows), "dim": int(dim), "row_bytes": int(row_bytes),
            "sequential": seq, "same_volume_pattern": rows}


# ---------------------------------------------------------------- part B

def part_b(sub: np.ndarray, q: np.ndarray, dims: list[int], repeats: int) -> dict:
    """Bytes per vector against throughput and fidelity — two ways, because only
    one of them is a fair test.

    B1 **holds the kernel fixed and shrinks the vector**: the same `IndexFlatIP`
    over a Johnson–Lindenstrauss random projection to d' < d. Same code path, same
    BLAS GEMM, strictly fewer bytes. If the scan is bandwidth-bound — which the L2
    harness concluded — throughput should track 1/bytes here, and this is the
    place that claim can actually be falsified. It is also the projection L4
    prices for ANN, so the recall column is worth having on its own.

    B2 **quantises**, and the comparison is confounded on purpose so the confound
    can be named: `IndexFlatIP` dispatches to a BLAS GEMM while
    `IndexScalarQuantizer` runs a per-vector scalar decode loop. Fewer bytes, worse
    kernel. Reporting it as "fp16 is slower than fp32" without that sentence would
    be a measurement artefact dressed up as a result.
    """
    import faiss
    d = sub.shape[1]
    n = sub.shape[0]
    rng_pairs = np.random.default_rng(11)
    print(f"\n== B. bytes vs throughput | {n:,} × {d}d | {q.shape[0]:,} queries, 1 thread")

    def bench(ix, queries):
        ix.search(queries[:64], TOPK)
        best = float("inf")
        for _ in range(repeats):
            t0 = time.perf_counter()
            _, idx = ix.search(queries, TOPK)
            best = min(best, time.perf_counter() - t0)
        return len(queries) / best, idx

    flat = faiss.IndexFlatIP(d)
    flat.add(sub)
    base_qps, gt = bench(flat, q)

    def fidelity(idx):
        return float(np.mean([len(set(a) & set(b)) / TOPK for a, b in zip(gt, idx)]))

    # JL's guarantee is about *distances*, not about which 100 ids come back, so a
    # projection that scores badly on recall@100 can still be doing exactly what the
    # theorem says. Sample pairs and report the inner-product error alongside.
    pair_q = rng_pairs.integers(0, len(q), 4000)
    pair_d = rng_pairs.integers(0, n, 4000)
    true_ip = np.einsum("ij,ij->i", q[pair_q], sub[pair_d])

    def dist_error(vq, vd):
        got = np.einsum("ij,ij->i", vq[pair_q], vd[pair_d])
        return float(np.abs(got - true_ip).mean())

    # ---- B1: same kernel, fewer dimensions (JL projection)
    rng = np.random.default_rng(0)
    b1 = []
    for dd in dims:
        if dd >= d:
            v, qq = sub, q
        else:
            R = (rng.standard_normal((d, dd)) / np.sqrt(dd)).astype(np.float32)
            v = np.ascontiguousarray(l2_normalise(sub @ R))
            qq = np.ascontiguousarray(l2_normalise(q @ R))
        ix = faiss.IndexFlatIP(dd)
        ix.add(v)
        qps, idx = bench(ix, qq)
        mb = n * dd * 4 / 2 ** 20
        err = dist_error(qq, v)
        # MACs per query is n*dd; dividing by the measured time exposes whether the
        # GEMM or the d-independent top-K selection is setting the pace.
        gflops = (n * dd * 2 * qps) / 1e9
        fid = fidelity(idx)
        b1.append({"dim": int(dd), "index_mb": round(mb, 1), "qps": round(qps, 1),
                   "speedup_vs_full": round(qps / base_qps, 2),
                   "bytes_ratio": round(d / dd, 2), "gflop_per_s": round(gflops, 1),
                   "mean_inner_product_error": round(err, 5),
                   f"recall@{TOPK}_vs_exact": round(fid, 4)})
        print(f"  JL d={dd:<4d} (same GEMM kernel) {mb:8.1f} MB  {qps:9,.0f} q/s  "
              f"{qps/base_qps:5.2f}× for {d/dd:5.1f}× fewer bytes  "
              f"{gflops:6.1f} GFLOP/s  ip_err {err:.4f}  recall {fid:.4f}")

    # ---- B2: quantisation, kernel changes underneath
    b2 = [{"precision": "float32 (IndexFlatIP, BLAS GEMM)", "bytes_per_dim": 4.0,
           "index_mb": round(n * d * 4 / 2 ** 20, 1), "qps": round(base_qps, 1),
           "speedup_vs_fp32": 1.0, f"recall@{TOPK}_vs_exact": 1.0}]
    for label, qt, bpd in (("float16 (SQ fp16, scalar decode)", faiss.ScalarQuantizer.QT_fp16, 2.0),
                           ("int8 (SQ 8bit, scalar decode)", faiss.ScalarQuantizer.QT_8bit, 1.0),
                           ("int4 (SQ 4bit, scalar decode)", faiss.ScalarQuantizer.QT_4bit, 0.5)):
        ix = faiss.IndexScalarQuantizer(d, qt, faiss.METRIC_INNER_PRODUCT)
        ix.train(sub)
        ix.add(sub)
        qps, idx = bench(ix, q)
        b2.append({"precision": label, "bytes_per_dim": bpd,
                   "index_mb": round(n * d * bpd / 2 ** 20, 1), "qps": round(qps, 1),
                   "speedup_vs_fp32": round(qps / base_qps, 2),
                   f"recall@{TOPK}_vs_exact": round(fidelity(idx), 4)})
        print(f"  {label:36s} {n*d*bpd/2**20:8.1f} MB  {qps:9,.0f} q/s  "
              f"{qps/base_qps:5.2f}×  fidelity {fidelity(idx):.4f}")

    return {"n_vectors": int(n), "dim": int(d), "n_queries": int(q.shape[0]),
            "b1_same_kernel_fewer_dims": b1, "b2_quantisation": b2}


# ---------------------------------------------------------------- part C

def build_hnsw(v: np.ndarray, M: int = 16, efc: int = 200, efs: int = 64):
    import faiss
    ix = faiss.IndexHNSWFlat(v.shape[1], M, faiss.METRIC_INNER_PRODUCT)
    ix.hnsw.efConstruction = efc
    ix.add(v)
    ix.hnsw.efSearch = efs
    return ix


def index_bytes(ix) -> int:
    import faiss
    return len(faiss.serialize_index(ix))


def part_c(sub: np.ndarray, q: np.ndarray, n_days: int, base_share: float,
           merge_every: int, repeats: int) -> dict:
    """Absorbing a week of new articles: rebuild vs append vs segments.

    The obvious framing -- "all three hold the same vectors, so this is a pure
    read/update/memory trade" -- is only true for two of them. HNSW is a greedily
    built graph, so inserting incrementally can leave a worse graph than building
    the same vectors from scratch, and that cost would be invisible in throughput.
    So every strategy is scored against an exact flat search over the same vectors
    on the same day, and the fidelity column is where an incremental index would
    give itself away.
    """
    n = sub.shape[0]
    base_n = int(n * base_share)
    per_day = (n - base_n) // n_days
    print(f"\n== C. the update axis | base {base_n:,} + {n_days} days × {per_day:,} "
          f"| merge every {merge_every} segments")

    import faiss

    # Ground truth per day, computed once and shared: an exact scan over exactly
    # the vectors that day's index holds.
    print("  computing exact ground truth per day ...", flush=True)
    gt = {}
    for day in range(n_days + 1):
        upto = base_n + day * per_day
        ex = faiss.IndexFlatIP(sub.shape[1])
        ex.add(sub[:upto])
        gt[day] = ex.search(q, TOPK)[1]

    def fidelity(searchers, day: int) -> float:
        """Merged top-K from these indices against that day's exact answer."""
        parts = [s_.search(q, TOPK) for s_ in searchers]
        if len(parts) == 1:
            got = parts[0][1]
        else:
            offs = [sp[0] for sp in seg_spans_for(searchers)]
            sc = np.concatenate([p[0] for p in parts], axis=1)
            ids = np.concatenate([p[1] + o for p, o in zip(parts, offs)], axis=1)
            order = np.argsort(-sc, axis=1)[:, :TOPK]
            got = np.take_along_axis(ids, order, axis=1)
        return float(np.mean([len(set(a) & set(b)) / TOPK
                              for a, b in zip(gt[day], got)]))

    def qps_of(searchers) -> float:
        """Throughput of a query answered by k indices merged at the root."""
        def one():
            for s in searchers:
                s.search(q, TOPK)
        one()
        best = min(timed(one, 1, warm=False) for _ in range(repeats))
        return len(q) / best

    span_registry: dict[int, tuple] = {}

    def seg_spans_for(searchers):
        return [span_registry[id(s_)] for s_ in searchers]

    strategies = {}

    # 1. rebuild wholesale, every day -- what the repo does today
    rows = []
    for day in range(n_days + 1):
        upto = base_n + day * per_day
        t0 = time.perf_counter()
        ix = build_hnsw(sub[:upto])
        upd = time.perf_counter() - t0
        span_registry[id(ix)] = (0, upto)
        rows.append({"day": day, "vectors": upto, "update_s": round(upd, 3),
                     "segments": 1, "bytes": index_bytes(ix),
                     "qps": round(qps_of([ix]), 1),
                     f"recall@{TOPK}_vs_exact": round(fidelity([ix], day), 4)})
        print(f"  rebuild   day {day}: {upto:>7,} vec  update {upd:7.2f}s  "
              f"1 seg  {rows[-1]['qps']:8,.0f} q/s  "
              f"fidelity {rows[-1][f'recall@{TOPK}_vs_exact']:.4f}")
    strategies["rebuild"] = rows

    # 2. append in place -- HNSW supports incremental add
    rows = []
    ix = build_hnsw(sub[:base_n])
    span_registry[id(ix)] = (0, base_n)
    rows.append({"day": 0, "vectors": base_n, "update_s": None, "segments": 1,
                 "bytes": index_bytes(ix), "qps": round(qps_of([ix]), 1),
                 f"recall@{TOPK}_vs_exact": round(fidelity([ix], 0), 4)})
    for day in range(1, n_days + 1):
        lo, hi = base_n + (day - 1) * per_day, base_n + day * per_day
        t0 = time.perf_counter()
        ix.add(sub[lo:hi])
        upd = time.perf_counter() - t0
        span_registry[id(ix)] = (0, hi)
        rows.append({"day": day, "vectors": hi, "update_s": round(upd, 3),
                     "segments": 1, "bytes": index_bytes(ix),
                     "qps": round(qps_of([ix]), 1),
                     f"recall@{TOPK}_vs_exact": round(fidelity([ix], day), 4)})
        print(f"  append    day {day}: {hi:>7,} vec  update {upd:7.2f}s  "
              f"1 seg  {rows[-1]['qps']:8,.0f} q/s  "
              f"fidelity {rows[-1][f'recall@{TOPK}_vs_exact']:.4f}")
    strategies["append_in_place"] = rows

    # 3. LSM-shaped: a base index plus one small segment per day, merged on a
    #    threshold. Reads pay for every segment; writes pay almost nothing until
    #    the merge, which pays for all of them at once.
    rows = []
    base_ix = build_hnsw(sub[:base_n])
    span_registry[id(base_ix)] = (0, base_n)
    segs = [base_ix]
    rows.append({"day": 0, "vectors": base_n, "update_s": None, "segments": 1,
                 "merge_s": 0.0, "bytes": sum(index_bytes(s) for s in segs),
                 "qps": round(qps_of(segs), 1),
                 f"recall@{TOPK}_vs_exact": round(fidelity(segs, 0), 4)})
    for day in range(1, n_days + 1):
        lo, hi = base_n + (day - 1) * per_day, base_n + day * per_day
        t0 = time.perf_counter()
        seg = build_hnsw(sub[lo:hi])
        span_registry[id(seg)] = (lo, hi)
        segs.append(seg)
        upd = time.perf_counter() - t0
        merge_s = 0.0
        if len(segs) > merge_every:
            t0 = time.perf_counter()
            merged = build_hnsw(sub[:hi])
            span_registry[id(merged)] = (0, hi)
            segs = [merged]
            merge_s = time.perf_counter() - t0
        rows.append({"day": day, "vectors": hi, "update_s": round(upd, 3),
                     "segments": len(segs), "merge_s": round(merge_s, 3),
                     "bytes": sum(index_bytes(s) for s in segs),
                     "qps": round(qps_of(segs), 1),
                     f"recall@{TOPK}_vs_exact": round(fidelity(segs, day), 4)})
        print(f"  segments  day {day}: {hi:>7,} vec  update {upd:7.2f}s "
              f"+ merge {merge_s:6.2f}s  {len(segs)} seg  {rows[-1]['qps']:8,.0f} q/s  "
              f"fidelity {rows[-1][f'recall@{TOPK}_vs_exact']:.4f}")
    strategies["lsm_segments"] = rows

    return {"n_vectors": int(n), "base_vectors": base_n, "per_day": per_day,
            "n_days": n_days, "merge_every": merge_every,
            "n_queries": int(q.shape[0]), "strategies": strategies}


# ---------------------------------------------------------------- part D

def part_d(fs: FeatureStore, emb: np.ndarray, tmp: Path, repeats: int) -> dict:
    """Compression: compute traded for bytes, on both halves of the store."""
    print(f"\n== D. compression | scratch {tmp}")
    tmp.mkdir(parents=True, exist_ok=True)
    src = fs.root / "articles.parquet"
    df = pl.read_parquet(src)
    table_rows = []
    for codec in ("uncompressed", "snappy", "lz4", "zstd"):
        path = tmp / f"articles_{codec}.parquet"
        t0 = time.perf_counter()
        df.write_parquet(path, compression=codec)
        write_s = time.perf_counter() - t0
        size_mb = path.stat().st_size / 2 ** 20
        cold_s = float("inf")
        for _ in range(repeats):
            drop_cache(path)
            t0 = time.perf_counter()
            pl.read_parquet(path)
            cold_s = min(cold_s, time.perf_counter() - t0)
        warm_s = timed(lambda p=path: pl.read_parquet(p), repeats)
        table_rows.append({"codec": codec, "size_mb": round(size_mb, 1),
                           "write_s": round(write_s, 3),
                           "cold_read_s": round(cold_s, 3), "warm_read_s": round(warm_s, 3),
                           "cold_mb_per_s": round(size_mb / cold_s, 1)})
        print(f"  articles.parquet {codec:14s} {size_mb:8.1f} MB  write {write_s:6.2f}s  "
              f"cold read {cold_s:6.2f}s  warm {warm_s:6.2f}s")
        path.unlink()

    vec_rows = []
    for label, arr in (("float32", emb.astype(np.float32)),
                       ("float16", emb.astype(np.float16))):
        path = tmp / f"vectors_{label}.npy"
        np.save(path, arr)
        size_mb = path.stat().st_size / 2 ** 20
        cold_s = float("inf")
        for _ in range(repeats):
            drop_cache(path)
            t0 = time.perf_counter()
            np.load(path)
            cold_s = min(cold_s, time.perf_counter() - t0)
        vec_rows.append({"dtype": label, "size_mb": round(size_mb, 1),
                         "cold_read_s": round(cold_s, 3),
                         "cold_mb_per_s": round(size_mb / cold_s, 1)})
        print(f"  vectors {label:8s} {size_mb:8.1f} MB  cold read {cold_s:6.3f}s "
              f"= {size_mb/cold_s:,.0f} MB/s")
        path.unlink()
    shutil.rmtree(tmp, ignore_errors=True)
    return {"parquet": table_rows, "vectors": vec_rows}


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="ebnerd")
    ap.add_argument("--variant", default="large")
    ap.add_argument("--split", default="test")
    ap.add_argument("--embedding", default="contrastive")
    ap.add_argument("--parts", default="abcd")
    ap.add_argument("--scope", default="catalogue", choices=["universe", "catalogue"])
    ap.add_argument("--gathers", default="100,1000,10000,100000")
    ap.add_argument("--dims", default="768,384,192,96,48")
    ap.add_argument("--queries", type=int, default=2000)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--base-share", type=float, default=0.8)
    ap.add_argument("--merge-every", type=int, default=4)
    ap.add_argument("--scratch", default="")
    ap.add_argument("--out", default="reports/l3")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    fs = FeatureStore(a.dataset, a.variant)
    emb = np.asarray(article_embeddings(fs, a.embedding))
    emb_path = fs.root / "embeddings" / f"{a.embedding}.npy"
    universe = (candidate_universe_for_split(fs, a.split, 7) if a.scope == "universe"
                else np.arange(emb.shape[0], dtype=np.int64))
    sub = np.ascontiguousarray(l2_normalise(emb[universe].astype(np.float32)))
    # The store hands back a memmap, and POSIX_FADV_DONTNEED cannot evict pages a
    # live mapping still holds -- with the memmap alive every "cold" read in part A
    # is served from cache and reports the warm number twice. Force a private RAM
    # copy and drop the mapping before measuring anything.
    emb = np.array(emb, dtype=emb.dtype, copy=True)
    gc.collect()
    rng = np.random.default_rng(0)
    qi = rng.choice(sub.shape[0], size=min(a.queries, sub.shape[0]), replace=False)
    q = np.ascontiguousarray(l2_normalise(
        sub[qi] + rng.normal(0, 0.05, (len(qi), sub.shape[1])).astype(np.float32)))

    print(f"L3 storage harness | {a.dataset}/{a.variant} | scope={a.scope} "
          f"({universe.size:,} vectors × {sub.shape[1]}d) | emb={a.embedding}")
    limit_faiss(1)
    res = {"dataset": a.dataset, "variant": a.variant, "scope": a.scope,
           "embedding": a.embedding, "vectors": int(universe.size),
           "dim": int(sub.shape[1]), "top_k": TOPK}
    if "a" in a.parts and emb_path.exists():
        res["part_a_hierarchy"] = part_a(
            emb_path, [int(x) for x in a.gathers.split(",")], a.repeats)
    if "b" in a.parts:
        dims = [min(int(x), sub.shape[1]) for x in a.dims.split(",")]
        res["part_b_precision"] = part_b(sub, q, sorted(set(dims), reverse=True), a.repeats)
    if "c" in a.parts:
        res["part_c_updates"] = part_c(sub, q, a.days, a.base_share,
                                       a.merge_every, a.repeats)
    if "d" in a.parts:
        scratch = Path(a.scratch) if a.scratch else fs.root / "_l3_scratch"
        res["part_d_compression"] = part_d(fs, emb, scratch, a.repeats)

    p = out / f"l3_storage_{a.dataset}_{a.variant}_{a.scope}.json"
    p.write_text(json.dumps(res, indent=2))
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
