"""L5 ablation: posting anatomy, compression, and top-k query processing.

L5 says a search engine is a sorted file -- compressed well, merged lazily, and
traversed in the cheapest order. This repo has the sorted file and none of the
rest. `BM25Index` builds a real dictionary and real postings and then throws the
lecture's entire second half away by scoring with one sparse matmul: no merge, no
skip lists, no cheapest-first ordering, no WAND, no compression, no tiering. The
docstring even admits it ("BM25 scoring is a dense matmul in disguise") without
ever measuring what that costs or buys.

The reason it is not obviously wrong is a property of *this* workload that the
lecture never contemplates. L5's query is two to five terms; a query here is the
bag of terms of a user's entire click history -- on EB-NeRD, around a thousand
distinct terms. Every optimisation in the lecture is a bet that most of the index
can be skipped, and that bet is priced by query length. So the central experiment
is not "is WAND faster" but "at what query length does WAND stop paying", which
turns an inherited implementation choice into a measured one.

Six parts:

A. **What is actually in a posting.** The slide prices docIDs at x1, +tf at x1.3
   and +positions at x2-4. All three are measurable here, and the position
   multiplier is a direct function of mean term frequency, which on 16-token news
   documents is nothing like a web page's.

B. **Compression.** d-gaps, then v-byte, block bit-packing (PForDelta's core),
   Elias-gamma and Roaring, against the slide's bits/gap table. Then the thesis --
   "shrink bytes to widen the bandwidth bottleneck" -- tested end to end by
   writing the postings out, evicting the page cache and reading them back, which
   is the same experiment L3 part D ran on parquet and got an inverted answer to.

C. **Top-k processing.** TAAT, DAAT, DAAT+WAND and the shipped matmul on identical
   queries, compared on *work* (postings touched, documents fully scored) rather
   than only on wall-clock, because a Python WAND measures Python. Rank-safety is
   verified against the exhaustive top-k rather than assumed. Then the query-length
   sweep that explains the whole design.

D. **Skips and cheapest-first.** The sqrt(L) skip list on the slide's own example
   -- a long list intersected with a short one -- and conjunction ordering by
   ascending document frequency, both counted in steps.

E. **Caching and tiering.** The term stream real queries produce is Zipf, so a
   postings cache should work; measure the hit rate against cache size and the
   bytes it saves. Then a hot tier by popularity, with the quality loss of early
   termination measured in recall rather than asserted to be small.

F. **Building the index.** The shipped build is single-pass and in-RAM. SPIMI
   spills sorted runs and merges them, capping memory at the block size. Measure
   both, in peak RSS and wall-clock, and price the MapReduce arithmetic the slide
   quotes against the rate we actually achieve.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import heapq
import json
import shutil
import threading
import time
from pathlib import Path

import numpy as np
import polars as pl
import scipy.sparse as sp

import newsrec.lexical as lexical
from newsrec.lexical import BM25Index, tokenize
from newsrec.retrieval import candidate_universe_for_split, user_histories
from newsrec.store import FeatureStore

KS = (50, 100, 200)
TOPK = 100


def drop_cache(path: Path | str) -> None:
    """Evict a file from the page cache without root -- see l3_storage.py."""
    fd = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(fd)
    except OSError:
        pass
    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    os.close(fd)


class RSSWatcher:
    """Sample resident set size while a block runs.

    `ru_maxrss` is a high-water mark that never falls, so it cannot separate two
    builds in one process. Sampling /proc/self/statm can, and the peak of a
    10 ms sampler is close enough to compare a build that holds everything in RAM
    against one that spills.
    """

    def __init__(self, interval: float = 0.01):
        self.interval, self.peak, self._stop = interval, 0, threading.Event()

    def _run(self) -> None:
        page = os.sysconf("SC_PAGE_SIZE")
        while not self._stop.wait(self.interval):
            try:
                rss = int(open("/proc/self/statm").read().split()[1]) * page
            except OSError:
                return
            self.peak = max(self.peak, rss)

    def __enter__(self) -> "RSSWatcher":
        self.baseline = int(open("/proc/self/statm").read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
        self.peak = self.baseline
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._t.join(timeout=1.0)

    @property
    def delta_mb(self) -> float:
        return (self.peak - self.baseline) / 1e6


# ------------------------------------------------------- term-major postings

class Postings:
    """The transpose L5 actually describes: term -> sorted docID list.

    `BM25Index` stores a document-major CSR because a batch of queries is then one
    sparse matmul. Every operator in this lecture -- merge, skip, WAND -- needs the
    other orientation, so it is built once here and shared by every part below.
    Documents are renumbered densely inside the candidate universe, which is both
    what a real shard does and what makes a bitmap encoding meaningful.
    """

    def __init__(self, idx: BM25Index, universe: np.ndarray | None = None):
        W = idx._W.tocsr()
        if universe is not None:
            W = W[universe]
        self.n_docs, self.n_terms = W.shape
        self.universe = universe
        C = W.tocsc()
        self.t_start = C.indptr[:-1].astype(np.int64)
        self.t_end = C.indptr[1:].astype(np.int64)
        self.doc = C.indices.astype(np.int32)         # sorted within each term
        self.w = C.data.astype(np.float32)
        self.df = (self.t_end - self.t_start).astype(np.int64)
        # per-term upper bound -- the "max score" WAND prunes with
        self.maxw = np.zeros(self.n_terms, dtype=np.float32)
        nz = self.df > 0
        self.maxw[nz] = np.maximum.reduceat(self.w, self.t_start[nz])
        self.n_postings = int(self.doc.size)

    def gaps(self) -> tuple[np.ndarray, np.ndarray]:
        """d-gaps within each term, and the postings-per-term lengths.

        The first entry of every list is stored as an absolute docID; the rest are
        differences. Sorted lists plus Zipf is what makes the differences small,
        which is the entire premise of the next twelve slides.
        """
        g = np.diff(self.doc.astype(np.int64), prepend=0)
        g[self.t_start[self.df > 0]] = self.doc[self.t_start[self.df > 0]]
        return g.astype(np.uint32), self.df


# ------------------------------------------------------------- integer codes

def vbyte_encode(vals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Byte-aligned variable-length integers, high bit terminating (IIR 5.3).

    Vectorised: the byte layout is entirely determined by each value's width, so
    the whole buffer is built with repeat/arange rather than a Python loop over
    three million postings.
    """
    v = vals.astype(np.uint32)
    nbytes = np.ones(v.size, dtype=np.int64)
    for shift in (7, 14, 21, 28):
        nbytes += (v >= (np.uint64(1) << np.uint64(shift))).astype(np.int64)
    total = int(nbytes.sum())
    starts = np.cumsum(nbytes) - nbytes
    grp = np.repeat(np.arange(v.size), nbytes)
    within = np.arange(total) - starts[grp]
    shift = (7 * (nbytes[grp] - 1 - within)).astype(np.uint64)
    out = ((v[grp].astype(np.uint64) >> shift) & np.uint64(0x7F)).astype(np.uint8)
    out[starts + nbytes - 1] |= np.uint8(0x80)
    return out, nbytes


def vbyte_decode(buf: np.ndarray) -> np.ndarray:
    """Inverse of `vbyte_encode`, also branch-free.

    Group boundaries are the terminator bits, so a cumulative sum labels every
    byte with the value it belongs to and `add.reduceat` sums the shifted payloads
    exactly in int64 -- no float rounding, which a bincount formulation would risk.
    """
    is_end = (buf & np.uint8(0x80)) != 0
    ends = np.flatnonzero(is_end)
    starts = np.r_[0, ends[:-1] + 1]
    nbytes = ends - starts + 1
    grp = np.repeat(np.arange(ends.size), nbytes)
    within = np.arange(buf.size) - starts[grp]
    shift = (7 * (nbytes[grp] - 1 - within)).astype(np.int64)
    payload = (buf & np.uint8(0x7F)).astype(np.int64) << shift
    return np.add.reduceat(payload, starts).astype(np.uint32)


BLOCK = 128


def bitpack_encode(vals: np.ndarray, block: int = BLOCK) -> tuple[np.ndarray, np.ndarray, int]:
    """Block-wise fixed-width packing -- the core of PForDelta / SIMD-BP128.

    Each block of `block` gaps is stored at the width its largest member needs, so
    a run of tiny gaps costs 3 bits each while one large gap widens only its own
    block. Real PForDelta additionally exceptions the outliers out; this is the
    part that does the work, and it is the part a SIMD decoder vectorises.
    """
    v = vals.astype(np.uint32)
    pad = (-v.size) % block
    vp = np.r_[v, np.zeros(pad, dtype=np.uint32)].reshape(-1, block)
    widths = np.maximum(1, np.ceil(np.log2(vp.max(axis=1) + 1.0)).astype(np.int64))
    chunks = []
    for row, b in zip(vp, widths):
        bits = ((row[:, None] >> np.arange(b, dtype=np.uint32)[None, :]) & 1).astype(np.uint8)
        chunks.append(np.packbits(bits.ravel()))
    return np.concatenate(chunks), widths, int(v.size)


def bitpack_decode(buf: np.ndarray, widths: np.ndarray, n: int,
                   block: int = BLOCK) -> np.ndarray:
    """Unpack block by block. numpy speed, not SIMD speed -- see the report."""
    out = np.empty(widths.size * block, dtype=np.uint32)
    pos = 0
    for i, b in enumerate(widths):
        nb = (block * int(b) + 7) // 8
        bits = np.unpackbits(buf[pos:pos + nb])[:block * int(b)].reshape(block, int(b))
        out[i * block:(i + 1) * block] = (bits.astype(np.uint32)
                                          << np.arange(int(b), dtype=np.uint32)[None, :]).sum(axis=1)
        pos += nb
    return out[:n]


def elias_gamma_bits(gaps: np.ndarray) -> int:
    """Exact size of Elias-gamma, without building it.

    Size is what the slide tabulates and size is exactly computable: a gap g costs
    2*floor(log2 g) + 1 bits. A decoder is not written because a bit-aligned
    decoder in numpy would measure numpy's bit-shuffling and not the code -- the
    honest column here is bits/gap, and the report says so instead of quoting a
    throughput that means nothing.
    """
    g = np.maximum(gaps.astype(np.int64), 1)
    return int((2 * np.floor(np.log2(g)).astype(np.int64) + 1).sum())


def roaring_bytes(post: Postings) -> int:
    """Roaring's size rule, applied per term and per 2^16 docID chunk.

    A chunk holding at most 4096 docIDs is an array of 16-bit values; a denser one
    becomes a 65,536-bit bitmap. This is a size model, not an implementation -- but
    it is Roaring's actual decision rule, so the bytes are the bytes Roaring would
    write.
    """
    total = 0
    for t in range(post.n_terms):
        s, e = post.t_start[t], post.t_end[t]
        if e == s:
            continue
        docs = post.doc[s:e]
        for _, cnt in zip(*np.unique(docs >> 16, return_counts=True)):
            total += 8192 if cnt > 4096 else int(cnt) * 2
        total += 8                                   # container header per term
    return total


# ==================================================================== part A

def part_a(fs: FeatureStore, idx: BM25Index, out: dict) -> None:
    """What is actually inside a posting, at this corpus's term statistics.

    Measured on the whole index rather than the candidate universe: the layout
    decision is made once, for every posting the index will ever hold, and scoping
    it to one week's live articles would quietly divide the postings by the
    universe while still counting occurrences over the whole corpus.
    """
    toks = tokenize(fs.texts(), fs.lang())
    occurrences = toks.height                         # every token position
    postings = int(idx.doc_ids.size)
    mean_tf = occurrences / max(postings, 1)
    tf = idx.tf
    out["anatomy"] = {
        "n_docs": idx.n_docs, "n_terms": idx.vocab_size,
        "postings": postings, "token_occurrences": occurrences,
        "mean_tf": mean_tf,
        "tf_eq_1_share": float((tf == 1).mean()),
        "tf_le_255_share": float((tf <= 255).mean()),
        "avgdl": idx.avgdl,
        "layouts": [
            {"layout": "docIDs only", "bytes": postings * 4, "multiplier": 1.0,
             "slide": 1.0},
            {"layout": "+ tf (uint8)", "bytes": postings * 5, "multiplier": 1.25,
             "slide": 1.3},
            {"layout": "+ tf (uint32)", "bytes": postings * 8, "multiplier": 2.0,
             "slide": None},
            {"layout": "+ positions (uint32)",
             "bytes": postings * 8 + occurrences * 4,
             "multiplier": (postings * 8 + occurrences * 4) / (postings * 4),
             "slide": "2-4"},
            {"layout": "+ positions (v-byte gaps)",
             "bytes": postings * 5 + occurrences * 1,
             "multiplier": (postings * 5 + occurrences * 1) / (postings * 4),
             "slide": "2-4"},
        ],
    }


# ==================================================================== part B

def part_b(post: Postings, out: dict, tmp: Path, repeats: int,
           stop_gaps: np.ndarray | None = None) -> None:
    """Compression: bits per gap, decode throughput, and the bandwidth thesis."""
    gaps, df = post.gaps()
    n = gaps.size
    raw_bits = n * 32

    vb, nbytes = vbyte_encode(gaps)
    bp, widths, _ = bitpack_encode(gaps)
    gamma_bits = elias_gamma_bits(gaps)
    roar = roaring_bytes(post)

    codes = [
        {"code": "raw docIDs (uint32)", "bytes": n * 4, "bits_per_gap": 32.0, "slide": None},
        {"code": "d-gaps + v-byte", "bytes": int(vb.size),
         "bits_per_gap": vb.size * 8 / n, "slide": "8-9"},
        {"code": "d-gaps + bit-packed blocks", "bytes": int(bp.size + widths.size),
         "bits_per_gap": (bp.size + widths.size) * 8 / n, "slide": "6-7"},
        {"code": "d-gaps + Elias-gamma", "bytes": int(np.ceil(gamma_bits / 8)),
         "bits_per_gap": gamma_bits / n, "slide": "5-6"},
        {"code": "Roaring bitmaps", "bytes": int(roar),
         "bits_per_gap": roar * 8 / n, "slide": "~2 (dense)"},
    ]
    for c in codes:
        c["ratio_vs_raw"] = (n * 4) / max(c["bytes"], 1)

    # decode throughput, best of `repeats` -- the same best-of-N discipline the L3
    # harness needed once a single cold read produced an outlier
    def timed(fn, *args) -> tuple[float, np.ndarray]:
        best, res = float("inf"), None
        for _ in range(repeats):
            t0 = time.perf_counter()
            res = fn(*args)
            best = min(best, time.perf_counter() - t0)
        return best, res

    # Decode is timed all the way to docIDs, so the gap codes pay for their prefix
    # sum and the raw layout pays nothing -- which is the actual comparison. Timing
    # only the unpacking would credit v-byte with work it has not finished.
    t_vb, dec_vb = timed(lambda x: np.cumsum(vbyte_decode(x), dtype=np.int64), vb)
    t_bp, dec_bp = timed(lambda *x: np.cumsum(bitpack_decode(*x), dtype=np.int64),
                         bp, widths, n)
    t_raw, _ = timed(lambda x: x.astype(np.uint32).copy(), gaps)
    ref = np.cumsum(gaps, dtype=np.int64)
    assert np.array_equal(dec_vb, ref), "v-byte round-trip failed"
    assert np.array_equal(dec_bp, ref), "bit-pack round-trip failed"
    for c, t in (("d-gaps + v-byte", t_vb), ("d-gaps + bit-packed blocks", t_bp),
                 ("raw docIDs (uint32)", t_raw)):
        row = next(x for x in codes if x["code"] == c)
        row["decode_seconds"] = t
        row["decode_mpostings_per_s"] = n / t / 1e6
        row["decode_gb_per_s"] = row["bytes"] / t / 1e9   # compressed bytes consumed
        row["decode_note"] = ("no decode -- a memcpy, shown as the ceiling"
                              if c.startswith("raw") else "numpy, not SIMD")
    out["codes"] = {"postings": n, "raw_bits": raw_bits, "rows": codes,
                    "mean_gap": float(gaps.mean()), "median_gap": float(np.median(gaps)),
                    "p99_gap": float(np.quantile(gaps, 0.99))}

    # d-gaps are small because Zipf makes some lists dense. The analyzer deletes
    # exactly those lists: stopword removal is Q2's recall decision, and this is
    # the bill it quietly sends to L5. Same corpus, both analyzers, bits/gap each.
    if stop_gaps is not None:
        g2 = stop_gaps
        vb2, _ = vbyte_encode(g2)
        out["codes"]["analyzer_effect"] = {
            "with_stopwords": {"postings": int(g2.size), "median_gap": float(np.median(g2)),
                               "vbyte_bits_per_gap": vb2.size * 8 / g2.size,
                               "vbyte_mb": vb2.size / 1e6},
            "without_stopwords": {"postings": int(n), "median_gap": float(np.median(gaps)),
                                  "vbyte_bits_per_gap": vb.size * 8 / n,
                                  "vbyte_mb": vb.size / 1e6},
        }

    # ---- the thesis: does shrinking the bytes widen the bottleneck *here*?
    tmp.mkdir(parents=True, exist_ok=True)
    files = {"raw": gaps.astype(np.uint32), "vbyte": vb, "bitpacked": bp}
    bench = []
    for name, arr in files.items():
        f = tmp / f"postings_{name}.bin"
        f.write_bytes(arr.tobytes())
        cold = []
        for _ in range(repeats):
            drop_cache(f)
            t0 = time.perf_counter()
            raw = np.fromfile(f, dtype=arr.dtype)
            if name == "vbyte":
                _ = vbyte_decode(raw)
            elif name == "bitpacked":
                _ = bitpack_decode(raw, widths, n)
            cold.append(time.perf_counter() - t0)
        warm = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            raw = np.fromfile(f, dtype=arr.dtype)
            if name == "vbyte":
                _ = vbyte_decode(raw)
            elif name == "bitpacked":
                _ = bitpack_decode(raw, widths, n)
            warm.append(time.perf_counter() - t0)
        bench.append({"code": name, "file_mb": f.stat().st_size / 1e6,
                      "cold_seconds": min(cold), "warm_seconds": min(warm),
                      "cold_mb_per_s": f.stat().st_size / 1e6 / min(cold)})
        f.unlink()

    # The slide's own arithmetic, so a loss here is attributable rather than just
    # reported: at 250 us/MB of SSD, how fast would the decoder have to run for the
    # compressed layout to win? The slide assumes 1-5 GB/s (SIMD C); this decoder
    # is numpy, and the gap between those two numbers is the whole explanation.
    raw_row = next(r for r in bench if r["code"] == "raw")
    model = []
    for r in bench:
        if r["code"] == "raw":
            continue
        code = next(c for c in codes
                    if c["code"].startswith("d-gaps")
                    and (("v-byte" in c["code"]) == (r["code"] == "vbyte")))
        io_saved_s = (raw_row["file_mb"] - r["file_mb"]) * 250e-6
        model.append({
            "code": r["code"],
            "mb_saved": raw_row["file_mb"] - r["file_mb"],
            "io_saved_ms_at_slide_ssd": io_saved_s * 1e3,
            "measured_decode_ms": code.get("decode_seconds", float("nan")) * 1e3,
            "required_decode_gb_per_s": (r["file_mb"] / 1e3) / max(io_saved_s, 1e-12),
            "measured_decode_gb_per_s": code.get("decode_gb_per_s", float("nan")),
        })
    out["bandwidth_thesis"] = bench
    out["thesis_model"] = model


# ==================================================================== part C

def exhaustive_topk(post: Postings, qt: np.ndarray, qw: np.ndarray, k: int
                    ) -> tuple[np.ndarray, np.ndarray]:
    """Ground truth: score every document that contains any query term."""
    acc = np.zeros(post.n_docs, dtype=np.float64)
    for t, w in zip(qt.tolist(), qw.tolist()):
        s, e = post.t_start[t], post.t_end[t]
        np.add.at(acc, post.doc[s:e], w * post.w[s:e])
    kk = min(k, post.n_docs)
    part = np.argpartition(-acc, kk - 1)[:kk]
    order = part[np.argsort(-acc[part])]
    return order, acc[order]


def taat(post: Postings, qt: np.ndarray, qw: np.ndarray, k: int) -> dict:
    """Term-at-a-time: sweep one list at a time into a dense accumulator.

    Simple, sequential and cache-friendly, and its memory is proportional to the
    candidate documents rather than to k -- which on a shard is a float per
    document, allocated whether or not the query touches it.
    """
    acc = np.zeros(post.n_docs, dtype=np.float64)
    touched = 0
    order = np.argsort(post.df[qt])           # cheapest-first by df, as the slide says
    for t, w in zip(qt[order].tolist(), qw[order].tolist()):
        s, e = post.t_start[t], post.t_end[t]
        np.add.at(acc, post.doc[s:e], w * post.w[s:e])
        touched += e - s
    nz = np.flatnonzero(acc)
    kk = min(k, nz.size)
    part = nz[np.argpartition(-acc[nz], kk - 1)[:kk]] if nz.size else nz
    top = part[np.argsort(-acc[part])]
    return {"top": top, "scores": acc[top], "postings_touched": int(touched),
            "contributions": int(touched),     # TAAT scores every posting it reads
            "docs_scored": int(nz.size), "accumulator_bytes": int(acc.nbytes)}


def daat(post: Postings, qt: np.ndarray, qw: np.ndarray, k: int,
         use_wand: bool = False, max_iters: int = 20_000_000) -> dict:
    """Document-at-a-time, optionally with WAND pruning.

    Both share one loop so the counters are comparable: the only difference is the
    threshold. With `use_wand=False` theta is -inf, every candidate document is
    scored, and the pivot is always the smallest current docID -- which is exactly
    plain DAAT. With WAND, theta is the k-th best score so far and the pivot is the
    first document whose cumulative term upper bounds can beat it, so everything
    before the pivot is skipped without being scored.

    Bounded memory by construction: one cursor per query term and a k-element heap,
    against TAAT's float per document.
    """
    T = qt.size
    cur = post.t_start[qt].copy()
    end = post.t_end[qt].copy()
    INF = np.int64(np.iinfo(np.int32).max)
    cur_doc = np.where(cur < end, post.doc[np.minimum(cur, end - 1)], INF).astype(np.int64)
    ub = (qw * post.maxw[qt]).astype(np.float64)      # per-term contribution ceiling

    heap: list[tuple[float, int]] = []
    theta = -np.inf
    touched = scored = iters = contrib = 0
    while iters < max_iters:
        iters += 1
        alive = cur_doc < INF
        if not alive.any():
            break
        order = np.argsort(cur_doc, kind="stable")
        if use_wand:
            cum = np.cumsum(np.where(cur_doc[order] < INF, ub[order], 0.0))
            over = np.flatnonzero(cum > theta)
            if over.size == 0:
                break                                  # nothing left can enter the heap
            piv = int(over[0])
        else:
            piv = 0
        pivot_doc = int(cur_doc[order[piv]])
        if pivot_doc >= INF:
            break
        if int(cur_doc[order[0]]) == pivot_doc:
            hit = np.flatnonzero(cur_doc == pivot_doc)
            s = float(np.sum(qw[hit] * post.w[cur[hit]]))
            scored += 1
            contrib += hit.size          # the multiply-adds actually performed
            if len(heap) < k:
                heapq.heappush(heap, (s, pivot_doc))
            elif s > heap[0][0]:
                heapq.heapreplace(heap, (s, pivot_doc))
            if use_wand and len(heap) == k:
                theta = heap[0][0]
            cur[hit] += 1
            touched += hit.size
            moved = hit
        else:
            # advance any term sitting before the pivot; the standard choice is the
            # one with the largest upper bound, since it is the most likely to be
            # the reason a later document survives
            before = np.flatnonzero((cur_doc < pivot_doc) & (cur_doc < INF))
            pick = before[np.argmax(ub[before])]
            s0, e0 = cur[pick], end[pick]
            nxt = s0 + int(np.searchsorted(post.doc[s0:e0], pivot_doc, side="left"))
            touched += nxt - s0
            cur[pick] = nxt
            moved = np.array([pick])
        cur_doc[moved] = np.where(cur[moved] < end[moved],
                                  post.doc[np.minimum(cur[moved], end[moved] - 1)], INF)
    top = np.array([d for _, d in sorted(heap, key=lambda x: -x[0])], dtype=np.int64)
    scores = np.array([s for s, _ in sorted(heap, key=lambda x: -x[0])], dtype=np.float64)
    return {"top": top, "scores": scores, "postings_touched": int(touched),
            "contributions": int(contrib), "docs_scored": int(scored),
            "iterations": int(iters), "accumulator_bytes": int(k * 16 + T * 24)}


def part_c(idx: BM25Index, post: Postings, Q: sp.csr_matrix, out: dict,
           n_queries: int, lengths: list[int], seed: int) -> None:
    """TAAT / DAAT / WAND / matmul on identical queries, counted in work."""
    rng = np.random.default_rng(seed)
    rows = rng.choice(Q.shape[0], size=min(n_queries, Q.shape[0]), replace=False)

    def query_terms(i: int, keep: int | None) -> tuple[np.ndarray, np.ndarray]:
        r = Q[int(i)].tocoo()
        t, w = r.col.astype(np.int64), r.data.astype(np.float64)
        live = post.df[t] > 0
        t, w = t[live], w[live]
        if keep and t.size > keep:
            # truncate by query weight, which is the history term count -- the same
            # thing a production system does when it caps a long query
            sel = np.argsort(-w)[:keep]
            t, w = t[sel], w[sel]
        return t, w

    full_len = int(np.median([query_terms(i, None)[0].size for i in rows]))
    results = []
    for keep in lengths:
        label = "full" if keep == 0 else str(keep)
        agg = {"taat": [], "daat": [], "wand": []}
        times = {"taat": 0.0, "daat": 0.0, "wand": 0.0, "matmul": 0.0}
        safe = {"daat": 0, "wand": 0}
        n_eff = 0
        med_terms = []
        for i in rows:
            qt, qw = query_terms(i, None if keep == 0 else keep)
            if qt.size == 0:
                continue
            n_eff += 1
            med_terms.append(qt.size)
            truth_docs, truth_scores = exhaustive_topk(post, qt, qw, TOPK)
            for name, fn in (("taat", lambda: taat(post, qt, qw, TOPK)),
                             ("daat", lambda: daat(post, qt, qw, TOPK, use_wand=False)),
                             ("wand", lambda: daat(post, qt, qw, TOPK, use_wand=True))):
                t0 = time.perf_counter()
                r = fn()
                times[name] += time.perf_counter() - t0
                agg[name].append((r["postings_touched"], r["docs_scored"],
                                  r["accumulator_bytes"], r["contributions"]))
                if name in safe:
                    # rank-safety: the scores must match, not merely the doc ids --
                    # ties make the id list ambiguous and the score list is what the
                    # pruning argument is actually about
                    got = np.sort(r["scores"])[::-1][:TOPK]
                    want = np.sort(truth_scores)[::-1][:got.size]
                    safe[name] += int(got.size == want.size
                                      and np.allclose(got, want, rtol=1e-5, atol=1e-6))
        # Build the truncated query block *outside* the timer. Leaving it inside
        # made the matmul look 18x faster on 432-term queries than on 2-term ones,
        # because the truncation loop it was actually timing does nothing when
        # there is nothing to truncate.
        ind, ptr, dat = [], [0], []
        for i in rows:
            qt, qw = query_terms(i, None if keep == 0 else keep)
            ind.append(qt); dat.append(qw); ptr.append(ptr[-1] + qt.size)
        sub = sp.csr_matrix(
            (np.concatenate(dat) if dat else np.zeros(0, dtype=np.float32),
             np.concatenate(ind) if ind else np.zeros(0, dtype=np.int64),
             np.array(ptr)), shape=(len(rows), Q.shape[1]), dtype=np.float32)
        Wsub = idx._W[post.universe] if post.universe is not None else idx._W
        t0 = time.perf_counter()
        S = (sub @ Wsub.T).toarray()
        np.argpartition(-S, TOPK - 1, axis=1)[:, :TOPK]
        times["matmul"] = time.perf_counter() - t0

        total_postings = float(post.n_postings)
        # The sparse product touches exactly the postings of the query's terms --
        # the same set TAAT sweeps. Recording it makes the four rows comparable in
        # work and not only in wall-clock, which is the difference between an
        # algorithmic claim and a statement about scipy.
        mm_work = float(np.mean([post.df[query_terms(i, None if keep == 0 else keep)[0]].sum()
                                 for i in rows]))
        row = {"query_terms": label,
               "median_terms": int(np.median(med_terms)) if med_terms else 0,
               "queries": n_eff, "matmul_ms_per_query": times["matmul"] / max(n_eff, 1) * 1e3,
               "matmul_postings_touched": mm_work}
        for name in ("taat", "daat", "wand"):
            tp = np.array([x[0] for x in agg[name]], dtype=np.float64)
            ds = np.array([x[1] for x in agg[name]], dtype=np.float64)
            ab = np.array([x[2] for x in agg[name]], dtype=np.float64)
            cn = np.array([x[3] for x in agg[name]], dtype=np.float64)
            row[name] = {
                "postings_touched": float(tp.mean()) if tp.size else 0.0,
                "contributions": float(cn.mean()) if cn.size else 0.0,
                "docs_scored": float(ds.mean()) if ds.size else 0.0,
                "index_share_touched": float(tp.mean() / total_postings) if tp.size else 0.0,
                "accumulator_bytes": float(ab.mean()) if ab.size else 0.0,
                "ms_per_query": times[name] / max(n_eff, 1) * 1e3,
            }
        base = row["daat"]["docs_scored"] or 1.0
        row["wand_prune_share"] = 1.0 - row["wand"]["docs_scored"] / base
        # Postings *touched* counts cursor movement, so WAND pays for the lists it
        # skips over as well; contributions counts the multiply-adds it avoided,
        # which is the work the pruning argument is actually about.
        row["wand_postings_share"] = (row["wand"]["postings_touched"]
                                      / max(row["daat"]["postings_touched"], 1.0))
        row["wand_contribution_share"] = (row["wand"]["contributions"]
                                          / max(row["daat"]["contributions"], 1.0))
        row["rank_safe"] = {k_: v / max(n_eff, 1) for k_, v in safe.items()}
        results.append(row)
    out["query_processing"] = {"median_full_query_terms": full_len,
                               "n_docs": post.n_docs, "top_k": TOPK, "rows": results}


def part_g(fs: FeatureStore, idx: BM25Index, Q: sp.csr_matrix, out: dict,
           split: str, scope_days: list[int], n_queries: int, seed: int,
           max_iters: int = 400_000) -> None:
    """Where the matmul stops being the right answer.

    Part C compares the four strategies at one candidate universe. That universe is
    a week of live articles -- about two thousand documents -- and at that size
    "skip most of the index" is competing against "do all of it in one vectorised
    kernel", which is not a fair fight in either direction. WAND's advantage is a
    ratio of arithmetic and the matmul's is a constant factor, so the two cross
    somewhere, and where they cross is the assignment's own "where does it break at
    10x" question asked about the query path.

    So: widen the universe and re-measure. Each row is the same queries against a
    larger candidate set.
    """
    rng = np.random.default_rng(seed)
    rowsel = rng.choice(Q.shape[0], size=min(n_queries, Q.shape[0]), replace=False)
    out_rows = []
    print("  universe |    docs |   postings | matmul ms |  WAND ms | WAND docs | work share",
          flush=True)
    for days in scope_days:
        uni = candidate_universe_for_split(fs, split, days) if days else None
        post = Postings(idx, uni)
        n_docs = post.n_docs
        touched = scored = contrib = capped = 0
        t_wand = 0.0
        mm_work = 0.0
        n_eff = 0
        for i in rowsel:
            r = Q[int(i)].tocoo()
            qt, qw = r.col.astype(np.int64), r.data.astype(np.float64)
            live = post.df[qt] > 0
            qt, qw = qt[live], qw[live]
            if qt.size == 0:
                continue
            n_eff += 1
            mm_work += float(post.df[qt].sum())
            t0 = time.perf_counter()
            w = daat(post, qt, qw, TOPK, use_wand=True, max_iters=max_iters)
            t_wand += time.perf_counter() - t0
            capped += int(w["iterations"] >= max_iters)
            touched += w["postings_touched"]; scored += w["docs_scored"]
            contrib += w["contributions"]
        Wsub = idx._W[uni] if uni is not None else idx._W
        sub = Q[rowsel]
        t0 = time.perf_counter()
        S = (sub @ Wsub.T).toarray()
        np.argpartition(-S, min(TOPK, S.shape[1]) - 1, axis=1)
        t_mm = time.perf_counter() - t0
        out_rows.append({
            "universe_days": days, "universe_docs": int(n_docs),
            "postings_in_scope": int(post.n_postings), "queries": n_eff,
            "matmul_ms_per_query": t_mm / max(n_eff, 1) * 1e3,
            "matmul_postings_touched": mm_work / max(n_eff, 1),
            "matmul_dense_cells": float(n_docs),
            "wand_ms_per_query": t_wand / max(n_eff, 1) * 1e3,
            "wand_docs_scored": scored / max(n_eff, 1),
            "wand_contributions": contrib / max(n_eff, 1),
            "wand_work_share": contrib / max(mm_work, 1.0),
            # A pivot loop over ~900 terms can advance one cursor per iteration, so
            # the worst case is superlinear in the candidate set. The cap is
            # reported rather than hidden: a capped row understates WAND's cost and
            # would flatter it, which is the wrong direction to be silent about.
            "iteration_cap": max_iters, "queries_hitting_cap": capped,
        })
        print(f"  {str(days) + ' d' if days else '  all':>8} | {n_docs:>7,} | "
              f"{post.n_postings:>10,} | {out_rows[-1]['matmul_ms_per_query']:9.3f} | "
              f"{out_rows[-1]['wand_ms_per_query']:8.2f} | "
              f"{out_rows[-1]['wand_docs_scored']:9,.0f} | "
              f"{out_rows[-1]['wand_work_share']:10.3f}"
              + (f"  ({capped} capped)" if capped else ""), flush=True)
        del post, S
    out["scale"] = {"split": split, "rows": out_rows}


# ==================================================================== part D

def part_d(post: Postings, out: dict, seed: int, n_pairs: int) -> None:
    """Skip lists and cheapest-first, both counted in steps."""
    rng = np.random.default_rng(seed)
    df = post.df
    long_terms = np.flatnonzero(df >= np.quantile(df[df > 0], 0.999))
    short_terms = np.flatnonzero((df >= 5) & (df <= 50))
    if long_terms.size == 0 or short_terms.size == 0:
        out["skips"] = {"note": "corpus too small for a long/short term pair"}
        return

    def merge_steps(a: np.ndarray, b: np.ndarray, skip: int = 0) -> tuple[int, int]:
        """Two-pointer intersection, optionally with a skip pointer every `skip`.

        Steps are counted, not timed: the slide's claim is about steps ("10^6 ->
        200K, 5x fewer") and a numpy intersect1d would answer a different question.
        """
        i = j = steps = hits = 0
        na, nb = a.size, b.size
        while i < na and j < nb:
            steps += 1
            if a[i] == b[j]:
                hits += 1; i += 1; j += 1
            elif a[i] < b[j]:
                if skip and i + skip < na and a[i + skip] <= b[j]:
                    while i + skip < na and a[i + skip] <= b[j]:
                        i += skip
                        steps += 1
                else:
                    i += 1
            else:
                if skip and j + skip < nb and b[j + skip] <= a[i]:
                    while j + skip < nb and b[j + skip] <= a[i]:
                        j += skip
                        steps += 1
                else:
                    j += 1
        return steps, hits

    rows = []
    for _ in range(n_pairs):
        tl = int(rng.choice(long_terms))
        ts = int(rng.choice(short_terms))
        a = post.doc[post.t_start[tl]:post.t_end[tl]].astype(np.int64)
        b = post.doc[post.t_start[ts]:post.t_end[ts]].astype(np.int64)
        skip = max(2, int(np.sqrt(a.size)))
        naive, hits = merge_steps(a, b, 0)
        skipped, hits2 = merge_steps(a, b, skip)
        assert hits == hits2, "skip pointers changed the answer"
        rows.append({"long_df": int(a.size), "short_df": int(b.size),
                     "skip_stride": skip, "naive_steps": naive,
                     "skip_steps": skipped, "speedup": naive / max(skipped, 1),
                     "skip_index_bytes": int(a.size / skip * 8), "hits": hits})
    out["skips"] = {"pairs": rows,
                    "mean_speedup": float(np.mean([r["speedup"] for r in rows])),
                    "sqrt_l_prediction": "sqrt(L) skips -> ~sqrt(L) fewer steps on the long list"}

    # cheapest-first: the order the slide prescribes, against its reverse
    orders = []
    # Sampled from the head of the df distribution, not uniformly: a conjunction of
    # four random terms in a Zipf index is empty after the first merge and measures
    # nothing. The slide's example is deliberately a mix of frequencies, so the
    # candidates here are terms above the 90th percentile of df.
    common = np.flatnonzero(df >= np.quantile(df[df > 0], 0.90))
    for _ in range(n_pairs):
        terms = rng.choice(common, size=4, replace=False)
        lists = [post.doc[post.t_start[t]:post.t_end[t]].astype(np.int64) for t in terms]
        for label, seq in (("ascending df (cheapest-first)",
                            [lists[i] for i in np.argsort([l.size for l in lists])]),
                           ("descending df",
                            [lists[i] for i in np.argsort([-l.size for l in lists])])):
            acc, steps = seq[0], 0
            for nxt in seq[1:]:
                s, _ = merge_steps(acc, nxt, 0)
                steps += s
                acc = np.intersect1d(acc, nxt)
            orders.append({"order": label, "steps": steps, "result": int(acc.size),
                           "dfs": sorted(int(l.size) for l in lists)})
    asc = [o["steps"] for o in orders if o["order"].startswith("ascending")]
    desc = [o["steps"] for o in orders if o["order"].startswith("descending")]
    out["conjunction_order"] = {
        "trials": len(asc),
        "ascending_mean_steps": float(np.mean(asc)) if asc else 0.0,
        "descending_mean_steps": float(np.mean(desc)) if desc else 0.0,
        "saving": 1 - (np.mean(asc) / np.mean(desc)) if desc and np.mean(desc) else 0.0,
    }


# ==================================================================== part E

def part_e(fs: FeatureStore, idx: BM25Index, post: Postings, post_full: Postings,
           Q: sp.csr_matrix, uids: np.ndarray, hists: list[np.ndarray], split: str,
           universe, out: dict, seed: int, cache_fracs: list[float],
           tiers: list[float]) -> None:
    """A Zipf-driven postings cache, and a hot tier priced in recall."""
    rng = np.random.default_rng(seed)
    rows = rng.permutation(Q.shape[0])
    stream = []
    for i in rows[:min(2000, rows.size)]:
        stream.append(Q[int(i)].tocoo().col.astype(np.int64))
    if not stream:
        return
    flat = np.concatenate(stream)
    freq = np.bincount(flat, minlength=post_full.n_terms)
    # Sized on the whole index: what a postings cache holds is a shard's real
    # lists, not the slice of them that this week's universe happens to touch.
    bytes_per_term = np.zeros(post_full.n_terms)
    bytes_per_term[post_full.df > 0] = post_full.df[post_full.df > 0] * 8

    cache_rows = []
    total_terms = flat.size
    total_bytes = float(bytes_per_term[flat].sum())
    index_bytes = float(bytes_per_term.sum())
    for frac in cache_fracs:
        budget = index_bytes * frac
        # static Zipf cache: hold the most-requested terms that fit, which is the
        # cache the slide's "cache wisely following Zipf's" actually describes
        order = np.argsort(-freq)
        cum = np.cumsum(bytes_per_term[order])
        held = order[cum <= budget]
        mask = np.zeros(post_full.n_terms, dtype=bool)
        mask[held] = True
        hits = int(mask[flat].sum())
        cache_rows.append({
            "budget_share": frac, "budget_mb": budget / 1e6,
            "terms_held": int(held.size),
            "term_share": held.size / max(int((post_full.df > 0).sum()), 1),
            "request_hit_rate": hits / max(total_terms, 1),
            "byte_hit_rate": float(bytes_per_term[flat][mask[flat]].sum()) / max(total_bytes, 1),
        })
    out["cache"] = {"requests": int(total_terms), "distinct_terms": int((freq > 0).sum()),
                    "index_mb": index_bytes / 1e6, "rows": cache_rows}

    # tiering: score only the hottest slice of the universe, and price the loss
    pop = (fs.impressions(split).select("clicked").collect()["clicked"]
           .explode().drop_nulls().to_numpy().astype(np.int64))
    counts = np.bincount(pop, minlength=fs.n_articles)
    uni = universe if universe is not None else np.arange(fs.n_articles)
    order = uni[np.argsort(-counts[uni])]
    full = recall_of(fs, split, uids, hists, idx, Q, uni)
    tier_rows = []
    for frac in tiers:
        keep = np.sort(order[:max(1, int(len(uni) * frac))])
        got = recall_of(fs, split, uids, hists, idx, Q, keep)
        # How much of the click mass the tier can even reach. Without this a tier
        # that raises recall looks like free quality; the ceiling says whether the
        # gain is the popularity prior working or the metric being restricted.
        ceiling = float(counts[keep].sum() / max(counts[uni].sum(), 1))
        tier_rows.append({"tier_share": frac, "tier_docs": int(keep.size),
                          "click_mass_in_tier": ceiling,
                          **{f"recall@{k}": got[f"recall@{k}"] for k in KS},
                          **{f"delta@{k}": got[f"recall@{k}"] - full[f"recall@{k}"]
                             for k in KS}})
    out["tiering"] = {"full": {f"recall@{k}": full[f"recall@{k}"] for k in KS},
                      "universe_docs": int(len(uni)), "rows": tier_rows}


def recall_of(fs: FeatureStore, split: str, uids: np.ndarray, hists: list[np.ndarray],
              idx: BM25Index, Q: sp.csr_matrix, universe: np.ndarray) -> dict:
    """recall@K for a given candidate universe, over the sampled users only."""
    _, top = idx.search_sparse(Q, top_k=max(KS), universe=universe)
    keep = set(np.asarray(uids).tolist())
    imp = (fs.impressions(split).select("user_idx", "clicked")
           .filter(pl.col("user_idx").is_in(pl.Series(sorted(keep), dtype=pl.UInt32).implode()))
           .filter(pl.col("clicked").list.len() > 0).collect())
    row_of = {int(u): i for i, u in enumerate(np.asarray(uids).tolist())}
    acc = {k: [] for k in KS}
    for u, clicked in imp.iter_rows():
        i = row_of.get(int(u))
        cl = set(np.asarray(clicked, dtype=np.int64).tolist())
        for k in KS:
            hit = 0 if i is None else len(cl & set(top[i, :k].tolist()))
            acc[k].append(hit / len(cl))
    return {f"recall@{k}": float(np.mean(acc[k])) if acc[k] else 0.0 for k in KS}


# ==================================================================== part F

def part_f(fs: FeatureStore, out: dict, tmp: Path, blocks: list[int]) -> None:
    """The build: single-pass in-RAM against SPIMI's spill-and-merge."""
    texts = fs.texts()
    lang = fs.lang()

    with RSSWatcher() as w:
        t0 = time.perf_counter()
        idx = BM25Index.build(texts, lang=lang)
        t_inram = time.perf_counter() - t0
    inram = {"strategy": "single-pass in RAM (shipped)", "seconds": t_inram,
             "peak_rss_delta_mb": w.delta_mb, "postings": int(idx.doc_ids.size),
             "vocab": idx.vocab_size, "spill_mb": 0.0}
    del idx

    rows = [inram]
    for block in blocks:
        spill = tmp / f"spimi_{block}"
        if spill.exists():
            shutil.rmtree(spill)
        spill.mkdir(parents=True, exist_ok=True)
        with RSSWatcher() as w:
            t0 = time.perf_counter()
            # SPIMI: hash each block's tokens into an in-memory dictionary, sort it,
            # write one immutable run, forget it. Memory is bounded by the block, not
            # by the corpus -- which is L3's LSM argument arriving one lecture later
            # with postings as the payload.
            runs, n_post = [], 0
            for b0 in range(0, len(texts), block):
                toks = tokenize(texts[b0:b0 + block], lang)
                # The shipped build stems the vocabulary, so SPIMI must too --
                # without this the two produce different postings counts (305,340
                # vs 298,886 on EB-NeRD small) and the comparison is between two
                # different indexes rather than two ways of building one. Stemming
                # is per-token and deterministic, so doing it per block is exactly
                # equivalent to doing it once globally.
                toks = (toks.join(lexical._stem_vocabulary(toks["token"], lang),
                                  on="token", how="left")
                        .select("doc_id", pl.col("stem").alias("token")))
                agg = (toks.with_columns((pl.col("doc_id") + b0).alias("doc_id"))
                       .group_by("token", "doc_id").agg(pl.len().alias("tf"))
                       .sort("token", "doc_id"))
                f = spill / f"run_{len(runs)}.parquet"
                agg.write_parquet(f, compression="zstd")
                runs.append(f)
                n_post += agg.height
                del toks, agg
            # k-way merge: the reduce side. Reduce keys arrive sorted, so the
            # dictionary comes out sorted for free -- the slide's point about why
            # MapReduce fit this job.
            merged = (pl.concat([pl.scan_parquet(f) for f in runs])
                      .group_by("token", "doc_id").agg(pl.col("tf").sum())
                      .sort("token", "doc_id").collect(engine="streaming"))
            t_spimi = time.perf_counter() - t0
        spill_mb = sum(f.stat().st_size for f in runs) / 1e6
        rows.append({"strategy": f"SPIMI, {block:,}-doc blocks", "seconds": t_spimi,
                     "peak_rss_delta_mb": w.delta_mb, "postings": int(merged.height),
                     "vocab": int(merged["token"].n_unique()), "runs": len(runs),
                     "spill_mb": spill_mb})
        del merged
        shutil.rmtree(spill, ignore_errors=True)
    out["build"] = {"n_docs": len(texts), "rows": rows}

    # the slide's distributed arithmetic, at the rate we actually achieve
    rate = inram["postings"] / max(inram["seconds"], 1e-9)
    out["build"]["scale"] = {
        "postings_per_s": rate,
        "web_postings": 1e12,
        "single_node_days": 1e12 / rate / 86400,
        "nodes_for_30_min": 1e12 / rate / 1800,
        "note": "no shuffle, straggler or coordination cost -- the slide's 12-200x, not 1000x",
    }


# ==================================================================== main

def main() -> None:
    p = argparse.ArgumentParser(description="L5 ablation: postings, compression, top-k")
    p.add_argument("--dataset", default="ebnerd")
    p.add_argument("--variant", default="large")
    p.add_argument("--split", default="test")
    p.add_argument("--parts", default="abcdefg")
    p.add_argument("--universe-days", type=int, default=7)
    p.add_argument("--users", type=int, default=2000)
    p.add_argument("--queries", type=int, default=60)
    p.add_argument("--lengths", default="1,2,5,10,25,100,0")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--pairs", type=int, default=25)
    p.add_argument("--cache-fracs", default="0.001,0.01,0.05,0.2,0.5")
    p.add_argument("--tiers", default="0.05,0.1,0.25,0.5")
    p.add_argument("--blocks", default="20000,5000")
    p.add_argument("--scope-days", default="7,30,90,0")
    p.add_argument("--scope-queries", type=int, default=12)
    p.add_argument("--scope-max-iters", type=int, default=400_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tmp", default=os.environ.get("L5_TMP", "/tmp/l5_postings"))
    p.add_argument("--out", default="reports/l5")
    a = p.parse_args()

    fs = FeatureStore(a.dataset, a.variant)
    tmp = Path(a.tmp)
    res: dict = {"dataset": a.dataset, "variant": a.variant, "split": a.split,
                 "n_articles": fs.n_articles, "lang": fs.lang(), "parts": a.parts,
                 "seed": a.seed}
    print(f"L5 postings harness | {a.dataset}/{a.variant} | {fs.n_articles:,} articles"
          f" | lang={fs.lang()}", flush=True)

    idx = BM25Index.build(fs.texts(), lang=fs.lang())
    universe = candidate_universe_for_split(fs, a.split, a.universe_days)
    uids, hists = user_histories(fs, a.split)
    if a.users and len(uids) > a.users:
        pick = np.random.default_rng(a.seed).choice(len(uids), size=a.users, replace=False)
        uids, hists = uids[pick], [hists[i] for i in pick]
    Q = idx.queries_from_history(hists)
    # Two views on purpose. Compression, posting layout and skip lists are
    # properties of the index as built, so they are measured over every document;
    # top-k processing and tiering are properties of a *query*, which only ever
    # sees the live candidate universe.
    post_full = Postings(idx, None)
    post = Postings(idx, universe)
    print(f"  index: {idx.vocab_size:,} terms, {post_full.n_postings:,} postings | "
          f"universe {post.n_docs:,} docs, {post.n_postings:,} postings in scope | "
          f"{Q.shape[0]:,} queries", flush=True)

    if "a" in a.parts:
        print("\n== A. what is actually in a posting", flush=True)
        part_a(fs, idx, res)
        an = res["anatomy"]
        print(f"  {an['postings']:,} postings, {an['token_occurrences']:,} occurrences, "
              f"mean tf {an['mean_tf']:.3f}, tf==1 for {an['tf_eq_1_share']:.1%}, "
              f"avgdl {an['avgdl']:.1f}", flush=True)
        for l in an["layouts"]:
            print(f"    {l['layout']:<28} {l['bytes'] / 1e6:8.2f} MB  "
                  f"x{l['multiplier']:.2f}   slide "
                  + (f"x{l['slide']}" if l['slide'] else "--"), flush=True)

    if "b" in a.parts:
        print("\n== B. compression: bits per gap, decode rate, and the thesis", flush=True)
        # the same corpus indexed with the stopwords left in, purely to price what
        # the analyzer costs the compressor
        saved = lexical.STOPWORDS
        lexical.STOPWORDS = {k: set() for k in saved}
        try:
            stop_gaps, _ = Postings(BM25Index.build(fs.texts(), lang=fs.lang()), None).gaps()
        finally:
            lexical.STOPWORDS = saved
        part_b(post_full, res, tmp, a.repeats, stop_gaps)
        cd = res["codes"]
        print(f"  {cd['postings']:,} gaps | mean {cd['mean_gap']:.1f} "
              f"median {cd['median_gap']:.0f} p99 {cd['p99_gap']:.0f}", flush=True)
        for row in cd["rows"]:
            thr = (f"{row['decode_mpostings_per_s']:8.1f} M/s" if "decode_seconds" in row
                   else "        --   ")
            print(f"    {row['code']:<28} {row['bytes'] / 1e6:8.2f} MB  "
                  f"{row['bits_per_gap']:6.2f} bits/gap (slide {row['slide']})  "
                  f"{row['ratio_vs_raw']:5.2f}x  {thr}", flush=True)
        for row in res["bandwidth_thesis"]:
            print(f"    {row['code']:<12} {row['file_mb']:7.2f} MB  cold "
                  f"{row['cold_seconds'] * 1e3:8.2f} ms ({row['cold_mb_per_s']:7.1f} MB/s)  "
                  f"warm {row['warm_seconds'] * 1e3:8.2f} ms", flush=True)
        ae = res["codes"].get("analyzer_effect")
        if ae:
            w_, o_ = ae["with_stopwords"], ae["without_stopwords"]
            print(f"    analyzer: stopwords IN  {w_['postings']:>9,} postings  median gap "
                  f"{w_['median_gap']:>6.0f}  {w_['vbyte_bits_per_gap']:5.2f} bits/gap  "
                  f"{w_['vbyte_mb']:6.3f} MB", flush=True)
            print(f"              stopwords OUT {o_['postings']:>9,} postings  median gap "
                  f"{o_['median_gap']:>6.0f}  {o_['vbyte_bits_per_gap']:5.2f} bits/gap  "
                  f"{o_['vbyte_mb']:6.3f} MB", flush=True)
        for m in res["thesis_model"]:
            print(f"    {m['code']:<12} saves {m['mb_saved']:6.2f} MB = "
                  f"{m['io_saved_ms_at_slide_ssd']:6.2f} ms of slide-SSD I/O, costs "
                  f"{m['measured_decode_ms']:7.2f} ms of decode | needs "
                  f"{m['required_decode_gb_per_s']:6.2f} GB/s, has "
                  f"{m['measured_decode_gb_per_s']:5.2f} GB/s", flush=True)

    if "c" in a.parts:
        print("\n== C. top-k processing: TAAT / DAAT / WAND / matmul", flush=True)
        part_c(idx, post, Q, res, a.queries,
               [int(x) for x in a.lengths.split(",")], a.seed)
        qp = res["query_processing"]
        print(f"  median full-length query: {qp['median_full_query_terms']:,} terms "
              f"over {qp['n_docs']:,} docs, top-{qp['top_k']}", flush=True)
        for row in qp["rows"]:
            print(f"  terms={row['query_terms']:>5} (median {row['median_terms']:>5,})  "
                  f"docs scored: DAAT {row['daat']['docs_scored']:>9,.0f}  "
                  f"WAND {row['wand']['docs_scored']:>9,.0f}  "
                  f"(pruned {row['wand_prune_share']:6.1%})  | WAND/DAAT postings "
                  f"{row['wand_postings_share']:5.2f} contrib "
                  f"{row['wand_contribution_share']:5.2f}  | rank-safe "
                  f"D {row['rank_safe']['daat']:.2f} W {row['rank_safe']['wand']:.2f}",
                  flush=True)
            print(f"        postings touched: taat/matmul "
                  f"{row['matmul_postings_touched']:>9,.0f}  wand contributions "
                  f"{row['wand']['contributions']:>9,.0f} "
                  f"({row['wand']['contributions'] / max(row['matmul_postings_touched'], 1):.2f}x)",
                  flush=True)
            print(f"        ms/query: taat {row['taat']['ms_per_query']:8.2f}  "
                  f"daat {row['daat']['ms_per_query']:8.2f}  "
                  f"wand {row['wand']['ms_per_query']:8.2f}  "
                  f"matmul {row['matmul_ms_per_query']:8.2f}  | accumulator "
                  f"taat {row['taat']['accumulator_bytes'] / 1e3:.0f} KB vs "
                  f"daat {row['daat']['accumulator_bytes'] / 1e3:.1f} KB", flush=True)

    if "d" in a.parts:
        print("\n== D. skip lists and cheapest-first", flush=True)
        part_d(post_full, res, a.seed, a.pairs)
        if "pairs" in res.get("skips", {}):
            sk = res["skips"]
            print(f"  {len(sk['pairs'])} long/short intersections, mean speedup "
                  f"{sk['mean_speedup']:.2f}x", flush=True)
            for r in sk["pairs"][:4]:
                print(f"    df {r['long_df']:>8,} ^ {r['short_df']:>4,}  stride "
                      f"{r['skip_stride']:>4}  steps {r['naive_steps']:>9,} -> "
                      f"{r['skip_steps']:>8,} ({r['speedup']:5.2f}x)  "
                      f"skip index {r['skip_index_bytes'] / 1e3:.1f} KB", flush=True)
            co = res["conjunction_order"]
            print(f"  4-term conjunctions ({co['trials']} trials): ascending df "
                  f"{co['ascending_mean_steps']:,.0f} steps vs descending "
                  f"{co['descending_mean_steps']:,.0f} ({co['saving']:+.1%})", flush=True)

    if "e" in a.parts:
        print("\n== E. caching and tiering", flush=True)
        part_e(fs, idx, post, post_full, Q, uids, hists, a.split, universe,
               res, a.seed,
               [float(x) for x in a.cache_fracs.split(",")],
               [float(x) for x in a.tiers.split(",")])
        ca = res["cache"]
        print(f"  {ca['requests']:,} term requests over {ca['distinct_terms']:,} distinct "
              f"terms, index {ca['index_mb']:.1f} MB", flush=True)
        for r in ca["rows"]:
            print(f"    cache {r['budget_share']:>6.1%} ({r['budget_mb']:7.2f} MB, "
                  f"{r['terms_held']:>7,} terms)  request hits {r['request_hit_rate']:6.1%}  "
                  f"byte hits {r['byte_hit_rate']:6.1%}", flush=True)
        ti = res["tiering"]
        print(f"  full universe r@100 {ti['full']['recall@100']:.5f}", flush=True)
        for r in ti["rows"]:
            print(f"    tier {r['tier_share']:>5.0%} ({r['tier_docs']:>7,} docs, "
                  f"{r['click_mass_in_tier']:5.1%} of clicks)  "
                  + "  ".join(f"r@{k} {r[f'recall@{k}']:.5f} ({r[f'delta@{k}']:+.5f})"
                              for k in KS), flush=True)

    if "g" in a.parts:
        print("\n== G. does the matmul stay right as the universe grows?", flush=True)
        part_g(fs, idx, Q, res, a.split,
               [int(x) for x in a.scope_days.split(",")], a.scope_queries, a.seed,
               a.scope_max_iters)


    if "f" in a.parts:
        print("\n== F. building the index: in-RAM vs SPIMI", flush=True)
        part_f(fs, res, tmp, [int(x) for x in a.blocks.split(",")])
        for r in res["build"]["rows"]:
            print(f"  {r['strategy']:<32} {r['seconds']:7.2f}s  peak RSS "
                  f"+{r['peak_rss_delta_mb']:8.1f} MB  spill {r['spill_mb']:7.2f} MB  "
                  f"postings {r['postings']:,}", flush=True)
        sc = res["build"]["scale"]
        print(f"  at {sc['postings_per_s']:,.0f} postings/s: 10^12 postings is "
              f"{sc['single_node_days']:.1f} single-node days, or "
              f"{sc['nodes_for_30_min']:,.0f} nodes for 30 minutes ({sc['note']})", flush=True)

    outdir = Path(a.out)
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"l5_postings_{a.dataset}_{a.variant}.json"
    # Merge rather than overwrite, so re-running one part to fix or extend it does
    # not silently delete the five that were not asked for.
    if path.exists():
        prev = json.loads(path.read_text())
        prev.update(res)
        prev["parts"] = "".join(sorted(set(prev.get("parts", "")) | set(a.parts)))
        res = prev
    path.write_text(json.dumps(res, indent=2, default=float))
    print(f"\nwrote {path}", flush=True)
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
