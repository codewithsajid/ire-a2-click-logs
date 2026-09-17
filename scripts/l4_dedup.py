"""L4 ablation: corpus laws, near-duplicate detection, and the sketch family.

L4's thesis is that you cannot afford to look at everything twice, so you
sketch: shared randomness plus a small deterministic summary plus a provable
error bound. Almost none of that is in this repo. The analyzer sweep in Q2 is
the one L4 idea already measured; Zipf and Heaps are quoted nowhere, the corpus
is never deduplicated, and the one place a Bloom filter obviously belongs -- the
"articles this user has already read" set that `drop_seen` maintains exactly --
uses a Python set per user.

Every part below tests a slide's *number*, not its story. The slides make five
falsifiable quantitative claims and this harness checks all five:

  1. Zipf: the rank-1 term is 6-7% of tokens, the top 10 are ~25% of postings,
     ~half the vocabulary appears once, and the log-log slope is ~ -1.
  2. Heaps: V = k*N^beta with k ~ 30, beta ~ 0.5 -- which is a *predictor*, so it
     is fitted on the small variant and used to predict the large one.
  3. MinHash: the standard error of J-hat is sqrt(J(1-J)/k) -- the 1/sqrt(k) law.
  4. Banding: P(candidate) = 1 - (1 - s^r)^b with threshold t ~ (1/b)^(1/r).
  5. Bloom: FP ~ 0.6185^(m/n), each 5 bits/key buying 10x fewer false positives.

Six parts:

A. **Corpus laws.** Zipf and Heaps on the raw token stream *and* on the stream
   the shipped analyzer produces, because the analyzer removes exactly the head
   of the Zipf curve and that is the whole reason stopword removal pays.

B. **Duplicates.** SHA-256 exact dedup first (the slide's "~free" step), then the
   shingle-size dial with exact Jaccard as ground truth, then MinHash accuracy
   against the 1/sqrt(k) law, then SimHash on the same pairs at 8 bytes a document.

C. **Banding and the O(n^2) wall.** The measured S-curve against the formula, the
   candidate-pair count against n(n-1)/2, and the wall-clock of all-pairs against
   sketch-band-verify -- extrapolated to the 10^9 documents the slide talks about.

D. **What dedup actually buys.** Near-dup clusters in the real article catalogue,
   how much click mass is split across copies, and what collapsing clusters to a
   canonical article does to recall@K. This is the L1/L4 claim ("dupes corrupt
   ranking signals -- clicks split across copies") priced in the metric.

E. **Sketches priced in the metric.** A Bloom filter for the seen-set, swept over
   bits/key, with the false-positive rate checked against the formula *and* the
   recall it costs -- because a Bloom false positive here is not an abstract
   error, it drops an article the user never saw. Then Count-Min vs Count-Sketch
   on the corpus's own Zipf term stream and on a flattened version of it, since
   the slide claims the ranking between them flips with skew. Then HyperLogLog
   for distinct users per day against 1.04/sqrt(m).

F. **Field weighting and extraction economics.** Q2 tested field *inclusion* and
   never field *weight*, so the BM25F preview the slide promises is unmeasured;
   and the LLM-vs-regex table is arithmetic we can do on our own token counts.

Nothing here needs a GPU. Parts A and D want the large bundle -- A because a
held-out Heaps prediction needs two corpus sizes, D because near-duplicate
stories only exist in a catalogue big enough to have republished any.
"""
from __future__ import annotations

import os

# Sized at import, so it has to be set before numpy loads -- see l2_serving.py.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import polars as pl
import scipy.sparse as sp

from newsrec.lexical import BM25Index, tokenize
from newsrec.retrieval import candidate_universe_for_split, user_histories
from newsrec.store import FeatureStore

KS = (50, 100, 200)
U64 = np.uint64


def u64(x: int) -> np.uint64:
    """A 64-bit constant, wrapped. numpy refuses to build a uint64 from a Python
    int above 2^63, which is where every good mixing constant lives."""
    return U64(x & 0xFFFFFFFFFFFFFFFF)


# ------------------------------------------------------------------ hashing
#
# Everything below hashes *integer shingle ids*, not strings. Interning the
# shingles once and hashing the integers is the slide's own argument in miniature:
# storing explicit permutations of a 10^10-shingle universe costs 8 TB, while
# hash(i, x) computed on the fly at only the |S| shingles a document has costs
# 100 seeds ~ 1 KB. Here it also turns every hash into a vectorised numpy
# expression instead of a Python loop over 17 million shingles.

def splitmix64(x: np.ndarray) -> np.ndarray:
    """A strong 64-bit mixer, vectorised.

    Used wherever a single well-distributed 64-bit hash of an id is needed
    (SimHash bits, Bloom probes, Count-Min rows, HyperLogLog). Shingle ids are
    dense and sequential, so they must be mixed before any bit of them is used as
    a bucket index: `id % w` directly would make a sketch's error a function of
    insertion order rather than of the stream.
    """
    z = x.astype(U64) + u64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> U64(30))) * u64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> U64(27))) * u64(0x94D049BB133111EB)
    return z ^ (z >> U64(31))


def hash_family(k: int, seed: int = 0) -> np.ndarray:
    """k seeds standing in for k permutations: h_i(x) = splitmix64(x ^ seed_i).

    This is the slide's construction verbatim -- "each pi_i is generated, not
    stored" -- and splitmix64 is a bijection on 64 bits, so XOR-ing a distinct
    seed in and mixing genuinely permutes the shingle universe.

    The obvious textbook alternative, h(x) = (a*x + b) mod (2^61 - 1), was tried
    first and is a trap at this scale. Keeping a*x inside 64 bits forces a < 2^31,
    and with shingle ids below 2^20 the product never reaches the modulus: the map
    is monotone in x, so every one of the k "permutations" returns the *same*
    shingle -- the one with the smallest id. The estimator then collapses to "do
    these two documents share their lowest-id shingle", which is 0 or 1 rather
    than a Jaccard estimate. It reported sd 0.49 where the law predicts 0.07, and
    gave byte-identical results for k = 16, 64 and 256, which is what exposed it.
    """
    return np.random.default_rng(seed).integers(
        0, 1 << 62, size=k, dtype=np.int64).astype(U64) * U64(2) + U64(1)


# ----------------------------------------------------------------- shingling

def shingle_docs(texts: list[str], k: int, unit: str = "word"
                 ) -> tuple[np.ndarray, np.ndarray, int]:
    """Documents -> (flat shingle ids, row offsets, vocabulary size).

    CSR-shaped on purpose: every downstream sketch is a reduction over a
    document's run of ids, which numpy does with `reduceat` and Python does with
    125,541 loop iterations.

    `unit="char"` is the slide's other option (9 characters) and matters here more
    than it does on the web. An EB-NeRD article is a title plus a subtitle, about
    25 words, so word 9-shingles leave a document with ~17 shingles and one edited
    word destroys nine of them -- the failure mode the slide warns about, at a
    corpus where it actually bites.
    """
    vocab: dict[str, int] = {}
    ids: list[np.ndarray] = []
    offs = np.zeros(len(texts) + 1, dtype=np.int64)
    get = vocab.setdefault
    for i, t in enumerate(texts):
        t = (t or "").lower()
        if unit == "word":
            toks = t.split()
            grams = {" ".join(toks[j:j + k]) for j in range(max(0, len(toks) - k + 1))}
        else:
            s = " ".join(t.split())
            grams = {s[j:j + k] for j in range(max(0, len(s) - k + 1))}
        row = np.fromiter((get(g, len(vocab)) for g in grams),
                          dtype=np.int64, count=len(grams))
        ids.append(row)
        offs[i + 1] = offs[i] + row.size
    flat = np.concatenate(ids) if ids else np.zeros(0, dtype=np.int64)
    return flat, offs, len(vocab)


def as_csr(flat: np.ndarray, offs: np.ndarray, n_vocab: int) -> sp.csr_matrix:
    """Binary document-by-shingle matrix -- the characteristic matrix, transposed.

    Only used for exact Jaccard: X @ X.T gives every pairwise intersection in one
    sparse product, which is the O(n^2) this whole lecture exists to avoid and is
    exactly why it never leaves the ground-truth sample.
    """
    return sp.csr_matrix((np.ones(flat.size, dtype=np.float32), flat, offs),
                         shape=(offs.size - 1, max(n_vocab, 1)))


def exact_jaccard_pairs(X: sp.csr_matrix) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Every pair sharing at least one shingle, with its exact Jaccard.

    Pairs sharing nothing are not returned: there are O(n^2) of them and their
    Jaccard is 0 by construction. Callers that need the count use n(n-1)/2 minus
    the length of this result.
    """
    sizes = np.asarray(X.sum(axis=1)).ravel()
    inter = sp.triu(X @ X.T, k=1).tocoo()
    r, c, v = inter.row.astype(np.int64), inter.col.astype(np.int64), inter.data
    union = sizes[r] + sizes[c] - v
    j = np.where(union > 0, v / np.maximum(union, 1e-9), 0.0)
    order = np.argsort(r * X.shape[0] + c, kind="stable")
    return r[order], c[order], j[order]


# ------------------------------------------------------------------ minhash

def minhash(flat: np.ndarray, offs: np.ndarray, k: int, seed: int = 0,
            chunk_cells: int = 32_000_000) -> np.ndarray:
    """k-dimensional MinHash signatures, (n_docs, k) uint64.

    Chunked over the flat shingle array rather than looped over documents: the
    whole (nnz x k) hash matrix for the EB-NeRD catalogue at k=128 is 18 GB, so it
    is materialised a fixed number of *cells* at a time -- the block shrinks as k
    grows -- and folded into the running minimum with `np.minimum.reduceat`. Empty
    documents keep a sentinel signature, which can never collide with anything
    real: the honest answer for a document with no shingles, and better than
    letting every empty document collide with every other.
    """
    seeds = hash_family(k, seed)
    n = offs.size - 1
    sig = np.full((n, k), np.iinfo(np.uint64).max, dtype=U64)
    lengths = offs[1:] - offs[:-1]
    chunk = max(1, chunk_cells // max(k, 1))          # cap the (nnz x k) block, not nnz
    start = 0
    while start < n:
        base = offs[start]
        end = int(np.searchsorted(offs[start + 1:], base + chunk, side="right")) + start
        end = min(max(end, start + 1), n)
        lo = offs[start]
        ids = flat[lo:offs[end]]
        if ids.size:
            H = splitmix64(ids[:, None].astype(U64) ^ seeds[None, :])
            local = offs[start:end] - lo
            ne = lengths[start:end] > 0
            if ne.any():
                sig[np.arange(start, end)[ne]] = np.minimum.reduceat(H, local[ne], axis=0)
        start = end
    return sig


def band_buckets(sig: np.ndarray, b: int, r: int, seed: int = 1) -> np.ndarray:
    """(n_docs, b) bucket ids -- one per band, hashed from r signature rows."""
    n, k = sig.shape
    assert b * r <= k, f"b*r={b * r} exceeds signature length {k}"
    out = np.zeros((n, b), dtype=U64)
    rng = np.random.default_rng(seed)
    coef = rng.integers(1, 1 << 62, size=r, dtype=np.int64).astype(U64)
    for i in range(b):
        block = sig[:, i * r:(i + 1) * r]
        acc = np.full(n, u64(i * 0x9E3779B97F4A7C15), dtype=U64)
        for j in range(r):
            acc = splitmix64(acc ^ (block[:, j] * coef[j]))
        out[:, i] = acc
    return out


def candidate_pairs(buckets: np.ndarray, cap_bucket: int = 2048
                    ) -> tuple[np.ndarray, int, int]:
    """Distinct (i, j) pairs sharing at least one band bucket, as a sorted key array.

    Returned as int64 keys i*n + j rather than a Python set of tuples: on the full
    catalogue this is millions of pairs, and a set of tuples costs ~100 bytes each.

    `cap_bucket` guards the degenerate case the slide never mentions. A bucket
    holding q documents emits q(q-1)/2 pairs, so one giant bucket -- every empty or
    boilerplate document colliding -- reintroduces exactly the quadratic term the
    banding was supposed to remove. Oversized buckets are dropped and *counted*;
    truncating them silently would understate recall without saying so.
    """
    n, nb = buckets.shape
    chunks = []
    dropped = dropped_docs = 0
    for i in range(nb):
        order = np.argsort(buckets[:, i], kind="stable")
        vals = buckets[order, i]
        edges = np.flatnonzero(np.r_[True, vals[1:] != vals[:-1], True])
        starts, ends = edges[:-1], edges[1:]
        sizes = ends - starts
        for s, e, m in zip(starts[sizes >= 2], ends[sizes >= 2], sizes[sizes >= 2]):
            if m > cap_bucket:
                dropped += 1
                dropped_docs += int(m)
                continue
            grp = np.sort(order[s:e])
            ii, jj = np.triu_indices(int(m), k=1)
            chunks.append(grp[ii] * n + grp[jj])
    if not chunks:
        return np.zeros(0, dtype=np.int64), dropped, dropped_docs
    return np.unique(np.concatenate(chunks)), dropped, dropped_docs


def simhash(flat: np.ndarray, offs: np.ndarray, seed: int = 7) -> np.ndarray:
    """64-bit SimHash per document -- one uint64, against MinHash's k*8 bytes.

    Accumulated one bit-plane at a time: the (nnz x 64) sign matrix would be 9 GB
    on the full catalogue, and 64 `reduceat` passes over an int8 column are both
    smaller and faster than one pass over a matrix that does not fit.
    """
    h = splitmix64(flat.astype(U64) ^ u64(seed * 0x9E3779B97F4A7C15))
    n = offs.size - 1
    ne = (offs[1:] - offs[:-1]) > 0
    out = np.zeros(n, dtype=U64)
    if not ne.any():
        return out
    starts = offs[:-1][ne]
    packed = np.zeros(starts.size, dtype=U64)
    for j in range(64):
        col = (((h >> U64(j)) & U64(1)).astype(np.int8) * 2 - 1).astype(np.int32)
        acc = np.add.reduceat(col, starts)
        packed |= (acc > 0).astype(U64) << U64(j)
    out[ne] = packed
    return out


def hamming(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    x = (a ^ b).copy()
    c = np.zeros(x.shape, dtype=np.int32)
    while x.any():
        c += (x & U64(1)).astype(np.int32)
        x >>= U64(1)
    return c


# ------------------------------------------------------------- corpus laws

def raw_tokens(texts: list[str]) -> pl.DataFrame:
    """The token stream *before* the analyzer -- what Zipf and Heaps describe."""
    return (pl.DataFrame({"text": texts}).with_row_index("doc_id")
            .with_columns(pl.col("text").fill_null("").str.to_lowercase()
                          .str.replace_all(r"[^\w\såæøÅÆØ]", " ")
                          .str.split(" ").alias("token"))
            .explode("token").filter(pl.col("token").str.len_chars() > 0)
            .select("doc_id", "token"))


def heaps_fit(toks: pl.DataFrame, seed: int = 0, points: int = 18) -> dict:
    """Fit V = k*N^beta by growing the corpus one document at a time.

    Documents, not tokens: the vocabulary of a corpus is a property of documents
    arriving. Sampling tokens uniformly from the finished corpus would already
    know the whole vocabulary exists and would bend the curve flat.
    """
    by_doc = toks.group_by("doc_id").agg(pl.col("token"))["token"].to_list()
    order = np.random.default_rng(seed).permutation(len(by_doc))
    by_doc = [by_doc[i] for i in order]
    stops = np.unique(np.round(np.logspace(
        np.log10(max(20, len(by_doc) // 5000)), np.log10(max(len(by_doc), 21)),
        points)).astype(int))
    seen: set[str] = set()
    n_tok, cursor, curve = 0, 0, []
    for stop in stops:
        for d in by_doc[cursor:stop]:
            seen.update(d)
            n_tok += len(d)
        cursor = int(stop)
        curve.append([int(n_tok), int(len(seen))])
    N = np.array([c[0] for c in curve], dtype=np.float64)
    V = np.array([c[1] for c in curve], dtype=np.float64)
    beta, logk = np.polyfit(np.log10(N), np.log10(V), 1)
    return {"curve": curve, "beta": float(beta), "k": float(10 ** logk),
            "n_tokens": int(N[-1]), "vocab": int(V[-1])}


# ================================================================== part A

def part_a(fs: FeatureStore, fs_small: FeatureStore | None, out: dict, seed: int) -> None:
    """Zipf and Heaps -- the two laws that price the index before it is built."""
    texts = fs.texts()
    lang = fs.lang()
    raw = raw_tokens(texts)
    ana = tokenize(texts, lang)

    rows = []
    for name, toks in (("raw", raw), ("analyzed", ana)):
        f = toks.group_by("token").agg(pl.len().alias("cf")).sort("cf", descending=True)
        cf = f["cf"].to_numpy().astype(np.float64)
        total = cf.sum()
        dfr = (toks.select("doc_id", "token").unique()
               .group_by("token").agg(pl.len().alias("df"))
               .sort("df", descending=True))
        dfv = dfr["df"].to_numpy()
        head = int(min(10_000, cf.size))
        r = np.arange(1, head + 1, dtype=np.float64)
        slope, intercept = np.polyfit(np.log10(r), np.log10(cf[:head]), 1)
        rows.append({
            "stream": name,
            "tokens": int(total), "vocab": int(cf.size),
            "top1_term": f["token"][0],
            "top1_share": float(cf[0] / total),
            "top10_share": float(cf[:10].sum() / total),
            "hapax_share": float((cf == 1).sum() / cf.size),
            "zipf_slope": float(slope), "zipf_intercept": float(intercept),
            "postings": int(dfv.sum()),
            "top10_postings_share": float(dfv[:10].sum() / dfv.sum()),
            "rank_curve": [[int(x), float(cf[x - 1])] for x in np.unique(
                np.round(np.logspace(0, np.log10(cf.size), 60)).astype(int))],
        })
    out["zipf"] = rows

    out["heaps"] = heaps_fit(raw, seed=seed)

    # A fit is only a law if it predicts. Fit both parameters on the *small*
    # variant and predict the large corpus's vocabulary at its own token count --
    # the only genuinely held-out point available on one machine.
    if fs_small is not None:
        small = heaps_fit(raw_tokens(fs_small.texts()), seed=seed)
        pred = small["k"] * (out["heaps"]["n_tokens"] ** small["beta"])
        out["heaps"]["holdout"] = {
            "fit_on": f"{fs_small.dataset}/{fs_small.variant}",
            "small_k": small["k"], "small_beta": small["beta"],
            "small_tokens": small["n_tokens"], "small_vocab": small["vocab"],
            "predicted_large_vocab": float(pred),
            "actual_large_vocab": out["heaps"]["vocab"],
            "rel_error": float(pred / out["heaps"]["vocab"] - 1.0),
        }


# ================================================================== part B

def part_b(fs: FeatureStore, out: dict, sample: int, seed: int,
           shingle_specs: list[str], sig_ks: list[int]) -> None:
    """Exact dedup, the shingle dial, MinHash error, SimHash."""
    texts = fs.texts()

    # (1) SHA-256 -- the free step the slide insists on doing first
    t0 = time.perf_counter()
    digests: dict[bytes, list[int]] = defaultdict(list)
    for i, t in enumerate(texts):
        digests[hashlib.sha256((t or "").encode("utf-8")).digest()].append(i)
    t_sha = time.perf_counter() - t0
    groups = {k: v for k, v in digests.items() if len(v) > 1}
    out["exact"] = {
        "n_docs": len(texts), "distinct_digests": len(digests),
        "duplicate_groups": len(groups),
        "duplicate_docs": int(sum(len(v) for v in groups.values())),
        "redundant_docs": int(sum(len(v) - 1 for v in groups.values())),
        "largest_group": max((len(v) for v in groups.values()), default=0),
        "seconds": t_sha, "docs_per_s": len(texts) / max(t_sha, 1e-9),
        "example_texts": [texts[v[0]][:90] for v in list(groups.values())[:3]],
    }

    # (2) the shingle-size dial, on a sample small enough for exact all-pairs
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(texts), size=min(sample, len(texts)), replace=False))
    sub = [texts[i] for i in idx]
    n = len(sub)
    all_pairs = n * (n - 1) // 2

    dial = []
    for spec in shingle_specs:
        unit, kk = ("char", int(spec[1:])) if spec.startswith("c") else ("word", int(spec))
        flat, offs, nv = shingle_docs(sub, kk, unit)
        X = as_csr(flat, offs, nv)
        _, _, j = exact_jaccard_pairs(X)
        sizes = np.asarray(X.sum(axis=1)).ravel()
        # "everything looks similar at small k, nothing does at large k" is a claim
        # about the whole distribution, so report where its mass sits rather than
        # one summary number
        dial.append({
            "unit": unit, "k": kk,
            "mean_shingles_per_doc": float(sizes.mean()),
            "empty_docs": int((sizes == 0).sum()), "vocab": int(nv),
            "pairs_with_overlap": int(j.size), "pairs_total": all_pairs,
            "overlap_share": j.size / max(all_pairs, 1),
            "mean_j_nonzero": float(j.mean()) if j.size else 0.0,
            "pairs_j_ge_0.5": int((j >= 0.5).sum()),
            "pairs_j_ge_0.8": int((j >= 0.8).sum()),
            "p999_j": float(np.quantile(j, 0.999)) if j.size else 0.0,
        })
    out["shingle_dial"] = dial

    # (3) MinHash accuracy against sqrt(J(1-J)/k) -- the central claim of the slide
    flat, offs, nv = shingle_docs(sub, 9, "char")
    X = as_csr(flat, offs, nv)
    r, c, jtrue = exact_jaccard_pairs(X)
    keep = jtrue >= 0.05
    r, c, jtrue = r[keep], c[keep], jtrue[keep]
    # stratified, so the error law is checked across J and not only where the
    # pairs happen to be dense (which is entirely at small J)
    bins = [(0.05, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)]
    take = [rng.choice(m, size=min(4000, m.size), replace=False)
            for lo, hi in bins
            if (m := np.flatnonzero((jtrue >= lo) & (jtrue < hi))).size]
    sel = np.concatenate(take) if take else np.zeros(0, dtype=np.int64)
    rr, cc, jj = r[sel], c[sel], jtrue[sel]

    acc = []
    for k in sig_ks:
        t0 = time.perf_counter()
        sig = minhash(flat, offs, k, seed=seed)
        t_sign = time.perf_counter() - t0
        est = (sig[rr] == sig[cc]).mean(axis=1)
        err = est - jj
        per_bin = []
        for lo, hi in bins:
            m = (jj >= lo) & (jj < hi)
            if m.sum() < 20:
                continue
            jbar = float(jj[m].mean())
            per_bin.append({"j_lo": lo, "j_hi": hi, "n": int(m.sum()), "j_mean": jbar,
                            "observed_sd": float(err[m].std()),
                            "predicted_sd": float(np.sqrt(jbar * (1 - jbar) / k)),
                            "bias": float(err[m].mean())})
        acc.append({"k": k, "sign_seconds": t_sign,
                    "docs_per_s": n / max(t_sign, 1e-9),
                    "bytes_per_doc": int(k * 8), "rmse": float(np.sqrt((err ** 2).mean())),
                    "bins": per_bin})
    out["minhash"] = {"unit": "char", "k_shingle": 9, "n_pairs": int(jj.size),
                      "n_docs": n, "accuracy": acc}

    # (4) SimHash: 8 bytes per document against MinHash's k*8, on the same pairs
    t0 = time.perf_counter()
    sh = simhash(flat, offs)
    t_sh = time.perf_counter() - t0
    d = hamming(sh[rr], sh[cc])
    corr = float(np.corrcoef(-d.astype(np.float64), jj)[0, 1]) if jj.size > 2 else float("nan")
    ops = []
    for thr in (3, 6, 10, 16):
        pred = d <= thr
        for jthr in (0.5, 0.8):
            tp = int((pred & (jj >= jthr)).sum())
            fp = int((pred & (jj < jthr)).sum())
            fn = int((~pred & (jj >= jthr)).sum())
            ops.append({"hamming_le": thr, "j_threshold": jthr,
                        "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1)})
    out["simhash"] = {"sign_seconds": t_sh, "bytes_per_doc": 8,
                      "corr_negdist_j": corr, "operating_points": ops}


# ================================================================== part C

def part_c(fs: FeatureStore, out: dict, sample: int, seed: int,
           bands: list[tuple[int, int]], sig_k: int) -> None:
    """Banding: the S-curve, the pair count, and the wall it removes."""
    texts = fs.texts()
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(texts), size=min(sample, len(texts)), replace=False))
    sub = [texts[i] for i in idx]
    n = len(sub)
    all_pairs = n * (n - 1) // 2

    flat, offs, nv = shingle_docs(sub, 9, "char")
    X = as_csr(flat, offs, nv)

    t0 = time.perf_counter()
    r, c, jtrue = exact_jaccard_pairs(X)
    t_brute = time.perf_counter() - t0
    keys_true = r * n + c                      # already sorted by exact_jaccard_pairs

    sig = minhash(flat, offs, sig_k, seed=seed)

    curves = []
    for b, rw in bands:
        t0 = time.perf_counter()
        bk = band_buckets(sig, b, rw, seed=seed + 3)
        cand, dropped, dropped_docs = candidate_pairs(bk)
        t_lsh = time.perf_counter() - t0

        # measured S-curve: of the pairs at true similarity s, what fraction became
        # candidates? Binned, because P(candidate) is a function of s and a single
        # aggregate number would hide the shape that is the entire argument.
        edges = np.linspace(0, 1, 21)
        obs, pred, cnt = [], [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (jtrue >= lo) & (jtrue < hi)
            if m.sum() < 5:
                obs.append(None); pred.append(None); cnt.append(int(m.sum()))
                continue
            km = keys_true[m]
            pos = np.searchsorted(cand, km)
            got = (pos < cand.size) & (cand[np.minimum(pos, cand.size - 1)] == km)
            s = float(jtrue[m].mean())
            obs.append(float(got.mean()))
            pred.append(float(1 - (1 - s ** rw) ** b))
            cnt.append(int(m.sum()))

        rows = []
        for thr in (0.5, 0.8, 0.9):
            m = jtrue >= thr
            km = keys_true[m]
            pos = np.searchsorted(cand, km)
            hit = int(((pos < cand.size) & (cand[np.minimum(pos, cand.size - 1)] == km)).sum())
            rows.append({"threshold": thr, "true_pairs": int(m.sum()), "found": hit,
                         "recall": hit / max(int(m.sum()), 1),
                         "precision": hit / max(cand.size, 1)})

        # candidates at J = 0 share no shingle at all, so they are pure false
        # positives of the sketch rather than of the threshold
        pos = np.searchsorted(keys_true, cand)
        known = (pos < keys_true.size) & (keys_true[np.minimum(pos, keys_true.size - 1)] == cand)
        curves.append({
            "b": b, "r": rw, "signature": sig_k,
            "theoretical_threshold": float((1.0 / b) ** (1.0 / rw)),
            "candidates": int(cand.size),
            "candidate_share_of_all_pairs": cand.size / max(all_pairs, 1),
            "oversized_buckets_dropped": dropped, "docs_in_dropped_buckets": dropped_docs,
            "candidates_with_zero_overlap": int((~known).sum()),
            "seconds": t_lsh,
            "s_curve": {"bin_lo": edges[:-1].tolist(), "observed": obs,
                        "predicted": pred, "n": cnt},
            "at_threshold": rows,
        })

    rate = all_pairs / max(t_brute, 1e-9)

    # Which banding to extrapolate from is a decision, not a minimum. Picking the
    # smallest candidate set picks the most timid curve -- b=10, r=12 emits one
    # pair on this sample and would report a 700,000x speedup for finding nothing.
    # The operating point is the cheapest banding that still meets a recall target
    # at the similarity we actually care about, which is the (b, r) choice the
    # slide says to make from a false-negative budget.
    def recall_at(curve: dict, thr: float) -> float:
        row = next((x for x in curve["at_threshold"] if x["threshold"] == thr), None)
        return row["recall"] if row and row["true_pairs"] else 0.0

    target, want = 0.5, 0.90
    ok = [c for c in curves if recall_at(c, target) >= want]
    best = (min(ok, key=lambda x: x["candidate_share_of_all_pairs"]) if ok else
            max(curves, key=lambda x: (recall_at(x, target),
                                       -x["candidate_share_of_all_pairs"])))
    out["banding"] = {
        "n_docs": n, "all_pairs": all_pairs, "brute_force_seconds": t_brute,
        "pairs_with_any_overlap": int(jtrue.size), "curves": curves,
        "wall": {
            "measured_pairs_per_s": rate,
            "brute_force_years_at_1e9_docs": (1e9 * (1e9 - 1) / 2) / rate / (365 * 86400),
            "operating_point_rule": f"cheapest banding with recall >= {want} at J >= {target}",
            "operating_point_met": bool(ok),
            "lsh_candidate_share": best["candidate_share_of_all_pairs"],
            "lsh_b": best["b"], "lsh_r": best["r"],
            "lsh_recall_at_target": recall_at(best, target),
            "lsh_verify_years_at_1e9_docs":
                (1e9 * (1e9 - 1) / 2 * best["candidate_share_of_all_pairs"])
                / rate / (365 * 86400),
        },
    }


# ================================================================== part D

def recall_under_canonical(fs: FeatureStore, split: str, user_ids: np.ndarray,
                           topk: np.ndarray, canonical: np.ndarray,
                           ks: tuple[int, ...] = KS) -> dict:
    """recall@K where both sides are collapsed onto canonical article ids.

    Dedup has to be scored this way or the comparison is rigged in both
    directions. Collapsing only the retrieved list would count a click on a copy
    as a miss and make dedup look catastrophic; collapsing only the clicks would
    give dedup credit for retrieving an article it removed. So the retrieved list
    is mapped, de-duplicated *keeping first occurrence* (a merged cluster occupies
    one slot, which is the entire point), truncated to K, and the clicked set is
    mapped by the same array. Passing `canonical = arange(n)` recovers the
    baseline through the identical code path.
    """
    keep = set(np.asarray(user_ids).tolist())
    imp = (fs.impressions(split).select("user_idx", "clicked")
           .filter(pl.col("user_idx").is_in(pl.Series(sorted(keep), dtype=pl.UInt32).implode()))
           .filter(pl.col("clicked").list.len() > 0).collect())
    row_of = {int(u): i for i, u in enumerate(np.asarray(user_ids).tolist())}
    kmax = max(ks)
    acc = {k: [] for k in ks}
    for u, clicked in imp.iter_rows():
        i = row_of.get(int(u))
        cl = canonical[np.asarray(clicked, dtype=np.int64)]
        cset = set(cl.tolist())
        if i is None:
            for k in ks:
                acc[k].append(0.0)
            continue
        mapped = canonical[topk[i]]
        _, first = np.unique(mapped, return_index=True)
        ordered = mapped[np.sort(first)][:kmax]
        for k in ks:
            hit = len(cset & set(ordered[:k].tolist()))
            acc[k].append(hit / len(cset))
    return {"n_impressions": len(acc[ks[0]]),
            **{f"recall@{k}": float(np.mean(acc[k])) if acc[k] else 0.0 for k in ks}}


def part_d(fs: FeatureStore, split: str, out: dict, seed: int, sig_k: int,
           b: int, r: int, thresholds: list[float], universe_days: int,
           n_users: int) -> None:
    """Near-dups in the real catalogue, and what they cost retrieval."""
    texts = fs.texts()
    lang = fs.lang()
    n = len(texts)

    t0 = time.perf_counter()
    flat, offs, nv = shingle_docs(texts, 9, "char")
    t_shingle = time.perf_counter() - t0
    t0 = time.perf_counter()
    sig = minhash(flat, offs, sig_k, seed=seed)
    t_sign = time.perf_counter() - t0
    t0 = time.perf_counter()
    bk = band_buckets(sig, b, r, seed=seed + 3)
    cand, dropped, dropped_docs = candidate_pairs(bk)
    t_band = time.perf_counter() - t0

    # Verify candidates exactly -- the slide's last step, and the only one that
    # decides anything. Verification is linear in the number of candidates, which
    # is the whole reason banding is allowed to be approximate.
    t0 = time.perf_counter()
    X = as_csr(flat, offs, nv)
    sizes = np.asarray(X.sum(axis=1)).ravel()
    ci, cj = cand // n, cand % n
    jv = np.zeros(cand.size, dtype=np.float64)
    B = 200_000
    for s in range(0, cand.size, B):
        e = min(s + B, cand.size)
        ii, jjx = ci[s:e], cj[s:e]
        inter = np.asarray(X[ii].multiply(X[jjx]).sum(axis=1)).ravel()
        union = sizes[ii] + sizes[jjx] - inter
        jv[s:e] = np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)
    t_verify = time.perf_counter() - t0

    out["catalogue"] = {
        "n_docs": n, "all_pairs": n * (n - 1) // 2,
        "shingle_seconds": t_shingle, "sign_seconds": t_sign,
        "band_seconds": t_band, "verify_seconds": t_verify,
        "signature_mb": sig.nbytes / 1e6, "shingle_mb": flat.nbytes / 1e6,
        "candidates": int(cand.size),
        "candidate_share": cand.size / max(n * (n - 1) // 2, 1),
        "oversized_buckets_dropped": dropped, "docs_in_dropped_buckets": dropped_docs,
        "b": b, "r": r, "signature": sig_k,
        "verified_ge_0.5": int((jv >= 0.5).sum()),
        "verified_ge_0.8": int((jv >= 0.8).sum()),
        "verified_ge_0.9": int((jv >= 0.9).sum()),
        "verified_precision_at_0.8": float((jv >= 0.8).mean()) if jv.size else 0.0,
    }

    clicks = (fs.impressions(split).select("clicked").collect()["clicked"]
              .explode().drop_nulls().to_numpy().astype(np.int64))
    click_count = np.bincount(clicks, minlength=n)

    universe = candidate_universe_for_split(fs, split, universe_days)
    uids, hists = user_histories(fs, split)
    if n_users and len(uids) > n_users:
        pick = np.random.default_rng(seed).choice(len(uids), size=n_users, replace=False)
        uids, hists = uids[pick], [hists[i] for i in pick]

    idx = BM25Index.build(texts, lang=lang)
    Q = idx.queries_from_history(hists)
    _, top = idx.search_sparse(Q, top_k=max(KS) + 64, universe=universe)
    base = recall_under_canonical(fs, split, uids, top, np.arange(n, dtype=np.int64))

    rows = []
    for thr in thresholds:
        parent = np.arange(n)

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = int(parent[x])
            return x

        m = jv >= thr
        for ii, jjx in zip(ci[m].tolist(), cj[m].tolist()):
            a_, b_ = find(ii), find(jjx)
            if a_ != b_:
                parent[max(a_, b_)] = min(a_, b_)
        root = np.fromiter((find(i) for i in range(n)), dtype=np.int64, count=n)
        members: dict[int, list[int]] = {}
        for i, rt in enumerate(root.tolist()):
            members.setdefault(rt, []).append(i)
        multi = {k_: v for k_, v in members.items() if len(v) > 1}

        # canonical = the most-clicked member, which is the merge a production
        # catalogue would actually perform (keep the copy the audience found)
        canonical = np.arange(n, dtype=np.int64)
        split_clicks = 0
        for grp in multi.values():
            best = max(grp, key=lambda x: click_count[x])
            canonical[grp] = best
            split_clicks += int(click_count[grp].sum() - click_count[best])

        got = recall_under_canonical(fs, split, uids, top, canonical)
        rows.append({
            "threshold": thr, "clusters": len(multi),
            "redundant_docs": int(n - len(set(canonical.tolist()))),
            "redundant_share": (n - len(set(canonical.tolist()))) / n,
            "largest_cluster": max((len(v) for v in multi.values()), default=0),
            "clicks_total": int(click_count.sum()),
            "clicks_on_non_canonical": split_clicks,
            "click_split_share": split_clicks / max(int(click_count.sum()), 1),
            "recall": {f"recall@{k}": got[f"recall@{k}"] for k in KS},
            "delta": {f"recall@{k}": got[f"recall@{k}"] - base[f"recall@{k}"] for k in KS},
        })
    out["dedup_effect"] = {"baseline": {f"recall@{k}": base[f"recall@{k}"] for k in KS},
                           "n_users": int(len(uids)),
                           "n_impressions": base["n_impressions"],
                           "split": split, "rows": rows}


# ================================================================== part E

def bloom_fp_rate(m_bits: int, n_keys: int, k_hash: int) -> float:
    return float((1 - np.exp(-k_hash * n_keys / max(m_bits, 1))) ** k_hash)


def _probe(keys: np.ndarray, k_hash: int, m_bits: int) -> np.ndarray:
    """k independent bit positions for each key, (k_hash, len(keys))."""
    h = splitmix64(keys.astype(np.int64))
    return np.stack([(splitmix64(h ^ u64(0x100000001B3 * (t + 1))) % U64(m_bits)).astype(np.int64)
                     for t in range(k_hash)])


def part_e(fs: FeatureStore, split: str, out: dict, seed: int,
           bits_per_key: list[int], universe_days: int, n_users: int) -> None:
    """Bloom, Count-Min/Count-Sketch, HyperLogLog -- each priced in its own metric."""
    texts = fs.texts()
    lang = fs.lang()
    n = len(texts)
    rng = np.random.default_rng(seed)

    universe = candidate_universe_for_split(fs, split, universe_days)
    uids, hists = user_histories(fs, split)
    if n_users and len(uids) > n_users:
        pick = rng.choice(len(uids), size=n_users, replace=False)
        uids, hists = uids[pick], [hists[i] for i in pick]

    idx = BM25Index.build(texts, lang=lang)
    Q = idx.queries_from_history(hists)
    _, top = idx.search_sparse(Q, top_k=max(KS) + 200, universe=universe)
    ident = np.arange(n, dtype=np.int64)

    # exact drop_seen -- what the pipeline ships
    exact_top = np.full((len(hists), max(KS)), 0, dtype=np.int64)
    for i, h in enumerate(hists):
        seen = set(np.asarray(h).tolist())
        keepd = [d for d in top[i].tolist() if d not in seen][:max(KS)]
        exact_top[i, :len(keepd)] = keepd
    exact = recall_under_canonical(fs, split, uids, exact_top, ident)
    total_keys = int(sum(len(h) for h in hists))
    exact_bytes = total_keys * 8            # one int64 per remembered article id

    bloom_rows = []
    for bpk in bits_per_key:
        k_hash = max(1, int(round(bpk * np.log(2))))
        fp_seen = probes = bits_total = 0
        outk = np.zeros((len(hists), max(KS)), dtype=np.int64)
        for i, h in enumerate(hists):
            keys = np.asarray(h, dtype=np.int64)
            m_bits = max(64, int(bpk * max(keys.size, 1)))
            bits_total += m_bits
            bits = np.zeros(m_bits, dtype=bool)
            if keys.size:
                bits[_probe(keys, k_hash, m_bits).ravel()] = True
            cand = top[i]
            hit = np.all(bits[_probe(cand, k_hash, m_bits)], axis=0)
            truth = np.isin(cand, keys)
            fp_seen += int((hit & ~truth).sum())
            probes += int((~truth).sum())
            keepd = cand[~hit][:max(KS)]
            outk[i, :keepd.size] = keepd
        got = recall_under_canonical(fs, split, uids, outk, ident)
        n_avg = total_keys / max(len(hists), 1)
        bloom_rows.append({
            "bits_per_key": bpk, "k_hash": k_hash,
            "observed_fp": fp_seen / max(probes, 1),
            "predicted_fp": bloom_fp_rate(int(bpk * n_avg), int(n_avg), k_hash),
            "rule_of_thumb_fp": float(0.6185 ** bpk),
            "bytes": int(bits_total / 8),
            "bytes_vs_exact": (bits_total / 8) / max(exact_bytes, 1),
            "recall": {f"recall@{k}": got[f"recall@{k}"] for k in KS},
            "recall_cost": {f"recall@{k}": got[f"recall@{k}"] - exact[f"recall@{k}"] for k in KS},
        })
    out["bloom"] = {"n_users": len(hists), "keys": total_keys, "exact_bytes": exact_bytes,
                    "exact_recall": {f"recall@{k}": exact[f"recall@{k}"] for k in KS},
                    "rows": bloom_rows}

    # ---- Count-Min vs Count-Sketch on the corpus's own term stream
    toks = tokenize(texts, lang)
    tid = toks["token"].cast(pl.Categorical).to_physical().to_numpy().astype(np.int64)
    true_cf = np.bincount(tid)
    rng2 = np.random.default_rng(seed + 1)
    # a flattened stream with the same support and length: the slide claims the
    # ranking between the two sketches flips with skew, which needs both streams
    stream_flat = rng2.integers(0, true_cf.size, size=tid.size)
    true_flat = np.bincount(stream_flat, minlength=true_cf.size)

    sketch_rows = []
    for label, stream, cf in (("zipf", tid, true_cf), ("flat", stream_flat, true_flat)):
        heavy = np.argsort(-cf)[:200]
        nz = np.flatnonzero(cf > 0)
        rand = rng2.choice(nz, size=min(2000, nz.size), replace=False)
        N = int(cf.sum())
        for w in (1 << 10, 1 << 12, 1 << 14, 1 << 16):
            d = 5
            h = np.stack([(splitmix64(stream.astype(U64) ^ u64(0x9E3779B1 * (i + 1))) % U64(w)).astype(np.int64)
                          for i in range(d)])
            sgn = np.stack([(((splitmix64(stream.astype(U64) ^ u64(0xC2B2AE3D * (i + 1))) >> U64(33))
                              & U64(1)).astype(np.int64) * 2 - 1) for i in range(d)])
            cm = np.stack([np.bincount(h[i], minlength=w) for i in range(d)])
            cs = np.stack([np.bincount(h[i], weights=sgn[i], minlength=w) for i in range(d)])

            def est(keys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
                hk = np.stack([(splitmix64(keys.astype(U64) ^ u64(0x9E3779B1 * (i + 1))) % U64(w)).astype(np.int64)
                               for i in range(d)])
                sk = np.stack([(((splitmix64(keys.astype(U64) ^ u64(0xC2B2AE3D * (i + 1))) >> U64(33))
                                 & U64(1)).astype(np.int64) * 2 - 1) for i in range(d)])
                return (np.min(np.stack([cm[i][hk[i]] for i in range(d)]), axis=0),
                        np.median(np.stack([sk[i] * cs[i][hk[i]] for i in range(d)]), axis=0))

            cm_h, cs_h = est(heavy)
            cm_r, cs_r = est(rand)
            sketch_rows.append({
                "stream": label, "width": w, "depth": d, "bytes": int(d * w * 8),
                "cm_mean_abs_err_heavy": float(np.abs(cm_h - cf[heavy]).mean()),
                "cs_mean_abs_err_heavy": float(np.abs(cs_h - cf[heavy]).mean()),
                "cm_mean_abs_err_random": float(np.abs(cm_r - cf[rand]).mean()),
                "cs_mean_abs_err_random": float(np.abs(cs_r - cf[rand]).mean()),
                "cm_predicted_bound": float(np.e * N / w),
                "cm_underestimates": int((cm_r < cf[rand]).sum()),
                "cs_underestimates": int((cs_r < cf[rand]).sum()),
            })
    out["frequency_sketches"] = {"n_tokens": int(tid.size), "vocab": int(true_cf.size),
                                 "rows": sketch_rows}

    # ---- HyperLogLog for distinct users per day
    imp = (fs.impressions(split).select("user_idx", "time").collect()
           .with_columns(pl.col("time").dt.date().alias("day")))
    hll_rows = []
    for m_log in (8, 10, 12, 14):
        m = 1 << m_log
        alpha = 0.7213 / (1 + 1.079 / m)
        errs = []
        for _, grp in imp.group_by(["day"]):
            keys = grp["user_idx"].unique().to_numpy().astype(np.int64)
            if keys.size < 50:
                continue
            h = splitmix64(keys)
            j = (h >> U64(64 - m_log)).astype(np.int64)
            w = (h << U64(m_log)) | U64(1)      # sentinel bit bounds the scan
            rho = np.ones(w.size, dtype=np.int64)
            tmp = w.copy()
            for _ in range(64):
                lead = (tmp >> U64(63)) == 0
                if not lead.any():
                    break
                rho[lead] += 1
                tmp = np.where(lead, tmp << U64(1), tmp)
            reg = np.zeros(m, dtype=np.int64)
            np.maximum.at(reg, j, rho)
            E = alpha * m * m / np.sum(2.0 ** -reg.astype(np.float64))
            if E <= 2.5 * m:                    # small-range correction
                z = int((reg == 0).sum())
                if z:
                    E = m * np.log(m / z)
            errs.append(abs(E - keys.size) / keys.size)
        hll_rows.append({"m": m, "bytes": m, "days": len(errs),
                         "mean_rel_error": float(np.mean(errs)) if errs else None,
                         "predicted_rel_error": float(1.04 / np.sqrt(m))})
    out["hll"] = {"split": split, "rows": hll_rows}


# ================================================================== part F

def part_f(fs: FeatureStore, split: str, out: dict, seed: int,
           weights: list[int], universe_days: int, n_users: int) -> None:
    """Field weighting (the BM25F preview) and the extraction-cost arithmetic."""
    have = set(fs.articles().collect_schema().names())
    # title + abstract only, which is what `text` is and therefore what every other
    # number in this repo is measured on. EB-NeRD also ships a body, but Q2 already
    # priced field *inclusion*; mixing it in here would confound the weight sweep
    # with a field change and make the rows incomparable to the rest of the suite.
    cols = [c for c in ("title", "abstract") if c in have]
    frames = fs.articles().select([pl.col(c).fill_null("") for c in cols]).collect()
    lang = fs.lang()

    universe = candidate_universe_for_split(fs, split, universe_days)
    uids, hists = user_histories(fs, split)
    if n_users and len(uids) > n_users:
        pick = np.random.default_rng(seed).choice(len(uids), size=n_users, replace=False)
        uids, hists = uids[pick], [hists[i] for i in pick]
    ident = np.arange(fs.n_articles, dtype=np.int64)

    title = frames["title"].to_list()
    rest = ([" ".join(frames[c][i] for c in cols if c != "title")
             for i in range(frames.height)] if len(cols) > 1 else [""] * frames.height)

    rows = []
    for w in weights:
        # Field weighting by repetition is the cheap form of BM25F: repeating the
        # title w times before tokenisation multiplies its term frequencies, which
        # is what BM25F's per-field weight does *inside* the saturation rather than
        # outside it. The two differ precisely once tf saturates -- which is the
        # interesting part, since the shipped index concatenates fields at weight 1
        # and has never been asked whether that is the right weight.
        texts = [((t + " ") * w) + rest[i] for i, t in enumerate(title)]
        idx = BM25Index.build(texts, lang=lang)
        Q = idx.queries_from_history(hists)
        _, top = idx.search_sparse(Q, top_k=max(KS), universe=universe)
        got = recall_under_canonical(fs, split, uids, top, ident)
        rows.append({"title_weight": w, "avgdl": idx.avgdl, "vocab": idx.vocab_size,
                     "postings": int(idx.doc_ids.size),
                     **{f"recall@{k}": got[f"recall@{k}"] for k in KS}})
    out["field_weighting"] = {"fields": cols, "rows": rows}

    texts_all = fs.texts()
    chars = sum(len(t or "") for t in texts_all)
    tokens = chars / 4.0                    # the usual bytes-per-token rule of thumb
    per_doc = tokens / max(len(texts_all), 1)
    out["extraction_cost"] = {
        "n_docs": len(texts_all), "mean_tokens_per_doc": per_doc,
        "corpus_tokens": tokens,
        "regex_cpu_hours": len(texts_all) * 0.1e-3 / 3600,
        "llm_batch_usd": tokens / 1e6 * 0.10,
        "llm_frontier_usd": tokens / 1e6 * 1.00,
        "at_1e9_docs": {
            "regex_cpu_hours": 1e9 * 0.1e-3 / 3600,
            "llm_batch_usd": 1e9 * per_doc / 1e6 * 0.10,
            "llm_frontier_usd": 1e9 * per_doc / 1e6 * 1.00,
        },
    }


# ==================================================================== main

def main() -> None:
    p = argparse.ArgumentParser(description="L4 ablation: corpus laws, dedup, sketches")
    p.add_argument("--dataset", default="ebnerd")
    p.add_argument("--variant", default="large")
    p.add_argument("--small-variant", default="small",
                   help="held-out corpus for the Heaps prediction; '' to skip")
    p.add_argument("--split", default="test")
    p.add_argument("--parts", default="abcdef")
    p.add_argument("--sample", type=int, default=5000,
                   help="documents for the exact-Jaccard ground truth (O(n^2))")
    p.add_argument("--shingles", default="2,3,5,9,c5,c9,c16")
    p.add_argument("--sig-ks", default="16,32,64,128,256")
    p.add_argument("--bands", default="32x4,25x5,16x8,10x12")
    p.add_argument("--sig-k", type=int, default=128)
    p.add_argument("--band-b", type=int, default=25)
    p.add_argument("--band-r", type=int, default=5)
    p.add_argument("--thresholds", default="0.5,0.8,0.9")
    p.add_argument("--bits-per-key", default="4,8,10,16")
    p.add_argument("--title-weights", default="1,2,3,5")
    p.add_argument("--users", type=int, default=3000)
    p.add_argument("--universe-days", type=int, default=7)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="reports/l4")
    a = p.parse_args()

    fs = FeatureStore(a.dataset, a.variant)
    fs_small = None
    if a.small_variant and a.small_variant != a.variant:
        try:
            fs_small = FeatureStore(a.dataset, a.small_variant)
        except FileNotFoundError:
            fs_small = None

    res: dict = {"dataset": a.dataset, "variant": a.variant, "split": a.split,
                 "n_articles": fs.n_articles, "lang": fs.lang(), "parts": a.parts,
                 "seed": a.seed, "sample": a.sample}
    print(f"L4 dedup/sketch harness | {a.dataset}/{a.variant} | {fs.n_articles:,} articles"
          f" | lang={fs.lang()}", flush=True)

    if "a" in a.parts:
        print("\n== A. corpus laws: Zipf and Heaps", flush=True)
        part_a(fs, fs_small, res, a.seed)
        for row in res["zipf"]:
            print(f"  {row['stream']:>8}  tokens {row['tokens']:>12,}  vocab {row['vocab']:>9,}"
                  f"  top1 {row['top1_share']:.4f} ('{row['top1_term']}')"
                  f"  top10 {row['top10_share']:.4f}  hapax {row['hapax_share']:.3f}"
                  f"  slope {row['zipf_slope']:+.3f}"
                  f"  top10-postings {row['top10_postings_share']:.4f}", flush=True)
        h = res["heaps"]
        print(f"  Heaps  V = {h['k']:.2f} * N^{h['beta']:.3f}"
              f"   (N={h['n_tokens']:,} -> V={h['vocab']:,})", flush=True)
        if "holdout" in h:
            ho = h["holdout"]
            print(f"    held out: fit on {ho['fit_on']} (k={ho['small_k']:.2f}, "
                  f"beta={ho['small_beta']:.3f}) -> predict {ho['predicted_large_vocab']:,.0f}"
                  f" vs actual {ho['actual_large_vocab']:,} ({ho['rel_error']:+.1%})", flush=True)

    if "b" in a.parts:
        print("\n== B. duplicates: exact, shingles, MinHash, SimHash", flush=True)
        part_b(fs, res, a.sample, a.seed,
               [s.strip() for s in a.shingles.split(",") if s.strip()],
               [int(x) for x in a.sig_ks.split(",")])
        e = res["exact"]
        print(f"  SHA-256: {e['redundant_docs']:,} redundant of {e['n_docs']:,} in "
              f"{e['duplicate_groups']:,} groups (largest {e['largest_group']}), "
              f"{e['seconds']:.2f}s = {e['docs_per_s']:,.0f} docs/s", flush=True)
        for d in res["shingle_dial"]:
            print(f"  {d['unit']:>4} k={d['k']:<3} shingles/doc {d['mean_shingles_per_doc']:7.1f}"
                  f"  empty {d['empty_docs']:>5}  overlapping pairs {d['overlap_share']:.3%}"
                  f"  J>=0.8 {d['pairs_j_ge_0.8']:>7,}  p99.9 J {d['p999_j']:.3f}", flush=True)
        for acc in res["minhash"]["accuracy"]:
            print(f"  MinHash k={acc['k']:<4} {acc['bytes_per_doc']:>5} B/doc  "
                  f"rmse {acc['rmse']:.4f}  {acc['docs_per_s']:>9,.0f} docs/s", flush=True)
            for bn in acc["bins"]:
                print(f"      J~{bn['j_mean']:.2f} (n={bn['n']:>5})  sd obs {bn['observed_sd']:.4f}"
                      f"  pred {bn['predicted_sd']:.4f}  bias {bn['bias']:+.4f}", flush=True)
        s = res["simhash"]
        print(f"  SimHash 8 B/doc in {s['sign_seconds']:.2f}s, "
              f"corr(-hamming, J) = {s['corr_negdist_j']:.3f}", flush=True)
        for op in s["operating_points"]:
            print(f"      hamming<={op['hamming_le']:>2} @ J>={op['j_threshold']}: "
                  f"P {op['precision']:.3f}  R {op['recall']:.3f}", flush=True)

    if "c" in a.parts:
        print("\n== C. banding and the O(n^2) wall", flush=True)
        bands = [(int(x.split("x")[0]), int(x.split("x")[1]))
                 for x in a.bands.lower().split(",")]
        part_c(fs, res, a.sample, a.seed, bands, a.sig_k)
        bd = res["banding"]
        w = bd["wall"]
        print(f"  brute force: {bd['all_pairs']:,} pairs in {bd['brute_force_seconds']:.2f}s "
              f"({w['measured_pairs_per_s']:,.0f} pairs/s -> "
              f"{w['brute_force_years_at_1e9_docs']:,.0f} years at 1e9 docs)", flush=True)
        for c_ in bd["curves"]:
            print(f"  b={c_['b']:>3} r={c_['r']:<3} t~{c_['theoretical_threshold']:.3f}  "
                  f"candidates {c_['candidates']:>9,} "
                  f"({c_['candidate_share_of_all_pairs']:.2e} of all pairs) "
                  f"in {c_['seconds']:.2f}s", flush=True)
            for row in c_["at_threshold"]:
                print(f"      J>={row['threshold']}: {row['true_pairs']:>8,} true, "
                      f"recall {row['recall']:.3f}  precision {row['precision']:.4f}", flush=True)
        print(f"  operating point b={w['lsh_b']} r={w['lsh_r']} "
              f"(recall {w['lsh_recall_at_target']:.3f} at J>=0.5, target met: "
              f"{w['operating_point_met']}): verification alone is "
              f"{w['lsh_verify_years_at_1e9_docs']:.3f} years at 1e9 docs, "
              f"{1 / max(w['lsh_candidate_share'], 1e-12):,.0f}x fewer pairs", flush=True)

    if "d" in a.parts:
        print("\n== D. near-dups in the catalogue, and what they cost retrieval", flush=True)
        part_d(fs, a.split, res, a.seed, a.sig_k, a.band_b, a.band_r,
               [float(x) for x in a.thresholds.split(",")], a.universe_days, a.users)
        c_ = res["catalogue"]
        print(f"  {c_['n_docs']:,} docs -> {c_['candidates']:,} candidates "
              f"({c_['candidate_share']:.2e} of {c_['all_pairs']:,}) | "
              f"shingle {c_['shingle_seconds']:.1f}s  sign {c_['sign_seconds']:.1f}s  "
              f"band {c_['band_seconds']:.1f}s  verify {c_['verify_seconds']:.1f}s", flush=True)
        print(f"  verified: {c_['verified_ge_0.5']:,} at J>=0.5, "
              f"{c_['verified_ge_0.8']:,} at J>=0.8, {c_['verified_ge_0.9']:,} at J>=0.9 "
              f"| signatures {c_['signature_mb']:.1f} MB vs shingles {c_['shingle_mb']:.1f} MB",
              flush=True)
        de = res["dedup_effect"]
        print(f"  baseline recall@100 {de['baseline']['recall@100']:.5f} "
              f"({de['n_users']:,} users, {de['n_impressions']:,} impressions)", flush=True)
        for row in de["rows"]:
            print(f"  J>={row['threshold']}: {row['clusters']:,} clusters, "
                  f"{row['redundant_docs']:,} redundant ({row['redundant_share']:.2%}), "
                  f"largest {row['largest_cluster']}, click split "
                  f"{row['click_split_share']:.3%} | r@100 {row['recall']['recall@100']:.5f} "
                  f"({row['delta']['recall@100']:+.5f})", flush=True)

    if "e" in a.parts:
        print("\n== E. sketches, priced in the metric they change", flush=True)
        part_e(fs, a.split, res, a.seed, [int(x) for x in a.bits_per_key.split(",")],
               a.universe_days, a.users)
        bl = res["bloom"]
        print(f"  Bloom seen-set: {bl['keys']:,} keys over {bl['n_users']:,} users; "
              f"exact {bl['exact_bytes'] / 1e6:.2f} MB, "
              f"exact r@100 {bl['exact_recall']['recall@100']:.5f}", flush=True)
        for row in bl["rows"]:
            print(f"    {row['bits_per_key']:>3} bits/key k={row['k_hash']}  fp obs "
                  f"{row['observed_fp']:.5f}  pred {row['predicted_fp']:.5f}  rule "
                  f"{row['rule_of_thumb_fp']:.5f} | {row['bytes_vs_exact']:.3f}x bytes | "
                  f"r@100 {row['recall']['recall@100']:.5f} "
                  f"({row['recall_cost']['recall@100']:+.5f})", flush=True)
        for row in res["frequency_sketches"]["rows"]:
            print(f"    {row['stream']:>4} w={row['width']:>6} ({row['bytes'] / 1e3:>5.0f} KB)  "
                  f"heavy CM {row['cm_mean_abs_err_heavy']:>11,.1f} / CS "
                  f"{row['cs_mean_abs_err_heavy']:>11,.1f} | rand CM "
                  f"{row['cm_mean_abs_err_random']:>9,.1f} / CS "
                  f"{row['cs_mean_abs_err_random']:>9,.1f} | CM bound "
                  f"{row['cm_predicted_bound']:>10,.0f}", flush=True)
        for row in res["hll"]["rows"]:
            print(f"    HLL m={row['m']:>6} ({row['bytes'] / 1e3:.1f} KB)  err obs "
                  f"{row['mean_rel_error']:.4f}  pred {row['predicted_rel_error']:.4f}  "
                  f"over {row['days']} days", flush=True)

    if "f" in a.parts:
        print("\n== F. field weighting and extraction economics", flush=True)
        part_f(fs, a.split, res, a.seed, [int(x) for x in a.title_weights.split(",")],
               a.universe_days, a.users)
        for row in res["field_weighting"]["rows"]:
            print(f"  title x{row['title_weight']:<3} avgdl {row['avgdl']:6.1f}  vocab "
                  f"{row['vocab']:>8,}  postings {row['postings']:>10,}  "
                  + "  ".join(f"r@{k} {row[f'recall@{k}']:.5f}" for k in KS), flush=True)
        ec = res["extraction_cost"]
        print(f"  extraction: {ec['n_docs']:,} docs x {ec['mean_tokens_per_doc']:.0f} tokens "
              f"= {ec['corpus_tokens'] / 1e6:.1f}M tokens | regex "
              f"{ec['regex_cpu_hours']:.2f} CPU-h | LLM batch ${ec['llm_batch_usd']:.2f} "
              f"| frontier ${ec['llm_frontier_usd']:.2f}", flush=True)
        at = ec["at_1e9_docs"]
        print(f"    at 1e9 docs: regex {at['regex_cpu_hours']:,.0f} CPU-h | LLM batch "
              f"${at['llm_batch_usd']:,.0f} | frontier ${at['llm_frontier_usd']:,.0f}", flush=True)

    outdir = Path(a.out)
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"l4_dedup_{a.dataset}_{a.variant}.json"
    path.write_text(json.dumps(res, indent=2, default=float))
    print(f"\nwrote {path}", flush=True)


if __name__ == "__main__":
    main()
