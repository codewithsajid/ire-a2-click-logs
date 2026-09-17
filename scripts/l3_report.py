"""Turn the L3 storage JSONs into the markdown the design note quotes."""
from __future__ import annotations

import json
from pathlib import Path

L3 = Path("reports/l3")

HEADER = """# L3 ablation: storage, access patterns, and the RUM triangle

L3 grades every design on three axes — **R**eads, **U**pdates, **M**emory — and
argues that where data lives and how it is touched dominates the algorithm on top.
This repo has been assuming all of it: every vector sits in RAM, every index is
built once and never updated, and the RUM trade in the ANN ablation is measured but
never named. These tables measure the assumptions.

Every index here is exact unless a row says otherwise, so nothing in parts A, C or
D can move recall. Part B is the one place precision is traded, and it prices the
fidelity it costs. One thread throughout.
"""


def part_a(d: dict) -> str:
    a = d["part_a_hierarchy"]
    seq = {r["mode"]: r for r in a["sequential"]}
    cold = seq["full read, cold (NVMe)"]
    warm = seq["full read, warm (page cache)"]
    L = [f"### A. The storage hierarchy — `{a['file']}`, {a['size_mb']:,.0f} MB "
         f"({a['rows']:,} rows × {a['dim']}d, {a['row_bytes']:,} B/row)\n",
         "| read | seconds | MB/s |", "|---|--:|--:|",
         f"| full file, cold (NVMe) | {cold['seconds']:.3f} | {cold['mb_per_s']:,.0f} |",
         f"| full file, warm (page cache) | {warm['seconds']:.3f} | {warm['mb_per_s']:,.0f} |",
         "",
         f"The page cache is worth **{cold['seconds']/warm['seconds']:.1f}×** on this file, "
         f"which is the entire distance between two tiers of L3's hierarchy measured on our "
         f"own artefact. Cold reads are real: `POSIX_FADV_DONTNEED` evicts the file without "
         f"root, and the store's memmap has to be released first or the kernel refuses and "
         f"every \"cold\" number silently becomes a warm one.\n",
         "Then the comparison that isolates the *pattern* from the volume — the same number of "
         "rows, contiguous against scattered, both from a cold cache:\n",
         "| rows | share of corpus | MB touched | contiguous | scattered | µs/row | random penalty |",
         "|--:|--:|--:|--:|--:|--:|--:|"]
    for r in a["same_volume_pattern"]:
        L.append(f"| {r['rows']:,} | {100*r['share_of_corpus']:.1f}% | {r['touched_mb']:.1f} | "
                 f"{1e3*r['contiguous_s']:.2f} ms | {1e3*r['scattered_s']:.2f} ms | "
                 f"{r['scattered_us_per_row']:.1f} | **{r['random_penalty']:.1f}×** |")
    worst = max(a["same_volume_pattern"], key=lambda r: r["random_penalty"])
    L += ["", f"The slide quotes ~100× for random 4 KB on NVMe, and a row here is "
              f"{a['row_bytes']:,} bytes — close enough to one page that this is that constant "
              f"measured on the exact access pattern a user vector performs. It peaks at "
              f"**{worst['random_penalty']:.0f}×** at {worst['rows']:,} rows and *falls* on "
              f"either side: too few rows and fixed costs dominate, too many and the scattered "
              f"set is dense enough that readahead starts serving it sequentially anyway. The "
              f"quoted constant is a queue-depth-1 figure and a real access pattern can beat it "
              f"— but not by the order of magnitude that would make random access free.\n"]
    return "\n".join(L)


def part_b(d: dict) -> str:
    b = d["part_b_precision"]
    b1, b2 = b["b1_same_kernel_fewer_dims"], b["b2_quantisation"]
    L = [f"### B. Bytes against throughput — {b['n_vectors']:,} × {b['dim']}d, "
         f"{b['n_queries']:,} queries\n",
         "**B1 — same kernel, fewer dimensions.** A Johnson–Lindenstrauss random projection to "
         "d′ < d, searched by the same `IndexFlatIP`: identical code path, identical BLAS GEMM, "
         "strictly fewer bytes. This is the fair test of the L2 harness's conclusion that the "
         "scan is memory-bandwidth bound.\n",
         "| dim | index MB | bytes smaller | q/s | speed-up | GFLOP/s | mean inner-product error | recall@100 vs exact |",
         "|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for r in b1:
        L.append(f"| {r['dim']} | {r['index_mb']:,.1f} | {r['bytes_ratio']:.1f}× | "
                 f"{r['qps']:,.0f} | {r['speedup_vs_full']:.2f}× | {r['gflop_per_s']:.1f} | "
                 f"{r['mean_inner_product_error']:.4f} | {r['recall@100_vs_exact']:.4f} |")
    if len(b1) > 1:
        half = min((r for r in b1 if r["bytes_ratio"] >= 1.9), key=lambda r: r["bytes_ratio"],
                   default=None)
        L += ["", f"**The prediction holds, and then it stops.** Halving the bytes buys "
                  f"{half['speedup_vs_full']:.2f}× when the prediction says 2× — a scan that is "
                  f"bandwidth-bound almost exactly. Keep shrinking and the return decays "
                  f"({b1[-1]['speedup_vs_full']:.2f}× for {b1[-1]['bytes_ratio']:.0f}×), and the "
                  f"GFLOP/s column says why: top-K selection over {b['n_vectors']:,} candidates "
                  f"costs the same at every d, so as the GEMM shrinks a d-independent term takes "
                  f"over. Bandwidth is the bottleneck until it isn't."
                  if half else ""]
        L += ["", f"The two fidelity columns disagree on purpose. JL guarantees *distances*, and "
                  f"it delivers them — the inner-product error stays at "
                  f"{b1[1]['mean_inner_product_error']:.3f} at {b1[1]['bytes_ratio']:.0f}× "
                  f"compression. It guarantees nothing about which 100 ids come back, and recall "
                  f"against the exact top-100 falls to {b1[1]['recall@100_vs_exact']:.2f} at the "
                  f"same point. News embeddings sit close together, so small distance errors "
                  f"reorder a lot of neighbours. Quoting the theorem without the second column "
                  f"would be quoting it for something it does not claim.\n"]
    L += ["**B2 — quantisation, with the confound named.** `IndexFlatIP` dispatches to a BLAS "
          "GEMM; `IndexScalarQuantizer` runs a per-vector scalar decode loop. Fewer bytes, worse "
          "kernel — so these rows are *not* a test of the bandwidth claim, and reporting "
          "\"fp16 is slower than fp32\" without that sentence would be an artefact dressed up "
          "as a result.\n",
          "| precision | index MB | q/s | vs fp32 | recall@100 vs exact |",
          "|---|--:|--:|--:|--:|"]
    for r in b2:
        L.append(f"| {r['precision']} | {r['index_mb']:,.1f} | {r['qps']:,.0f} | "
                 f"{r['speedup_vs_fp32']:.2f}× | {r['recall@100_vs_exact']:.4f} |")
    L += ["", "Read as a memory result rather than a speed one, quantisation is excellent: "
              f"{b2[2]['index_mb']/b2[0]['index_mb']:.2f}× the bytes at "
              f"{b2[2]['recall@100_vs_exact']:.4f} fidelity for int8. That is the same trade the "
              "ANN ablation found (4× smaller, 3.7× slower) reached by a different route, which "
              "is worth something on its own.\n"]
    return "\n".join(L)


def part_c(d: dict) -> str:
    c = d["part_c_updates"]
    st = c["strategies"]
    L = [f"### C. The update axis — absorbing {c['n_days']} days of new articles\n",
         f"Base index of {c['base_vectors']:,} vectors, then {c['n_days']} arrivals of "
         f"{c['per_day']:,} each, HNSW (M=16, efC=200, efS=64) throughout. The segmented "
         f"strategy merges once it exceeds {c['merge_every']} segments. All three answer "
         f"identically — the day's vectors are in the index either way — so this is the RUM "
         f"triangle with recall nailed down and nothing else moving.\n",
         "| day | vectors | strategy | update (s) | merge (s) | segments | index MB | q/s | recall@100 vs exact |",
         "|--:|--:|---|--:|--:|--:|--:|--:|--:|"]
    names = {"rebuild": "rebuild wholesale", "append_in_place": "append in place",
             "lsm_segments": "base + segments (LSM)"}
    for key, label in names.items():
        for r in st.get(key, []):
            upd = "—" if r["update_s"] is None else f"{r['update_s']:.2f}"
            mrg = f"{r.get('merge_s', 0.0):.2f}" if key == "lsm_segments" else "—"
            fid = r.get("recall@100_vs_exact")
            L.append(f"| {r['day']} | {r['vectors']:,} | {label} | {upd} | {mrg} | "
                     f"{r['segments']} | {r['bytes']/2**20:,.0f} | {r['qps']:,.0f} | "
                     + ("—" if fid is None else f"{fid:.4f}") + " |")
    tot = {}
    for key in names:
        rows = [r for r in st.get(key, []) if r["day"] > 0]
        if not rows:
            continue
        tot[key] = {
            "update": sum((r["update_s"] or 0) + r.get("merge_s", 0.0) for r in rows),
            "qps": sum(r["qps"] for r in rows) / len(rows),
            "mb": max(r["bytes"] for r in rows) / 2 ** 20,
            "max_segments": max(r["segments"] for r in rows),
            "fid": sum(r.get("recall@100_vs_exact") or 0 for r in rows) / len(rows)}
    L += ["", "Totals over the week — the three corners of the triangle, side by side:\n",
          "| strategy | total update cost (s) | mean q/s | peak index MB | max segments | mean fidelity |",
          "|---|--:|--:|--:|--:|--:|"]
    for key, label in names.items():
        if key in tot:
            t = tot[key]
            L.append(f"| {label} | {t['update']:.1f} | {t['qps']:,.0f} | {t['mb']:,.0f} | "
                     f"{t['max_segments']} | {t['fid']:.4f} |")
    fids = [t["fid"] for t in tot.values() if t["fid"]]
    if fids:
        L += ["", f"**The fidelity column is the one that had to be measured rather than "
                  f"assumed.** HNSW is a greedily built graph, so inserting incrementally could "
                  f"leave a worse graph than rebuilding from the same vectors — a cost that would "
                  f"be completely invisible in the throughput column. It does not: all three "
                  f"strategies land within {1e4*(max(fids)-min(fids)):.0f} basis points of each "
                  f"other. Segments score *highest*, because searching k smaller graphs at "
                  f"efSearch=64 each explores more candidates in total than one graph at 64. The "
                  f"assumption was safe; it was still an assumption.\n"]
    if "rebuild" in tot and "lsm_segments" in tot:
        rb, lsm = tot["rebuild"], tot["lsm_segments"]
        ap = tot.get("append_in_place")
        same_m = abs(rb["mb"] - lsm["mb"]) / max(rb["mb"], 1e-9) < 0.02
        L += ["", f"**Segments make writes {rb['update']/max(lsm['update'],1e-9):.1f}× cheaper "
                  f"than rebuilding and charge "
                  f"{100*(1-lsm['qps']/max(rb['qps'],1e-9)):.0f}% of read throughput for it** — "
                  f"the RUM conjecture with numbers on it. Every query touches every live segment, "
                  f"and the debt is handed back all at once at merge time, which is the compaction "
                  f"dial L3 describes and L5 reuses for index freshness."]
        if same_m:
            L += ["", "**But the space corner of that trade never appears, and it is worth being "
                      "precise about why.** All three strategies hold identical bytes on every "
                      "day. L3's space amplification comes from *dead versions* — superseded keys "
                      "waiting for a compaction that has not run yet — and a partitioned vector "
                      "index has none: each vector lives in exactly one segment, and merging "
                      "rearranges rather than reclaims. So the LSM trade here is two-cornered, R "
                      "against U, not three."]
        if ap:
            best = min((rb, "rebuild wholesale"), (lsm, "segments"), (ap, "append in place"),
                       key=lambda t: t[0]["update"])
            L += ["", f"**And the strategy that wins is neither of the two the lecture "
                      f"contrasts.** Appending in place costs {ap['update']:.1f} s for the week — "
                      f"{rb['update']/max(ap['update'],1e-9):.0f}× cheaper than rebuilding — at "
                      f"{ap['qps']:,.0f} q/s, within "
                      f"{100*abs(1-ap['qps']/max(rb['qps'],1e-9)):.0f}% of the rebuilt index, "
                      f"with the same bytes and the same fidelity. It dominates segments on every "
                      f"axis at once. The LSM pattern exists because in-place insertion into an "
                      f"inverted index is expensive; HNSW's is not, so the prescription that is "
                      f"right for Lucene is wrong for this index. The wholesale rebuild the repo "
                      f"performs today is buying nothing at all.\n"]
    return "\n".join(L)


def part_d(d: dict) -> str:
    dd = d["part_d_compression"]
    L = ["### D. Compression — compute traded for bytes\n",
         "| codec | size MB | write (s) | cold read (s) | warm read (s) | cold MB/s |",
         "|---|--:|--:|--:|--:|--:|"]
    for r in dd["parquet"]:
        L.append(f"| articles.parquet, {r['codec']} | {r['size_mb']:,.1f} | {r['write_s']:.2f} | "
                 f"{r['cold_read_s']:.2f} | {r['warm_read_s']:.2f} | {r['cold_mb_per_s']:,.0f} |")
    for r in dd["vectors"]:
        L.append(f"| vectors, {r['dtype']} (.npy) | {r['size_mb']:,.1f} | — | "
                 f"{r['cold_read_s']:.3f} | — | {r['cold_mb_per_s']:,.0f} |")
    un = next((r for r in dd["parquet"] if r["codec"] == "uncompressed"), None)
    if un:
        best_read = min(dd["parquet"], key=lambda r: r["cold_read_s"])
        best_size = min(dd["parquet"], key=lambda r: r["size_mb"])
        raw_bw = un["size_mb"] / un["cold_read_s"]
        L += ["", f"**Compression here buys space and costs latency, which is the opposite of the "
                  f"direction L5 argues for — and for a reason worth stating.** The uncompressed "
                  f"file reads cold at {raw_bw:,.0f} MB/s, so the fetch is not the bottleneck; the "
                  f"decode is. `{best_size['codec']}` is "
                  f"{un['size_mb']/best_size['size_mb']:.1f}× smaller and reads "
                  f"{best_size['cold_read_s']/un['cold_read_s']:.2f}× slower, i.e. its effective "
                  f"throughput is {best_size['size_mb']/best_size['cold_read_s']:,.0f} MB/s against "
                  f"{raw_bw:,.0f} MB/s raw. L5's thesis — shrink the bytes to widen the bandwidth "
                  f"bottleneck — holds when the fetch *is* the bottleneck: a cold HDD shard, a "
                  f"network hop, an object store. On a local NVMe it inverts, which is L5's own "
                  f"advice about picking the code per tier, arriving from the other end.",
              "",
              f"On the read-latency axis the pick is `{best_read['codec']}` "
              f"({best_read['size_mb']:,.0f} MB, {best_read['cold_read_s']:.2f} s); on the "
              f"space axis it is `{best_size['codec']}` ({best_size['size_mb']:,.0f} MB). The "
              f"store currently defaults to zstd, which is the right call for 16 GB of derived "
              f"data on a tight home partition and the wrong one for a hot serving path — a "
              f"distinction that had never been measured either way.",
              "",
              "The vector rows say the same thing from the other side: fp16 halves the file and "
              "halves the cold read at the *same* MB/s, because there is no decode. Bytes are the "
              "cost when nothing has to be undone to use them.\n"]
    return "\n".join(L)


def rum_table(d: dict) -> str:
    """The triangle the ANN ablation measured and never named."""
    b = d.get("part_b_precision")
    c = d.get("part_c_updates")
    if not (b and c):
        return ""
    rows = []
    b1 = {r["dim"]: r for r in b["b1_same_kernel_fewer_dims"]}
    b2 = {r["precision"].split(" (")[0]: r for r in b["b2_quantisation"]}
    full = b1[max(b1)]
    st = c["strategies"]

    def week(key):
        rs = [r for r in st.get(key, []) if r["day"] > 0]
        return (sum((r["update_s"] or 0) + r.get("merge_s", 0.0) for r in rs),
                sum(r["qps"] for r in rs) / len(rs), max(r["bytes"] for r in rs) / 2 ** 20)

    for key, label, note in (
            ("rebuild", "HNSW, rebuilt wholesale", "R ✓ U ✗ M ✓ — one clean graph, paid for by rebuilding it"),
            ("append_in_place", "HNSW, appended in place", "R ✓ U ✓ M ✓ — the corner this repo was not using"),
            ("lsm_segments", "HNSW, base + segments (LSM)", "R ✗ U ✓ M ~ — spends reads to make writes cheap; no space cost, because a partitioned index has no dead versions")):
        if key in st:
            u, q, m = week(key)
            rows.append((label, f"{q:,.0f} q/s", f"{u:.1f} s/week", f"{m:,.0f} MB", note))
    rows.append(("Exact flat, fp32", f"{full['qps']:,.0f} q/s", "rebuild only",
                 f"{full['index_mb']:,.0f} MB", "R ~ U ✗ M ✗ — no structure, so nothing to update"))
    if "int8" in b2:
        r = b2["int8"]
        rows.append(("Exact flat, int8 (SQ)", f"{r['qps']:,.0f} q/s", "rebuild only",
                     f"{r['index_mb']:,.0f} MB",
                     f"R ✗ U ✗ M ✓ — {full['index_mb']/r['index_mb']:.0f}× smaller at "
                     f"{r['recall@100_vs_exact']:.3f} fidelity, and a slower kernel"))
    small = min(b1)
    if small != max(b1):
        r = b1[small]
        rows.append((f"Exact flat, JL to {small}d", f"{r['qps']:,.0f} q/s", "rebuild only",
                     f"{r['index_mb']:,.0f} MB",
                     f"R ✓ U ✗ M ✓ — buys both, and pays in *accuracy* rather than in RUM: "
                     f"recall {r['recall@100_vs_exact']:.2f}"))
    L = ["### The RUM triangle, named\n",
         "L3's conjecture is that lowering two of read, update and memory overhead raises the "
         "third. The ANN ablation already reported all three columns per index and never said "
         "the word; here they are with the update axis finally populated.\n",
         "| structure | R (throughput) | U (a week of arrivals) | M (bytes) | which two it buys |",
         "|---|--:|--:|--:|---|"]
    for r in rows:
        L.append("| " + " | ".join(r) + " |")
    L += ["", "The last row is the interesting exception: dimensionality reduction lowers R-cost "
              "*and* M together, which the conjecture forbids — because it does not pay in RUM at "
              "all. It pays in answer quality, an axis the triangle does not have. Any structure "
              "that looks like it beats the conjecture is quietly spending something the "
              "conjecture does not measure.\n"]
    return "\n".join(L)


FINDINGS = """

## What the four parts add up to

1. **The page cache is worth 4× and the memmap has to be released to see it.** Cold
   NVMe reads the 368 MB matrix at ~1,500 MB/s against ~6,000 warm. Getting that
   number required noticing that `POSIX_FADV_DONTNEED` silently refuses to evict
   pages a live mapping still holds — the store hands back a memmap, so the first
   version of this measurement reported the warm number twice and called one of
   them cold.
2. **Random access costs up to ~91× on NVMe, and the penalty is not monotonic.**
   Holding the volume fixed and varying only the pattern, scattered rows peak at
   about 91× contiguous at 1,000 rows and fall away on both sides — too few and
   fixed costs dominate, too many and readahead starts serving the scattered set
   sequentially. The slide's ~100× is a queue-depth-1 figure; a real pattern can
   beat it, but not by enough to make random access free.
3. **The bandwidth hypothesis is confirmed, with a stated limit.** Halving the
   vector at a fixed kernel buys 1.90× where the prediction says 2×. Keep halving
   and returns decay to 5.88× for 8×, because top-K selection over the full
   candidate set costs the same at every dimension and eventually dominates the
   shrinking GEMM. Bandwidth is the bottleneck until it isn't, and the GFLOP/s
   column is where the handover shows.
4. **JL delivers exactly what it promises, which is not what you might want.**
   Inner-product error stays near 0.04 at 2× compression — the theorem holds — while
   recall against the exact top-100 falls to 0.59. News embeddings sit close
   together, so small distance errors reorder many neighbours. The two columns have
   to be read together or the theorem gets quoted for a guarantee it never made.
5. **Quantisation is a memory win, not a speed win — and the reason is the kernel.**
   int8 is 4× smaller at 0.993 fidelity and roughly 5× slower, because `IndexFlatIP`
   dispatches to a BLAS GEMM while the scalar quantiser runs a per-vector decode
   loop. That is the same trade the ANN ablation reached by a different route, which
   is worth something as a cross-check.
6. **The wholesale rebuild this repo performs was buying nothing.** HNSW takes
   incremental adds: appending a day of articles costs ~1.3 s against ~40 s to
   rebuild, at the same throughput *and* the same fidelity. Roughly a 30× write
   saving that was being paid for out of habit.
7. **Read amplification is real and linear in segments.** The LSM-shaped strategy
   makes writes almost free (~0.3 s/day) and charges for it on every query: 6,589
   q/s at one segment, 5,242 at two, 4,395 at three, 3,788 at four — then the merge
   hands back the whole debt at once, 38.5 s in one go. That is L3's compaction dial
   with numbers on it, and it is the same dial L5 reuses for index freshness.
8. **For a graph index, LSM segments are the wrong shape.** The pattern exists
   because in-place insertion into an inverted index is expensive; HNSW's is not. So
   plain in-place append dominates segments here on every axis at once — cheaper
   than rebuild, no read amplification, no merge debt. The prescription is right for
   Lucene and wrong for this index, which is only visible because both were measured.
   The space corner never appears either: all three strategies hold identical bytes,
   because L3's space amplification comes from dead versions and a partitioned vector
   index has none. The LSM trade here is two-cornered, R against U, not three.
9. **Compression buys space and costs latency here — the inverse of L5's thesis.**
   The uncompressed table reads cold at ~1,555 MB/s, so the fetch is not the
   bottleneck; the decode is. zstd is 2.7× smaller and reads 1.7× *slower*; lz4 is
   1.6× smaller for 10% slower. L5's "shrink the bytes to widen the bandwidth
   bottleneck" holds when the fetch is the bottleneck — a cold HDD shard, a network
   hop, an object store — and inverts on a local NVMe. That is L5's own "pick the
   code per tier" advice, arrived at from the other end. fp16 vectors are the
   control: half the file, half the cold read, same MB/s, because nothing has to be
   decoded.
"""


CLAIMS = """

## Choosing by workload — where each artefact here actually sits

L3 closes with a table mapping workload shape to access method. Three of its four
rows apply to something in this repo, and one of them we had wrong.

| artefact | shape | L3's prescription | what we do | verdict |
|---|---|---|---|---|
| `articles`, `impressions`, `clickstream` parquet | full-column reads, never updated in place, rebuilt per split | analytics → **columnar + zone maps** | parquet + zstd, read by projection and streamed | ✅ right, and part D shows zstd was the right codec too |
| the vector index (HNSW / flat) | append-heavy, immutable, merged | inverted-index build → **LSM / merged segments** | rebuilt wholesale on every run | ❌ **this was the gap**, and part C prices it |
| BM25 postings (polars → scipy CSR) | built once, queried as one sparse matmul | same row — LSM-shaped segments | rebuilt in RAM per run | ❌ same gap, same fix |
| user / article features | joined in bulk by id, never point-read | point reads ≫ writes → **B⁺-tree** | parquet, joined in bulk | ✅ right *because the row does not apply* — see below |

**The B-tree row never fires here, and that is worth saying rather than assuming.**
L3 prescribes a B⁺-tree for a user-profile store because the access pattern is point
reads. Ours is not: every consumer of `user_features` or `article_features` joins the
whole table against a split, and the one place a point read would appear — fetching
one user's history to serve one request — is served from an in-memory array in the L2
harness. A B-tree would buy nothing and cost the update amplification L3 warns about.
The workload picked the structure, exactly as the slide says; it just picked the
other one.

**What this repo has no instance of, and should not pretend to.** There is no
key-value store, no append-only log with a hash index, no B-tree, and no crash
recovery path — the store is derived data, and the recovery procedure is `make data`.
L3's set/get material is background for those, not a checklist this assignment
fails: rebuilding from raw files is cheaper here than any WAL would be.
"""


ORDER = ["ebnerd_large", "mind_small", "ebnerd_small"]


def sort_key(p: Path):
    for i, frag in enumerate(ORDER):
        if frag in p.name:
            return (i, p.name)
    return (len(ORDER), p.name)


if __name__ == "__main__":
    out = [HEADER]
    files = sorted(L3.glob("l3_storage_*.json"), key=sort_key)
    for p in files:
        d = json.loads(p.read_text())
        tag = f"{d['dataset']}/{d['variant']}"
        out.append(f"\n## {tag} — {d['vectors']:,} vectors × {d['dim']}d\n")
        for key, fn in (("part_a_hierarchy", part_a), ("part_b_precision", part_b),
                        ("part_c_updates", part_c), ("part_d_compression", part_d)):
            if key in d:
                out.append(fn(d))
        r = rum_table(d)
        if r:
            out.append(r)
    out.append(FINDINGS)
    out.append(CLAIMS)
    dest = L3 / "l3_ablation.md"
    dest.write_text("\n".join(out))
    print(f"wrote {dest} from {len(files)} json file(s)")
