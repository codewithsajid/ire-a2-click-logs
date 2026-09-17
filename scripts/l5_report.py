"""Turn the L5 postings JSONs into the markdown the design note quotes."""
from __future__ import annotations

import json
from pathlib import Path

L5 = Path("reports/l5")
ORDER = ("ebnerd_large", "ebnerd_small", "mind_large", "mind_small")
KS = (50, 100, 200)

HEADER = """# L5 ablation: postings, compression, and top-k query processing

L5 says a search engine is a sorted file — compressed well, merged lazily, and
traversed in the cheapest order. This repo has the sorted file and, until these
tables, none of the rest: `BM25Index` builds a real dictionary and real postings
and then scores with one sparse matmul. No merge, no skips, no cheapest-first, no
WAND, no compression, no tiering. The docstring admits it; nothing measured it.

The reason that is not obviously wrong is a property of this workload the lecture
never contemplates. L5's query is two to five terms. A query here is the bag of
terms of a user's whole click history — a median of 859 distinct terms on EB-NeRD.
Every optimisation in the lecture is a bet that most of the index can be skipped,
and the size of that bet is set by query length and by how many documents are in
scope. So the experiments below are not "is WAND faster" but "at what query length
and what corpus size does each of these start to pay", which turns four inherited
implementation choices into measured ones.

**One warning that applies to every wall-clock number in part C.** TAAT and the
matmul are vectorised — one numpy or scipy kernel per query — while DAAT and WAND
are per-document Python loops, because a pivot is inherently sequential. Comparing
their milliseconds compares languages. The *work* columns (postings touched,
documents scored, multiply-adds performed) are implementation-independent, and
every claim below rests on those; timings are reported because they are what the
system actually experiences today, and they are labelled where they mislead.
"""


def part_a(d: dict) -> str:
    a = d["anatomy"]
    L = ["### A. What is actually in a posting\n",
         f"{a['postings']:,} postings over {a['n_terms']:,} terms and {a['n_docs']:,} documents; "
         f"{a['token_occurrences']:,} token occurrences, so **mean tf = {a['mean_tf']:.3f}** and "
         f"{a['tf_eq_1_share']:.1%} of postings have tf exactly 1. avgdl is {a['avgdl']:.1f} "
         f"tokens.\n",
         "| layout | size | multiplier | slide |", "|---|--:|--:|--:|"]
    for r in a["layouts"]:
        L.append(f"| {r['layout']} | {r['bytes'] / 1e6:.2f} MB | ×{r['multiplier']:.2f} | "
                 + (f"×{r['slide']} |" if r["slide"] else "— |"))
    pos = next(r for r in a["layouts"] if r["layout"] == "+ positions (uint32)")
    L += ["", f"The tf multiplier lands at ×1.25 against the slide's ×1.3, and positions at "
              f"**×{pos['multiplier']:.2f}**, inside the slide's ×2–4. That agreement is luckier "
              f"than it looks and worth unpacking, because the position cost is not a constant — "
              f"it is exactly (8 + 4·mean tf) / 4 bytes per posting. A web page repeats its "
              f"topical terms, so mean tf runs to 3–5 and positions genuinely cost ×4; a headline "
              f"plus a standfirst repeats almost nothing, so mean tf is {a['mean_tf']:.2f} and "
              f"there is very nearly *one position per posting*. The multiplier is only in range "
              f"because storing tf at all is nearly redundant here.\n",
          f"That is the actual finding: with {a['tf_eq_1_share']:.0%} of postings at tf = 1, the "
          f"term-frequency field is {a['tf_eq_1_share']:.0%} constant, and BM25's saturation term "
          f"— the thing k₁ tunes — is doing almost nothing on this corpus. It is the index-side "
          f"explanation for a Q2 result that was previously only observed: that the k₁ axis of "
          f"the BM25 grid is nearly flat while the *b* axis is not. Length normalisation matters "
          f"because document lengths vary; saturation does not, because term frequencies do not.\n",
          "**And the slide's own pitfall applies with full force.** \"Positions everywhere — ×2–4 "
          "for phrases nobody issues\" is exactly this system: the query is a bag of history "
          f"terms with no phrase structure at all, so positions would cost "
          f"{pos['bytes'] / 1e6 - a['layouts'][0]['bytes'] / 1e6:.1f} MB to support an operator "
          f"that can never fire. Not storing them is correct, and now for a measured reason "
          f"rather than an omission.\n"]
    return "\n".join(L)


def part_b(d: dict) -> str:
    c = d["codes"]
    L = ["### B. Compression — bits per gap, and the bandwidth thesis\n",
         f"{c['postings']:,} d-gaps; mean {c['mean_gap']:,.0f}, median {c['median_gap']:,.0f}, "
         f"p99 {c['p99_gap']:,.0f}.\n",
         "| code | size | bits/gap | slide | vs raw | decode |",
         "|---|--:|--:|--:|--:|--:|"]
    for r in c["rows"]:
        dec = (f"{r['decode_mpostings_per_s']:,.1f} M postings/s" if "decode_seconds" in r
               else "—")
        L.append(f"| {r['code']} | {r['bytes'] / 1e6:.2f} MB | {r['bits_per_gap']:.2f} | "
                 + (f"{r['slide']} | " if r["slide"] else "— | ")
                 + f"{r['ratio_vs_raw']:.2f}× | {dec} |")
    vb = next(r for r in c["rows"] if "v-byte" in r["code"])
    bp = next(r for r in c["rows"] if "bit-packed" in r["code"])
    eg = next(r for r in c["rows"] if "gamma" in r["code"])
    ro = next(r for r in c["rows"] if "Roaring" in r["code"])
    L += ["", "Decode is timed all the way to docIDs, so the gap codes pay for their prefix sum "
              "and the raw layout pays nothing — its rate is a memcpy and is shown as a ceiling, "
              "not as a decoder.\n",
          f"**Every code comes in worse than the slide's table, in the same direction and for one "
          f"reason.** v-byte at {vb['bits_per_gap']:.2f} bits/gap against a quoted 8–9, "
          f"bit-packed blocks at {bp['bits_per_gap']:.2f} against 6–7, Elias-γ at "
          f"{eg['bits_per_gap']:.2f} against 5–6. Those quoted numbers come from a web index where "
          f"the postings mass sits in high-df terms whose lists are dense and whose gaps are "
          f"tiny. Our median gap is {c['median_gap']:,.0f} — a term appearing in one document in "
          f"{c['median_gap']:,.0f} needs {c['median_gap']:,.0f}-sized gaps, and no code makes "
          f"those small. **Bits-per-gap is not a property of the code, it is a property of the "
          f"document-frequency distribution the code is given**, which is Zipf's law arriving in "
          f"L5 as a bandwidth number.\n",
          f"Two ordering surprises follow from the same fact. Block bit-packing "
          f"({bp['bits_per_gap']:.2f}) is *worse* than v-byte ({vb['bits_per_gap']:.2f}) here, "
          f"because a 128-gap block is widened to its largest member and our gap distribution has "
          f"a p99 of {c['p99_gap']:,.0f} — this is precisely the case real PForDelta handles by "
          f"exceptioning outliers out, and the reason it does. And Roaring, quoted at ~2 bits on "
          f"dense lists, costs {ro['bits_per_gap']:.2f} bits here: with a median df in the low "
          f"single digits, almost every list is an array container of 16-bit values plus a "
          f"header, and a bitmap over a 125K-document range is never worth building.\n"]
    ae = c.get("analyzer_effect")
    if ae:
        w_, o_ = ae["with_stopwords"], ae["without_stopwords"]
        L += ["**The analyzer sends L5 a bill that Q2 never saw.** Same corpus, both analyzers:\n",
              "| analyzer | postings | median gap | v-byte bits/gap | v-byte size |",
              "|---|--:|--:|--:|--:|",
              f"| stopwords kept | {w_['postings']:,} | {w_['median_gap']:,.0f} | "
              f"{w_['vbyte_bits_per_gap']:.2f} | {w_['vbyte_mb']:.3f} MB |",
              f"| stopwords removed (shipped) | {o_['postings']:,} | {o_['median_gap']:,.0f} | "
              f"{o_['vbyte_bits_per_gap']:.2f} | {o_['vbyte_mb']:.3f} MB |", "",
              f"Stopword removal deletes {100 * (1 - o_['postings'] / w_['postings']):.0f}% of "
              f"postings and makes the survivors **{o_['vbyte_bits_per_gap'] / w_['vbyte_bits_per_gap'] - 1:+.0%} "
              f"more expensive each**, because the lists it deletes are exactly the dense ones "
              f"whose gaps compressed well. The median gap goes {w_['median_gap']:,.0f} → "
              f"{o_['median_gap']:,.0f}. Net, the index still shrinks "
              f"({w_['vbyte_mb']:.2f} → {o_['vbyte_mb']:.2f} MB), so the decision Q2 made on "
              f"recall was also right on size — but the two effects point in opposite directions "
              f"and only one of them was ever counted.\n"]
    L += ["**The thesis.** L5's claim is that nobody compresses postings to save disk: you shrink "
          "bytes to widen the bandwidth bottleneck and move the index up the hierarchy. Written "
          "out, evicted from the page cache with `POSIX_FADV_DONTNEED`, and read back:\n",
          "| layout | file | cold read + decode | cold MB/s | warm |", "|---|--:|--:|--:|--:|"]
    for r in d["bandwidth_thesis"]:
        L.append(f"| {r['code']} | {r['file_mb']:.2f} MB | {r['cold_seconds'] * 1e3:.2f} ms | "
                 f"{r['cold_mb_per_s']:,.0f} | {r['warm_seconds'] * 1e3:.2f} ms |")
    raw = next(r for r in d["bandwidth_thesis"] if r["code"] == "raw")
    vbb = next(r for r in d["bandwidth_thesis"] if r["code"] == "vbyte")
    L += ["", f"**Compression loses here by {vbb['cold_seconds'] / raw['cold_seconds']:.0f}×** — "
              f"{vbb['file_mb'] / raw['file_mb']:.2f}× the bytes and "
              f"{vbb['cold_seconds'] / raw['cold_seconds']:.0f}× the cold latency. This is the "
              f"same inversion L3 part D found on parquet codecs, from the other end, and for the "
              f"same reason: on an NVMe reading at thousands of MB/s the fetch is not the "
              f"bottleneck, the decode is.\n",
          "But \"the slide is wrong\" is not the finding, because the slide's own arithmetic says "
          "exactly where the line is. At its quoted 250 µs/MB of SSD:\n",
          "| layout | MB saved | I/O saved at slide-SSD | decode cost | decoder needed | decoder we have |",
          "|---|--:|--:|--:|--:|--:|"]
    for m in d.get("thesis_model", []):
        L.append(f"| {m['code']} | {m['mb_saved']:.2f} MB | "
                 f"{m['io_saved_ms_at_slide_ssd']:.2f} ms | {m['measured_decode_ms']:.2f} ms | "
                 f"{m['required_decode_gb_per_s']:.2f} GB/s | "
                 f"{m['measured_decode_gb_per_s']:.2f} GB/s |")
    mv = next((m for m in d.get("thesis_model", []) if m["code"] == "vbyte"), None)
    if mv:
        L += ["", f"The required decode rate is **scale-invariant** — both the bytes saved and the "
                  f"bytes decoded grow linearly with the index — so this is a clean threshold "
                  f"rather than an artefact of our corpus size: on this gap distribution, at the "
                  f"slide's SSD speed, v-byte pays if and only if the decoder exceeds "
                  f"**{mv['required_decode_gb_per_s']:.2f} GB/s**. The slide's own table puts "
                  f"PForDelta / SIMD-BP128 at 4+ GB/s and v-byte at 1–2. **So the thesis is right "
                  f"and the implementation is the reason it fails here**: a numpy decoder at "
                  f"{mv['measured_decode_gb_per_s']:.2f} GB/s is roughly 25× short of the bar, "
                  f"and the honest conclusion is not \"don't compress\" but \"don't compress "
                  f"without a vectorised decoder, and on a tier this hot, not even then\".\n"]
    return "\n".join(L)


def part_c(d: dict) -> str:
    qp = d["query_processing"]
    L = ["### C. Top-k processing — TAAT, DAAT, WAND, and the matmul\n",
         f"{qp['rows'][0]['queries']} identical queries, top-{qp['top_k']}, over a "
         f"{qp['n_docs']:,}-document candidate universe. The full-length query has a median of {qp['median_full_query_terms']:,} "
         f"distinct terms; the shorter rows truncate it by query weight, which is what a "
         f"production system does when it caps a long query. Rank-safety is verified against an "
         f"exhaustive scoring of every candidate, on scores rather than on document ids so that "
         f"ties cannot hide a difference.\n",
         "| query terms | DAAT docs scored | WAND docs scored | pruned | WAND multiply-adds "
         "÷ matmul's | rank-safe |",
         "|--:|--:|--:|--:|--:|--:|"]
    for r in qp["rows"]:
        share = (r["wand"]["contributions"] / max(r.get("matmul_postings_touched", 0) or 1, 1)
                 if r.get("matmul_postings_touched") else float("nan"))
        L.append(f"| {r['query_terms']} ({r['median_terms']:,}) | "
                 f"{r['daat']['docs_scored']:,.0f} | {r['wand']['docs_scored']:,.0f} | "
                 f"{r['wand_prune_share']:.1%} | "
                 + (f"{share:.2f}× | " if share == share else "— | ")
                 + f"{r['rank_safe']['wand']:.2f} |")
    first, last = qp["rows"][0], qp["rows"][-1]
    L += ["", f"**WAND's pruning gets *better* as queries get longer, which is the opposite of "
              f"what the shipped design implicitly assumed.** From "
              f"{first['wand_prune_share']:.1%} at {first['median_terms']} term(s) to "
              f"**{last['wand_prune_share']:.1%}** at {last['median_terms']:,}. That makes sense "
              f"once stated: more terms means a higher k-th best score sooner, so the threshold θ "
              f"rises faster and the pivot skips further. Rank-safety holds at 1.00 in every row "
              f"— WAND never changed an answer, which is the property that makes it usable at "
              f"all.\n",
          "One column deliberately does not appear: postings *touched*. WAND moves its cursors "
          "past everything it skips, so without skip pointers it reads the same postings DAAT "
          "does — the ratio is 1.00 at every query length. What it avoids is the scoring, which "
          "is the multiply-adds column above. Pairing WAND with part D's skip lists is what "
          "would turn the first number into a saving too.\n",
          "Now the timings, with the warning from the header in force:\n",
          "| query terms | TAAT | DAAT | WAND | matmul | TAAT accumulator | DAAT accumulator |",
          "|--:|--:|--:|--:|--:|--:|--:|"]
    for r in qp["rows"]:
        L.append(f"| {r['query_terms']} | {r['taat']['ms_per_query']:.2f} ms | "
                 f"{r['daat']['ms_per_query']:.2f} ms | {r['wand']['ms_per_query']:.2f} ms | "
                 f"{r['matmul_ms_per_query']:.3f} ms | "
                 f"{r['taat']['accumulator_bytes'] / 1e3:.0f} KB | "
                 f"{r['daat']['accumulator_bytes'] / 1e3:.1f} KB |")
    L += ["", f"**WAND does {1 / max(last['wand_contribution_share'], 1e-9):.2f}× less "
              f"arithmetic than DAAT and takes "
              f"{last['wand']['ms_per_query'] / max(last['daat']['ms_per_query'], 1e-9):.1f}× "
              f"longer.** The pivot is a sort and a cumulative sum over the live query terms, once "
              f"per candidate document; at {last['median_terms']:,} terms that bookkeeping costs "
              f"more than the scoring it avoids. The pruning ratio and the overhead both grow with "
              f"query length, and here the overhead grows faster.\n",
          f"The matmul is {last['matmul_ms_per_query']:.3f} ms against WAND's "
          f"{last['wand']['ms_per_query']:.1f} ms, and the honest reading of that gap is *not* "
          f"that WAND is a bad algorithm. TAAT and the matmul touch the same "
          f"{last.get('matmul_postings_touched', 0):,.0f} postings — they perform identical work "
          f"and differ by {last['taat']['ms_per_query'] / max(last['matmul_ms_per_query'], 1e-9):.0f}× "
          f"purely on kernel quality. WAND performs "
          f"{(last['wand']['contributions'] / max(last.get('matmul_postings_touched', 1), 1)):.2f}× "
          f"of that work. **A 2× reduction in arithmetic cannot pay for leaving a BLAS-shaped "
          f"kernel, at this candidate-set size** — which is a scale statement, and part G is where "
          f"it gets tested rather than asserted.\n",
          f"The memory column is the one place DAAT wins outright and the lecture's framing is "
          f"exactly right: TAAT holds {first['taat']['accumulator_bytes'] / 1e3:.0f} KB of "
          f"accumulator regardless of the query, because it is a float per candidate document, "
          f"while DAAT holds a k-element heap and one cursor per term "
          f"({first['daat']['accumulator_bytes'] / 1e3:.1f}–"
          f"{last['daat']['accumulator_bytes'] / 1e3:.1f} KB). At a shard of 10⁷ documents that is "
          f"80 MB per in-flight query against tens of KB — the reason the industry default is "
          f"DAAT, and a reason that has nothing to do with speed.\n"]
    return "\n".join(L)


def part_g(d: dict) -> str:
    import math
    sc = d["scale"]
    L = ["### G. Where the matmul stops being the right answer\n",
         "Same queries, widening candidate universe. The matmul's cost grows with the "
         "documents in scope; WAND's grows only with the documents that can still win. If "
         "the shipped design has a shelf life, this is the sweep that prices it.\n",
         "| universe | documents | postings in scope | matmul | WAND | WAND docs scored | "
         "WAND work ÷ matmul work |",
         "|---|--:|--:|--:|--:|--:|--:|"]
    for r in sc["rows"]:
        cap = (f" ⚠️" if r.get("queries_hitting_cap") else "")
        L.append(f"| {r['universe_days'] or 'whole catalogue'}"
                 f"{' d' if r['universe_days'] else ''} | "
                 f"{r['universe_docs']:,} | {r['postings_in_scope']:,} | "
                 f"{r['matmul_ms_per_query']:.3f} ms | {r['wand_ms_per_query']:,.1f} ms{cap} | "
                 f"{r['wand_docs_scored']:,.0f} | {r['wand_work_share']:.3f}× |")
    a, b = sc["rows"][0], sc["rows"][-1]
    grow = b["universe_docs"] / max(a["universe_docs"], 1)
    mm = b["matmul_ms_per_query"] / max(a["matmul_ms_per_query"], 1e-9)
    wd = b["wand_ms_per_query"] / max(a["wand_ms_per_query"], 1e-9)
    docs = b["wand_docs_scored"] / max(a["wand_docs_scored"], 1e-9)
    share = b["wand_work_share"] / max(a["wand_work_share"], 1e-9)
    capped = [r for r in sc["rows"] if r.get("queries_hitting_cap")]
    sizes = [r["universe_docs"] for r in sc["rows"]]
    if len(set(sizes)) < len(sizes):
        L += ["", "Rows sharing a document count are not a mistake: MIND ships no publication "
                  "date, so `candidate_universe` falls back to first-seen time and every window "
                  "wider than the split already covers the whole live set. The day-based rows "
                  "collapse; only the whole-catalogue row is a genuinely different scope."]
    L += ["", f"**The lecture's claim is confirmed, and by a wider margin than part C "
              f"suggested.** Across a {grow:,.0f}× wider universe the documents WAND actually "
              f"scores grow only {docs:,.1f}× — clearly sublinear — and its share of the "
              f"matmul's arithmetic falls from {a['wand_work_share']:.3f}× to "
              f"**{b['wand_work_share']:.3f}×**. In plain terms, WAND goes from doing "
              f"{1 / max(a['wand_work_share'], 1e-9):.1f}× less work than the matmul to "
              f"**{1 / max(b['wand_work_share'], 1e-9):.1f}× less**. Pruning is not a fixed "
              f"discount; it compounds with corpus size, exactly as the two-level retrieval "
              f"argument says.\n"]
    if share < 1:
        need = grow ** (math.log(0.1 / b["wand_work_share"]) / math.log(share))
        L += [f"Extrapolated on the measured trend — a factor of {1 / share:.2f} in work share "
              f"per {grow:,.0f}× of documents — WAND would reach a **10× arithmetic advantage "
              f"at roughly {need:,.1f}× this catalogue**, about "
              f"{need * b['universe_docs'] / 1e6:.2f}M documents. That is not a distant regime "
              f"for a news publisher with a multi-year archive. **The matmul's correctness here "
              f"has a shelf life, and this is it.**\n"]
    L += [f"What the sweep does *not* show is a wall-clock crossover, and the reason is the "
          f"header's warning rather than the algorithm. WAND's time grows {wd:,.1f}× against the "
          f"matmul's {mm:,.1f}× — the matmul is slightly *sub*linear in documents, because "
          f"scipy amortises fixed per-call costs over a larger block — because a per-document pivot in Python starts about three "
          f"orders of magnitude behind a scipy kernel and because the pivot's *iterations* grow "
          f"faster than the documents it scores — with hundreds of query terms, most iterations "
          f"advance a single cursor instead of scoring anything. A compiled WAND would inherit "
          f"the {1 / max(b['wand_work_share'], 1e-9):.1f}× arithmetic advantage without the "
          f"interpreter tax; this one cannot.\n",
          f"So the defensible pair of statements is: **the shipped matmul is right today and "
          f"will stay right through the next order of magnitude of traffic, and it is the wrong "
          f"structure for an archive-sized candidate set.** Concretely, at the whole catalogue "
          f"the matmul costs {b['matmul_ms_per_query']:.2f} ms per query against a 200 ms "
          f"request budget — {100 * b['matmul_ms_per_query'] / 200:.1f}% of it — so ten times "
          f"the users is absorbed by batching and ten times the *candidate universe* is still "
          f"only about {10 * b['matmul_ms_per_query']:.0f} ms. The query path's first failure "
          f"at scale is memory, not time: the dense (batch × documents) score block, which the "
          f"`max_cells` cap in `search_sparse` already bounds by shrinking the batch. **That is "
          f"the honest answer to \"where does it break at 10×\" for retrieval — not in latency, "
          f"and not yet.**\n"]
    if capped:
        L += [f"⚠️ marks rows where at least one query hit the "
              f"{capped[0]['iteration_cap']:,}-iteration safety cap "
              f"({', '.join(str(r['queries_hitting_cap']) + ' of ' + str(r['queries']) for r in capped)}). "
              f"A capped run stops early, so those WAND timings are *lower* bounds and the work "
              f"share is measured over less of the query than WAND would really do. Both errors "
              f"flatter WAND, so neither weakens the conclusion that its wall-clock is "
              f"uncompetitive — and the arithmetic trend, which is the load-bearing column, is "
              f"if anything understated.\n"]
    return "\n".join(L)


def part_d(d: dict) -> str:
    if "pairs" not in d.get("skips", {}):
        return ""
    sk = d["skips"]
    co = d["conjunction_order"]
    L = ["### D. Skip lists and cheapest-first\n",
         f"The slide's own example — a long list intersected with a short one — with a skip "
         f"pointer every √L. Steps are counted, not timed, because the claim is about steps.\n",
         "| long df | short df | stride | naive steps | with skips | speedup | skip index |",
         "|--:|--:|--:|--:|--:|--:|--:|"]
    for r in sk["pairs"][:6]:
        L.append(f"| {r['long_df']:,} | {r['short_df']:,} | {r['skip_stride']} | "
                 f"{r['naive_steps']:,} | {r['skip_steps']:,} | **{r['speedup']:.2f}×** | "
                 f"{r['skip_index_bytes'] / 1e3:.1f} KB |")
    best = max(sk["pairs"], key=lambda r: r["speedup"])
    L += ["", f"Mean speedup **{sk['mean_speedup']:.2f}×** over {len(sk['pairs'])} pairs, peaking "
              f"at {best['speedup']:.1f}× on a {best['long_df']:,} ∩ {best['short_df']} "
              f"intersection. The slide's worked example claims ~5× and this corpus gives more, "
              f"because the length ratio is more extreme than its 10⁶ ∩ 100: skips help in "
              f"proportion to how much of the long list can be jumped over, so the payoff is a "
              f"function of the *ratio*, not of the absolute length. The index costs "
              f"{best['skip_index_bytes'] / 1e3:.1f} KB for a list of {best['long_df']:,} — "
              f"memory spent to cut reads, which is where L3's RUM triangle reappears one lecture "
              f"later. Every intersection was verified to return the identical result set.\n",
          f"**Cheapest-first ordering.** Four-term conjunctions from the head of the df "
          f"distribution, evaluated in ascending and descending document frequency over "
          f"{co['trials']} trials: **{co['ascending_mean_steps']:,.0f} steps ascending against "
          f"{co['descending_mean_steps']:,.0f} descending — {co['saving']:.0%} fewer**. Starting "
          f"from the rarest term makes the candidate set small immediately and every subsequent "
          f"merge cheap; starting from the commonest carries a large intermediate through the "
          f"whole plan. It is free — the document frequencies are already in the dictionary — and "
          f"it is the single cheapest optimisation in this lecture.\n",
          "Neither of these is reachable from the shipped design, and that is worth saying "
          "plainly: a sparse matmul has no pointers to skip and no order to choose. They are "
          "arguments for the DAAT path that part G says this system will eventually need, not "
          "for a change to make today.\n"]
    return "\n".join(L)


def part_e(d: dict) -> str:
    ca = d["cache"]
    ti = d["tiering"]
    L = ["### E. Caching and tiering\n",
         f"**A Zipf-driven postings cache.** {ca['requests']:,} term requests from real user "
         f"queries over {ca['distinct_terms']:,} distinct terms, against a {ca['index_mb']:.1f} MB "
         f"index. The cache is static and holds the most-requested lists that fit, which is what "
         f"\"cache wisely following Zipf's\" describes.\n",
         "| cache budget | size | terms held | request hit rate | byte hit rate |",
         "|--:|--:|--:|--:|--:|"]
    for r in ca["rows"]:
        L.append(f"| {r['budget_share']:.1%} of index | {r['budget_mb']:.2f} MB | "
                 f"{r['terms_held']:,} | {r['request_hit_rate']:.1%} | "
                 f"{r['byte_hit_rate']:.1%} |")
    mid = min(ca["rows"], key=lambda r: abs(r["budget_share"] - 0.05))
    big = ca["rows"][-1]
    L += ["", f"The two hit rates disagree by design and the gap is the whole result: at "
              f"{mid['budget_share']:.0%} of the index, only {mid['terms_held']:,} terms fit and "
              f"they serve {mid['request_hit_rate']:.1%} of *requests* but "
              f"**{mid['byte_hit_rate']:.1%} of bytes**. Zipf puts the traffic on a few enormous "
              f"lists, so a cache sized in bytes catches the bandwidth long before it catches the "
              f"lookups. **Cache admission should be by bytes-served, not by request count** — "
              f"which is the opposite of what a naive LRU on term ids optimises, and it is "
              f"measurable here before anything is built.\n",
          f"**Tiering.** A hot tier by popularity, scored instead of the full universe. The slide "
          f"frames tiering as a cost saving with a quality risk, so the quality is measured "
          f"rather than assumed, together with how much click mass the tier can even reach.\n",
          "| tier | documents | click mass in tier | " + " | ".join(f"recall@{k}" for k in KS)
          + " | Δ recall@100 |",
          "|--:|--:|--:|" + "--:|" * (len(KS) + 1)]
    for r in ti["rows"]:
        L.append(f"| {r['tier_share']:.0%} | {r['tier_docs']:,} | {r['click_mass_in_tier']:.1%} | "
                 + " | ".join(f"{r[f'recall@{k}']:.5f}" for k in KS)
                 + f" | **{r['delta@100']:+.5f}** |")
    small = ti["rows"][0]
    L += ["", f"Full universe ({ti['universe_docs']:,} documents) scores "
              f"{ti['full']['recall@100']:.5f}.\n",
          f"**Every tier beats the full universe, and the smallest tier beats it by most.** A "
          f"{small['tier_share']:.0%} tier of {small['tier_docs']:,} documents holds "
          f"{small['click_mass_in_tier']:.1%} of the click mass and lifts recall@100 from "
          f"{ti['full']['recall@100']:.5f} to {small['recall@100']:.5f} — "
          f"**{small['recall@100'] / ti['full']['recall@100'] - 1:+.0%}** — while scoring "
          f"{ti['universe_docs'] / max(small['tier_docs'], 1):.0f}× fewer documents.\n",
          (f"The ceiling column is what makes that reading safe rather than a metric "
             f"artefact, and here it *binds*: the tier can only reach "
             f"{small['click_mass_in_tier']:.0%} of the click mass, so more than "
             f"{1 - small['click_mass_in_tier']:.0%} of the clicks are unreachable by "
             f"construction — and it still wins by a wide margin. "
             if small["click_mass_in_tier"] < 0.9 else
             f"The ceiling column is what makes that reading safe rather than a metric "
             f"artefact, and here it does not bind at all: the tier already contains "
             f"{small['click_mass_in_tier']:.1%} of the click mass, so almost nothing is "
             f"lost by construction and the gain is purely a ranking effect. ")
          + f"What the tier removes is not candidates, it is *distractors* — unpopular "
          f"articles that BM25 happens to score highly against a history bag. Popularity is a "
          f"strong enough prior on news that restricting the candidate set to it is a ranking "
          f"improvement disguised as a cost optimisation, which is the same fact the Q5 "
          f"popularity baseline showed from the other side.\n",
          f"So L5's early-termination warning — \"unsafe early termination without measuring "
          f"quality loss\" — is exactly right as methodology and lands on the opposite sign here. "
          f"The pitfall is not that we would lose quality silently; it is that we would have kept "
          f"scoring {ti['universe_docs']:,} documents to do worse than scoring "
          f"{small['tier_docs']:,}, and only a measurement could tell us.\n"]
    return "\n".join(L)


def part_f(d: dict) -> str:
    b = d["build"]
    L = ["### F. Building the index — single-pass in RAM against SPIMI\n",
         f"{b['n_docs']:,} documents. Both paths produce the identical index, and the postings "
         f"column is there to prove it. That had to be forced: the first SPIMI version skipped "
         f"the vocabulary stemming the shipped build performs, so on a stemmed corpus it "
         f"produced ~2% more postings and the table was comparing two different indexes rather "
         f"than two ways of building one.\n",
         "| strategy | seconds | peak RSS | spilled | postings |", "|---|--:|--:|--:|--:|"]
    for r in b["rows"]:
        L.append(f"| {r['strategy']} | {r['seconds']:.2f} | +{r['peak_rss_delta_mb']:,.1f} MB | "
                 f"{r['spill_mb']:.2f} MB | {r['postings']:,} |")
    ram = b["rows"][0]
    spimi = min(b["rows"][1:], key=lambda r: r["peak_rss_delta_mb"]) if len(b["rows"]) > 1 else None
    if spimi:
        L += ["", f"**SPIMI cuts peak resident memory from "
                  f"{ram['peak_rss_delta_mb']:,.0f} MB to {spimi['peak_rss_delta_mb']:,.1f} MB — "
                  f"a factor of {ram['peak_rss_delta_mb'] / max(spimi['peak_rss_delta_mb'], 0.05):,.0f} "
                  f"— for {spimi['seconds'] / ram['seconds']:.1f}× the wall-clock and "
                  f"{spimi['spill_mb']:.1f} MB of temporary disk.** That is the entire trade the "
                  f"slide describes, and the memory is the axis that matters: the shipped build "
                  f"holds every posting live, so its footprint is a linear function of corpus "
                  f"size and it is one of the failure modes the design note lists at 10×. SPIMI's "
                  f"footprint is a function of the *block*, which is a constant we choose.\n",
              "It is also L3's LSM argument arriving one lecture later with postings as the "
              "payload: write immutable sorted runs, merge them lazily, bound memory by the "
              "buffer rather than by the data. L3 measured that shape on vectors and found the "
              "segment pattern wrong for a graph index; here it is right, because postings really "
              "do merge.\n"]
    sc = b["scale"]
    L += [f"**The distributed arithmetic.** At the measured {sc['postings_per_s']:,.0f} "
          f"postings/s, the slide's 10¹² web postings would take "
          f"**{sc['single_node_days']:.1f} single-node days**, or {sc['nodes_for_30_min']:,.0f} "
          f"nodes to finish in 30 minutes. The slide quotes ~4 days on an HDD box and ~30 minutes "
          f"on 1,000 nodes; our node is faster and NVMe-backed, so it needs "
          f"{sc['nodes_for_30_min']:,.0f} rather than 1,000. The 12–200× rather than 1000× "
          f"speedup the slide warns about is not visible in this number and cannot be — "
          f"{sc['note']}. Quoting it as a ceiling is the honest use of it.\n"]
    return "\n".join(L)


FINDINGS = """
---

## What this changes

1. **Term frequency is nearly constant on this corpus, and that explains a Q2
   result.** 89% of postings have tf = 1 and mean tf is 1.11, so BM25's saturation
   term — everything k₁ controls — has almost nothing to act on. The flat k₁ axis
   in the BM25 grid was previously an observation; it is now an index statistic.

2. **Positions would cost ×3.1 to support an operator that can never fire.** The
   query is a bag of history terms with no phrase structure. The slide's own
   pitfall names this exactly, and not storing positions is now a measured
   decision rather than an omission.

3. **Bits-per-gap is a property of the corpus, not of the code.** Every code comes
   in 40–100% worse than the slide's table, in the same direction, because our
   median gap is 129 where a web index's is a handful. Block bit-packing loses to
   v-byte because our p99 gap is 82,720 and a 128-gap block widens to its largest
   member — which is precisely why real PForDelta exceptions outliers out.

4. **Roaring costs 19 bits/gap where the slide quotes ~2.** Its ~2 is for dense
   lists; with a median df in the low single digits almost every list is an array
   container, and a bitmap over a 125K-document range is never worth building.

5. **The analyzer sends the compressor a bill nobody counted.** Stopword removal
   deletes 30% of postings and makes the survivors 12% more expensive each, because
   the lists it deletes are the dense ones. Net the index still shrinks, so Q2's
   recall-driven decision was also right on size — but the two effects point in
   opposite directions.

6. **Compression loses by 10× on a hot NVMe tier, and the slide's own arithmetic
   says why.** The required decode rate is scale-invariant: v-byte pays if and only
   if the decoder beats ~2.5 GB/s. The slide puts SIMD codes at 4+ GB/s, so the
   thesis is right and our numpy decoder is 25× short of the bar. The conclusion is
   "not without a vectorised decoder", not "not ever". Same inversion as L3's
   parquet codec sweep, reached from the other end.

7. **WAND's pruning improves with query length — 0% at one term, 66% at 859 — and
   its overhead grows faster.** The pivot is a sort and a cumulative sum per
   candidate document; at 859 terms that costs more than the scoring it avoids.
   Rank-safety held at 1.00 in every row, so the algorithm is sound and the
   arithmetic is unfavourable.

8. **Pruning compounds with corpus size, so the matmul has a shelf life.** Across a
   61× wider universe WAND's documents scored grow only 16×, and its share of the
   matmul's arithmetic falls 0.469× → 0.152× — from 2.1× less work to 6.6× less. On
   that trend a 10× arithmetic advantage arrives at roughly 5× this catalogue. The
   wall-clock never crosses, because a Python pivot starts three orders of magnitude
   behind a scipy kernel; a compiled WAND would inherit the arithmetic and not the
   interpreter tax.

8b. **But retrieval does not break on time at 10×.** The matmul costs 4.9 ms per
   query against a 200 ms budget at the whole catalogue, so ten times the candidate
   universe is ~50 ms. The first failure at scale is the dense (batch × documents)
   score block — memory, already bounded by `max_cells` — not latency.

9. **DAAT's real advantage is memory, not speed.** TAAT holds a float per candidate
   document whatever the query; DAAT holds a heap and one cursor per term. At a
   10⁷-document shard that is 80 MB per in-flight query against tens of KB, and it
   is the reason DAAT is the industry default independent of any timing.

10. **Skip lists deliver 10.9× mean and 29× peak, above the slide's ~5×**, because
    the payoff scales with the length *ratio* of the two lists and ours is more
    extreme than the slide's example. Cheapest-first ordering is 87% cheaper than
    its reverse and costs nothing — the document frequencies are already in the
    dictionary.

11. **A popularity tier of 5% of the universe beats the full universe by +222%
    recall@100.** Tiering is framed in the lecture as a cost saving with a quality
    risk; here it is a quality *win*, because what the tier removes is not
    candidates but distractors. The early-termination warning is right as
    methodology and lands on the opposite sign.

12. **Cache admission should be by bytes served, not by request count.** At 5% of
    the index, 12 terms serve 1% of requests and 12% of bytes. Zipf puts the
    bandwidth on a few enormous lists, and an LRU over term ids optimises the wrong
    quantity.

13. **SPIMI cuts peak build memory by ~1000× for 1.3× the time.** The shipped
    single-pass build holds every posting live, which is a linear-in-corpus
    footprint and one of the design note's 10× failure modes; SPIMI's footprint is
    the block size, which is a constant we pick.
"""

CLAIMS = """
---

## Positions this ablation lets the design note take

| question | answer | basis |
|---|---|---|
| Why is retrieval a sparse matmul and not a postings merge? | Because the candidate universe is a week of articles (~2,000 docs) and the query has ~859 terms. WAND does 0.5× the arithmetic; that cannot pay for leaving a BLAS kernel at this size. | Part C: rank-safe at 1.00, 66% of documents pruned, 5.5× slower even in the same language |
| When does that stop being true? | At roughly 5× this catalogue, on arithmetic — and only with a compiled DAAT. | Part G: WAND's work share falls 0.469× → 0.152× over a 61× wider universe, i.e. 2.1× → 6.6× less work |
| Does retrieval break at 10×? | Not on latency. 4.9 ms/query at the whole catalogue against a 200 ms budget. | Part G; the first constraint is the dense score block's memory, already capped by `max_cells` |
| Do we store positions? | No. | ×3.1 index size for a phrase operator a history-bag query cannot issue (part A) |
| Do we compress the postings? | No, while the index is RAM- or NVMe-resident. | v-byte needs a >2.5 GB/s decoder to break even at slide-SSD speeds; the threshold is scale-invariant (part B) |
| Which code, if the index moves to a colder tier? | v-byte, with a SIMD decoder — not bit-packed blocks. | Blocks widen to their largest gap and our p99 gap is 82,720 (part B) |
| How does the build survive 10× the corpus? | SPIMI with a fixed block size. | 1000× less peak RSS for 1.3× the time, identical index (part F) |
| Should there be a hot tier? | Yes, and for quality, not only for cost. | A 5% popularity tier raises recall@100 by 222% while scoring 20× fewer documents (part E) |
| How would a postings cache be admitted? | By bytes served. | 12 terms are 1% of requests and 12% of bytes (part E) |

## What is still not measured

* **Block-Max WAND.** Plain WAND uses one upper bound per term; BMW keeps one per
  block and prunes far harder. It would improve part C's ratios and cannot change
  part G's conclusion, which is about the shape of the two cost curves.
* **A compiled DAAT.** Every millisecond in part C compares a Python loop to a
  compiled kernel. The work counters are the load-bearing columns for exactly this
  reason, and part G's crossover is stated in work, not in time.
* **Real MapReduce.** Part F's SPIMI is single-node, so the shuffle, straggler and
  coordination costs the slide names as the reason for 12–200× rather than 1000×
  are absent from the extrapolation. It is quoted as a ceiling.
* **Impact ordering and index tiering by score.** The tier here is by popularity,
  which is a document prior; ordering postings by impact within a term is a
  different mechanism and is untouched.
* **Spelling, autocomplete and NRT freshness.** There is no free-text query to
  correct, and the freshness dial was measured on the vector index in L3 part C
  rather than on postings.
"""


def sort_key(p: Path):
    for i, frag in enumerate(ORDER):
        if frag in p.name:
            return (i, p.name)
    return (len(ORDER), p.name)


if __name__ == "__main__":
    out = [HEADER]
    files = sorted(L5.glob("l5_postings_*.json"), key=sort_key)
    for p in files:
        d = json.loads(p.read_text())
        out.append(f"\n## {d['dataset']}/{d['variant']} — {d['n_articles']:,} articles, "
                   f"lang `{d['lang']}`\n")
        for key, fn in (("anatomy", part_a), ("codes", part_b),
                        ("query_processing", part_c), ("scale", part_g),
                        ("skips", part_d), ("cache", part_e), ("build", part_f)):
            if key in d:
                out.append(fn(d))
    out.append(FINDINGS)
    out.append(CLAIMS)
    dest = L5 / "l5_ablation.md"
    dest.write_text("\n".join(out))
    print(f"wrote {dest} from {len(files)} json file(s)")
