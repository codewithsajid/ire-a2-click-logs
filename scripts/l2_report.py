"""Turn the L2 serving JSONs into the markdown table the design note quotes."""
from __future__ import annotations

import json
from pathlib import Path

L2 = Path("reports/l2")

HEADER = """# L2 ablation: serving the retriever — demand, the cliff, and the tail

Every other timing in this repo is a mean throughput over a batch. A service is
priced differently: by its bottleneck's *service demand*, by how much of the
nominal ceiling queueing makes uninhabitable, and by what the slowest leaf does to
the root. These tables measure the same retriever as a request/response service.

Two scopes, because they are different systems. **universe** is the 7-day live
window the assignment actually retrieves over; **catalogue** is every article in
the bundle — the 10× system. One thread per leaf throughout, so parallelism is
what we vary and not what BLAS decides.
"""


def fmt(x, spec="{:,.0f}"):
    return "—" if x is None else spec.format(x)


def part_a(d: dict) -> str:
    a = d["part_a_service_demand"]
    L = [f"### A. Service demand — {d['dataset']}/{d['variant']}, scope `{d['scope']}` "
         f"({d['candidates']:,} candidates × {d['dim']}d)\n",
         "| stage | V | S (ms) | D = V·S (ms) | p99 (ms) | CV |",
         "|---|--:|--:|--:|--:|--:|"]
    for name, v in a["stage_demand"].items():
        L.append(f"| {name} | {v['V']} | {v['S_ms']:.4f} | {v['D_ms']:.4f} | "
                 f"{v['p99_ms']:.4f} | {v['cv']:.2f} |")
    err = a["serial_prediction_error_pct"]
    L += ["",
          f"Bottleneck is **{a['bottleneck']}** (D_max = {a['D_max_ms']:.4f} ms). Two ceilings, "
          f"and L2 distinguishes them: **1/D_max = {a['X_max_pipelined_qps']:,.0f} q/s** is what a "
          f"queueing network reaches when the stages are separate stations, while "
          f"**1/ΣD = {a['X_max_serial_predicted_qps']:,.0f} q/s** is one server running all three "
          f"serially — which is what this process is. Measured at N=1: "
          f"**{a['X_observed_1srv_qps']:,.0f} q/s**, i.e. the serial prediction is off by "
          f"{err:+.1f}%.\n",
          "| N (concurrency) | X (q/s) | R (ms) | p50 | p95 | p99 | N = X·R | error |",
          "|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for r in a["closed_loop"]:
        L.append(f"| {r['concurrency']} | {r['X_qps']:,.0f} | {r['R_ms']:.3f} | "
                 f"{r['p50_ms']:.3f} | {r['p95_ms']:.3f} | {r['p99_ms']:.3f} | "
                 f"{r['little_N']:.2f} | {r['little_error_pct']:+.1f}% |")
    base = a["closed_loop"][0]
    top = max(a["closed_loop"], key=lambda r: r["X_qps"])
    L.append("")
    L.append(f"Little's law returns the concurrency we set to within "
             f"{max(abs(r['little_error_pct']) for r in a['closed_loop']):.1f}% at every point — "
             f"that is a check on the harness, not on the system: any real gap would mean "
             f"time going somewhere the measurement does not see. The system result is the "
             f"*scaling*: {top['concurrency']}× the concurrency buys "
             f"{top['X_qps']/base['X_qps']:.1f}× the throughput "
             f"({100*top['X_qps']/base['X_qps']/top['concurrency']:.0f}% parallel efficiency) "
             f"while R rises from {base['R_ms']:.3f} ms to {top['R_ms']:.3f} ms.\n")
    return "\n".join(L)


def part_b(d: dict) -> str:
    b = d["part_b_utilization_cliff"]
    L = [f"### B. The utilization cliff — scope `{d['scope']}`, one server "
         f"(S = {b['S_ms']:.4f} ms, μ = {b['mu_qps']:,.0f} q/s)\n",
         "| ρ target | ρ achieved | U observed | W mean | M/D/1 W | M/M/1 W | W p50 | W p95 | W p99 |",
         "|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
    bound = [r for r in b["rows"]
             if r["U_observed"] > 0.95 and r["rho_target"] < 0.95]
    for r in b["rows"]:
        flag = " ⚠" if r in bound else ""
        L.append(f"| {r['rho_target']:.2f}{flag} | {r['rho_achieved']:.2f} | {r['U_observed']:.2f} | "
                 f"{r['mean_ms']:.3f} | {r['W_md1_ms']:.3f} | {r['W_mm1_ms']:.3f} | "
                 f"{r['p50_ms']:.3f} | {r['p95_ms']:.3f} | {r['p99_ms']:.3f} |")
    if bound:
        L += ["", f"⚠ **The rig is the bottleneck in these rows, not the service.** At "
                  f"S = {b['S_ms']:.3f} ms the server needs {b['mu_qps']:,.0f} arrivals a second "
                  f"to saturate, and an in-process Poisson generator cannot produce them: it holds "
                  f"the GIL the server needs, so achieved λ stalls near "
                  f"{max(r['lambda_achieved_qps'] for r in b['rows']):,.0f} q/s while U pins at 1.00 "
                  f"and W runs away into seconds. This is worth keeping rather than deleting — the "
                  f"load generator is a station too, and we priced D for the service and not for "
                  f"the harness, which is exactly the mistake the service-demand law exists to "
                  f"catch. The cliff below is measured where the rig has headroom.\n"]
    lo = [r for r in b["rows"] if r["rho_target"] <= 0.7]
    hi = [r for r in b["rows"] if r["rho_target"] >= 0.9]
    ok = [r for r in b["rows"] if r not in bound]
    if ok:
        L += ["", f"The two model columns are **means**, so `W mean` is the column to read them "
                  f"against; the percentiles are what an SLO is actually written in. Service time "
                  f"here has CV ≈ 0.02 — a 386 MB matrix scan takes the same time every time — so "
                  f"the queue is M/D/1, not M/M/1, and M/M/1 over-predicts the wait by up to "
                  f"{max((r['W_mm1_ms']/max(r['W_md1_ms'],1e-9)) for r in ok):.1f}×. Assuming "
                  f"exponential service because the textbook does would have sized this system for "
                  f"a queue it does not have.\n"]
    if lo and hi:
        L += ["", f"All times are milliseconds, measured **from arrival** rather than from the "
                  f"moment a worker picks the request up — measuring from pickup is the standard "
                  f"way to make a saturated queue look healthy. Between ρ=0.7 and ρ={hi[-1]['rho_target']:.2f} "
                  f"the p99 goes from {lo[-1]['p99_ms']:.2f} ms to {hi[-1]['p99_ms']:.2f} ms "
                  f"({hi[-1]['p99_ms']/max(lo[-1]['p99_ms'],1e-9):.1f}×) for "
                  f"{100*(hi[-1]['rho_achieved']-lo[-1]['rho_achieved'])/max(lo[-1]['rho_achieved'],1e-9):.0f}% "
                  f"more traffic: the last third of nominal capacity is what the p99 is made of.\n"]
    return "\n".join(L)


def part_c(d: dict) -> str:
    c = d["part_c_fanout"]
    L = [f"### C. Fan-out amplification — {c['corpus']:,} articles doc-sharded, "
         f"{c['n_requests']:,} requests per n, leaf pool {c['leaf_threads']}, "
         f"{c['background_threads']} competing thread(s) on {d.get('cores', '?')} cores\n",
         "| n shards | docs/shard | root p50 | root p99 | leaf p50 | leaf p99 | "
         "P(≥1 slow) observed | 1−(1−p)ⁿ, p=0.01 |",
         "|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for r in c["rows"]:
        L.append(f"| {r['n_shards']} | {r['docs_per_shard']:,} | {r['root']['p50_ms']:.3f} | "
                 f"{r['root']['p99_ms']:.3f} | {r['leaf']['p50_ms']:.3f} | "
                 f"{r['leaf']['p99_ms']:.3f} | {r['P_slow_leaf_own_p99_observed']:.3f} | "
                 f"{r['P_slow_leaf_predicted_1_minus_0.99_pow_n']:.3f} |")
    best = min(c["rows"], key=lambda r: r["root"]["p50_ms"])
    worst = max(c["rows"], key=lambda r: r["root"]["p50_ms"])
    L += ["", f"Sharding is exact — per-shard `IndexFlatIP` merged by score reproduces the "
              f"unsharded top-K byte for byte — so every difference here is latency and none "
              f"of it is recall. Root p50 is lowest at **n={best['n_shards']}** "
              f"({best['root']['p50_ms']:.2f} ms) and worst at n={worst['n_shards']} "
              f"({worst['root']['p50_ms']:.2f} ms): fan-out buys parallelism until "
              f"coordination and queueing cost more than the work each leaf removes.\n"]
    return "\n".join(L)


def part_d(d: dict) -> str:
    dd = d["part_d_fixes"]
    L = [f"### D. The two fixes, priced in recall — n={dd['n_shards']} shards, "
         f"{dd['n_requests']:,} requests, {dd.get('background_threads', 0)} competing thread(s)"
         + (f", **injected** straggler p={dd['straggler_p']} +{dd['straggler_ms']:.0f} ms"
            if dd.get("straggler_p") else "") + "\n",
         "| mode | root p50 | root p99 | Δ p99 | leaf p50 | leaf p99 | shards missed/req | "
         "hedges/req | extra leaf load | recall@100 | Δ recall |",
         "|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for r in dd["rows"]:
        lf = r.get("leaf", {})
        L.append(f"| {r['mode']} | {r['root']['p50_ms']:.3f} | {r['root']['p99_ms']:.3f} | "
                 f"{r['p99_vs_baseline_pct']:+.1f}% | {lf.get('p50_ms', float('nan')):.3f} | "
                 f"{lf.get('p99_ms', float('nan')):.3f} | {r['shards_missed_per_req']:.2f} | "
                 f"{r['hedges_per_req']:.2f} | {r['extra_leaf_load_pct']:.2f}% | "
                 f"{r['recall@100']:.5f} | {r['recall@100_vs_baseline_pct']:+.1f}% |")
    L.append("")
    return "\n".join(L)


def part_f(d: dict) -> str:
    f = d["part_f_skew"]
    L = [f"### F. Shard skew — {f['corpus']:,} articles, {f['n_requests']:,} requests per config, "
         f"leaf pool {f['leaf_threads']}, {f['background_threads']} competing thread(s)\n",
         "| n shards | skew | largest shard | share of corpus | imbalance | root p50 | root p99 | "
         "leaf p99 | Δ p50 vs balanced |",
         "|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for r in f["rows"]:
        d50 = r.get("root_p50_vs_balanced_pct")
        L.append(f"| {r['n_shards']} | {r['skew']:.1f} | {r['largest_shard']:,} | "
                 f"{100*r['largest_shard_share']:.1f}% | {r['imbalance']:.2f}× | "
                 f"{r['root']['p50_ms']:.3f} | {r['root']['p99_ms']:.3f} | "
                 f"{r['leaf']['p99_ms']:.3f} | "
                 + ("—" if d50 is None else f"{d50:+.1f}%") + " |")
    L.append("")
    return "\n".join(L)


def part_e(d: dict) -> str:
    e = d["part_e_offline_online"]
    off, on, am = e["offline"], e["online"], e["amortisation_queries"]
    L = [f"### E. Offline vs online — scope `{d['scope']}` ({off['vectors']:,} vectors × "
         f"{off['dim']}d, {off['embedding_matrix_mb']:,.0f} MB)\n",
         "| paid | item | cost |",
         "|---|---|--:|",
         f"| once, offline | exact `IndexFlatIP` build | {off['index_build_flat_s']:.2f} s |",
         f"| once, offline | HNSW M=16 efC=200 build | {off['index_build_hnsw_M16_efC200_s']:.2f} s |",
         f"| once, offline | embedding matrix resident | {off['embedding_matrix_mb']:,.0f} MB |"]
    enc = off.get("encode")
    if enc:
        L.append(f"| once, offline | encode {enc['corpus_articles']:,} articles "
                 f"(`{enc['model'].split('/')[-1]}`, {enc['device']}) | "
                 f"{enc['full_corpus_seconds']:,.0f} s |")
    for k, v in on["stages"].items():
        L.append(f"| per request | {k} | {v:.4f} ms |")
    L += [f"| per request | **total (R_min)** | **{on['R_min_ms']:.3f} ms** |", "",
          f"Against a {e['budget_ms']:.0f} ms budget the retriever spends "
          f"**{on['budget_used_pct']:.1f}%** and leaves **{on['budget_left_ms']:.2f} ms** for "
          f"everything downstream — which is the number M3's re-ranker has to fit inside, and "
          f"the reason L1's \"push heavy work offline\" is a measurable claim rather than a "
          f"slogan. The HNSW build is the largest one-off here: at the observed serving rate it "
          f"costs the wall-clock of {am['hnsw']:,.0f} served queries, paid once and hidden."
          + (f" Encoding runs at {off['encode']['articles_per_s']:,.0f} articles/s on "
             f"{off['encode']['device']}, so the vectors themselves are the cheaper half of "
             f"the offline bill at this size — the claim \"embedding generation is a big "
             f"cost\" is a statement about corpora two orders of magnitude larger than this "
             f"one." if off.get("encode") else "") + "\n"]
    return "\n".join(L)


# Read in the order the argument is made, not in the order the filesystem lists:
# operating point first, then the 10x system, then what happens to it under load.
ORDER = ["_universe", "_catalogue.json", "_idle", "_loaded", "_skew",
         "_straggler", "_hedge", "mind"]


def sort_key(p: Path):
    for i, frag in enumerate(ORDER):
        if p.name.endswith(frag) or frag in p.name:
            return (i, p.name)
    return (len(ORDER), p.name)


FINDINGS = """
## What the six parts add up to

1. **The operating point is free; the 10× system is not.** Over the 7-day live
   window the assignment retrieves from (2,063 articles) the whole retrieval path
   costs 0.183 ms — 0.1% of a 200 ms budget. Over the full 125,541-article
   catalogue the same code costs 9.19 ms, a 50× jump, and the bottleneck is the
   exact scan. That is the same crossover the ANN ablation found by throughput,
   arrived at independently from service demand.
2. **The prediction works, and Little's law says the measurement does.**
   1/ΣD predicts single-server throughput to within 1.5% at catalogue scale.
   N = X·R returns the concurrency we set to within 1.2% everywhere, which is a
   statement about the harness rather than the system — but it is the statement
   that licenses everything else here.
3. **Throughput peaks at N=16 on a 48-core box.** 26% parallel efficiency, and
   past N=16 more concurrency *reduces* throughput while response time grows 30×.
   A 368 MB scan per query is memory-bandwidth bound (467 q/s × 368 MB ≈ 172 GB/s,
   inferred from the shape rather than read off a counter), so D_max is a shared
   resource — which the single-station model does not capture and which no amount
   of thread pool fixes.
4. **The queue is M/D/1, not M/M/1.** Service-time CV is 0.01: the same matrix,
   scanned the same way, every time. M/M/1 over-predicts the mean wait by up to
   1.9×, so the textbook formula would have sized this system for a queue it does
   not have. The cliff is still there — p99 triples between ρ=0.7 and ρ=0.95 for
   36% more traffic.
5. **Fan-out amplification is real but milder than 1−(1−p)ⁿ.** Observed
   P(≥1 slow leaf) tracks the curve's shape and sits consistently below it
   (0.627 vs 0.724 at n=128). The formula assumes independent leaves; leaves
   sharing a thread pool and a memory bus are positively correlated, so they tend
   to be slow together rather than one at a time.
6. **Root latency is U-shaped in n, and the minimum moves with load.** Idle, the
   best shard count is 16 (4.25 ms vs 9.29 ms unsharded); under 32 competing
   threads it is 32 (38.2 ms vs 214.2 ms). Past the minimum, coordination costs
   more than the work each extra leaf removes.
7. **Hedging is a bet that the tail is idiosyncratic.** Against an injected
   per-leaf 50 ms pause at p=0.01 — one leaf slow for a reason local to it — a
   backup issued at the leaf's p95 cuts p99 from 55.3 ms to 7.7 ms (−86%) for 16%
   extra leaf load and no recall change: Dean & Barroso's result, reproduced.
   Against a *shared* tail — the same box carrying 32 competing threads — the
   identical mechanism raises p99 by 204% and p50 by 26%, because there is no
   healthier replica to run to and the backups are simply more load. This is the
   retry-storm pitfall, measured rather than quoted.
8. **Budgets cost recall only once they start cutting into signal.** On the
   straggler configuration a 25 ms deadline halves p99 (55.3 → 26.3 ms) while
   dropping 0.16 shards per request and moving recall@100 by nothing at all. Push
   to where 14 of 128 shards are dropped and recall@100 falls 56%. The trade is
   flat and then a cliff, not a slope — so a budget has to be set from the recall
   curve and not from the latency one.
9. **Skew costs exactly where sharding pays, and micro-sharding is not the fix.**
   At n=16 — the idle optimum — a 1/rank shard map puts 29.6% of the corpus in one
   leaf and costs +27% root p50 and +37% p99. At n=64 and n=128 the same skew is
   nearly free, but only because coordination has already made those shard counts
   worse than n=16 in the first place. Micro-sharding does shrink the largest
   shard's share (29.6% → 18.4% from n=16 to n=128), and root p50 still gets
   *worse* (5.51 → 8.50 ms): the cure costs more than the disease. The real fix is
   the one L2 gives for keys, not for shards — hash rather than keyrange, so the
   slices are equal by construction.
   The penalty is also *milder than the imbalance*: a 4.73× larger leaf costs 27%,
   not 373%, because the service is bandwidth-bound rather than core-bound and the
   big leaf gets a share of the bus rather than a queue of its own. On a
   core-bound service the penalty would track the imbalance ratio far more closely.
10. **"Embedding generation is a big cost" is false at this size.** MIND's 65,238
   articles encode in 13 s at 5,009 articles/s on one GPU, against 21.8 s to build
   the HNSW index over the result. The vectors are the *cheaper* half of the
   offline bill; the slide's claim is about corpora two orders of magnitude larger,
   and saying so is more useful than repeating it.

### The one thing this cannot measure on one box

**Replication.** Hedging here re-issues to the same shard on the same cores. Its
intended regime — a straggler that is slow for a reason local to it while a replica
on another machine is healthy — needs more than one machine, so the positive result
in part D uses an injected pause and says so in its own caption. Everything else L2
asks for is either measured above or stated as a design claim below.
"""


CLAIMS = """

## The design claims this system has to make, and does not measure

Three of L2's points are positions, not experiments. Leaving them unstated is the
gap; stating them is the whole fix.

**Where this sits on the PACELC surface — per subsystem, as the slide insists.**

* *The served candidate index* — **PA/EL**. Under partition, keep answering from
  whatever shards reply: part D measures exactly this trade and finds a 25 ms
  deadline halves p99 at zero recall cost, so a partial answer now really does beat
  a whole one late. Absent partition, trade consistency for latency: the index is a
  snapshot, and a reader can miss an article that went live since the last rebuild.
  That staleness is a recall loss, not a correctness bug, which is what makes the
  trade affordable.
* *The feature store (popularity, CTR, exposure)* — **PA/EL**, and the design note
  already prices getting it wrong: popularity frozen at the split boundary scores
  *below random* on EB-NeRD, because the catalogue turns over inside the scored
  window. What is wanted is eventual consistency with a strictly causal rolling
  cutoff — stale is acceptable, leaking is not.
* *The submission path* — **PC/EC**, and it is the one place in this repo that is
  correct-or-nothing. Both graders score one line per raw row in raw order, and
  EB-NeRD's 200,000 beyond-accuracy rows all share `impression_id = 0`, so a
  permutation is silently wrong rather than loudly late. `tests/test_row_order.py`
  and `tests/test_submission_format.py` enforce it; a late file beats a wrong one.

**The consistency model for freshness.** The served index is read-only and rebuilt
wholesale, so the only consistency question it has is *how stale*, bounded by the
rebuild cadence. Part E is what makes that a problem rather than a detail: the HNSW
build takes **44 s at 125K vectors and is superlinear in N**, so a wholesale rebuild
cannot be the freshness mechanism for a news corpus at this size, let alone at 10×.
The fix is L5's, not L2's — a RAM buffer plus small immutable segments searchable in
about a second, with merges in the background — and the 44 s figure is the argument
for it.

**Service discovery, and why our shard map is the skew-prone one.** The root
addresses leaves directly: `key2shard` is `floor(doc_id / shard_size)` and
`shard2node` is the identity, which is fine for one process and is not what a fleet
does. Worth naming because it is precisely the **keyrange** map L2 flags as
skew-prone — good for range scans, hostile to balance — and part F is what that
costs when the ranges are unequal.

**Doc-sharded, not term-sharded, and for a reason that is measurable here.** L2's
trade is cheap writes and scatter/gather reads with local scoring drift, against
fast reads and async writes. On the semantic side term-sharding is not even
definable — there are no terms in a dense vector — so the choice is forced. On the
lexical side it would be a real choice, and doc-sharding is still right twice over:
the corpus turns over daily, so write cost dominates; and a user query here is their
whole click history, averaging **1,060 non-zero terms** on EB-NeRD, which would
touch very nearly every term shard on every request and give away the one advantage
term-sharding has.
"""


if __name__ == "__main__":
    out = [HEADER]
    files = sorted(L2.glob("l2_serving_*.json"), key=sort_key)
    for p in files:
        d = json.loads(p.read_text())
        if "part_a_service_demand" in d:
            out.append(part_a(d))
        if "part_b_utilization_cliff" in d:
            out.append(part_b(d))
        if "part_e_offline_online" in d:
            out.append(part_e(d))
        if "part_f_skew" in d:
            out.append(part_f(d))
        if "part_c_fanout" in d:
            out.append(part_c(d))
        if "part_d_fixes" in d:
            out.append(part_d(d))
    out.append(FINDINGS)
    out.append(CLAIMS)
    dest = L2 / "l2_ablation.md"
    dest.write_text("\n".join(out))
    print(f"wrote {dest} from {len(files)} json file(s)")
