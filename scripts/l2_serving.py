"""L2 ablation: the serving side -- service demand, the utilization cliff, and the tail.

Every timing this repo reports so far is a *mean throughput* over a batch: "20,463
q/s", "113 s for 12.5M impressions". That is the right number for a batch job and
the wrong number for a service. L2's whole argument is that a retrieval system is
priced by its bottleneck's service demand, that queueing makes the last 30% of
nominal capacity uninhabitable, and that at fan-out the leaf's p99 becomes the
root's median. None of that is visible in a mean.

So this script turns the semantic retriever into an actual request/response service
and measures it the way L2 says to. Four parts, deliberately in the lecture's order:

A. **Service demand.** Time each pipeline stage in isolation to get S_i, count its
   visits V_i, and form D_i = V_i S_i. The largest D_i is the bottleneck and
   X_max = 1/D_max is the ceiling *no scheduling can beat*. Then drive the service
   to saturation with a closed loop and check the prediction against reality --
   including Little's law, N = X*R, which must return the concurrency we set.

B. **The utilization cliff.** Poisson arrivals into a single-server queue at
   rho = lambda/X_max from 0.3 to 0.95. Measure W's percentiles, and compare against
   M/M/1's W = S/(1-U). Our service time is nearly deterministic, so M/M/1 should
   *over*-predict the wait -- M/D/1 is the honest model, and the gap between the two
   is the value of knowing your service-time distribution.

C. **Fan-out amplification.** Doc-shard the 125K-article catalogue into n leaves
   (L2's "local doc-sharded" option, the one search engines actually use), scatter,
   and gather. The root waits for the slowest leaf, so P(slow root) = 1-(1-p)^n.
   Measured against real leaf tails, not a simulation.

D. **The two fixes, priced in recall.** Hedged requests (re-issue after the leaf's
   p95) and per-leaf budgets with partial results. Hedging costs load; budgets cost
   recall. Both are measured on the assignment's own metric, so the trade is stated
   in the units we are graded in rather than in milliseconds alone.

Sharding is exact (IndexFlatIP per shard, merged), so the sharded root returns
*identical* results to the unsharded index. Any latency difference is therefore
attributable to fan-out alone, and any recall difference in part D is attributable
to the budget alone. Threads, not processes: FAISS and BLAS both release the GIL
inside search, which is where all the time goes.
"""
from __future__ import annotations

import os

# Pin every math runtime to one thread *before* numpy loads. Two reasons, and the
# second one is not optional: (a) the whole point is that one thread serves one
# leaf, so an inner BLAS pool would silently parallelise what we are trying to
# count; (b) OpenBLAS allocates a buffer per thread it has ever seen and refuses
# past its cap -- "Program is Terminated. Because you tried to allocate too many
# memory regions" -- which is what the closed-loop sweep triggers when it walks
# concurrency up to 64. threadpool_limits() alone is too late: the pool is sized
# at import.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import math
import random
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor
from concurrent.futures import wait as futures_wait
from pathlib import Path

import numpy as np

from newsrec.embeddings import article_embeddings
from newsrec.retrieval import (candidate_universe_for_split, evaluate_recall,
                               summarise, user_histories)
from newsrec.semantic import l2_normalise, user_vectors
from newsrec.store import FeatureStore

KS = (50, 100, 200)
TOPK = 200
# Padding for short result lists. A dropped shard genuinely leaves fewer candidates
# in the top-K, so the slot must still occupy its rank -- it just can never match a
# clicked article. UInt32 max is the sentinel because the store's article index is
# UInt32 and the evaluator refuses a negative.
NO_DOC = np.iinfo(np.uint32).max


# ---------------------------------------------------------------- helpers

def pct(xs, q: float) -> float:
    """Percentile in milliseconds, nearest-rank -- p99 of 100 samples is the 99th."""
    a = np.sort(np.asarray(xs, dtype=np.float64))
    if a.size == 0:
        return float("nan")
    i = min(a.size - 1, max(0, int(math.ceil(q / 100.0 * a.size)) - 1))
    return float(a[i] * 1e3)


def stats_ms(xs) -> dict:
    a = np.asarray(xs, dtype=np.float64)
    return {"n": int(a.size), "mean_ms": round(float(a.mean()) * 1e3, 4),
            "p50_ms": round(pct(a, 50), 4), "p95_ms": round(pct(a, 95), 4),
            "p99_ms": round(pct(a, 99), 4), "p999_ms": round(pct(a, 99.9), 4),
            "max_ms": round(float(a.max()) * 1e3, 4),
            "cv": round(float(a.std() / max(a.mean(), 1e-12)), 4)}


def pin_threads(n: int):
    """One thread per leaf, so fan-out is what we measure and not BLAS's own pool."""
    import faiss
    faiss.omp_set_num_threads(n)
    try:
        from threadpoolctl import threadpool_limits
        return threadpool_limits(limits=n)
    except ImportError:
        return None


class BackgroundLoad:
    """Competing traffic, so the leaf tail is queueing and not imagination.

    On an idle 48-core box a 1,300-candidate exact search has p99 ~ p50: there is
    no tail, so nothing for hedging or a budget to fix, and reporting that as
    "hedging does not help" would be an artefact of the bench rather than a result.
    L2 is explicit that the tail *comes from* utilization, so the honest way to
    observe one is to put the machine under load and measure -- not to inject a
    synthetic straggler.
    """

    def __init__(self, vectors: np.ndarray, n_threads: int, dim: int):
        self.vectors = vectors
        self.n_threads = n_threads
        self.dim = dim
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []

    def _spin(self, seed: int):
        rng = np.random.default_rng(seed)
        ix = flat_index(self.vectors)
        q = np.ascontiguousarray(l2_normalise(
            rng.standard_normal((1, self.dim)).astype(np.float32)))
        while not self._stop.is_set():
            ix.search(q, 200)

    def __enter__(self):
        for t in range(self.n_threads):
            th = threading.Thread(target=self._spin, args=(t,), daemon=True)
            th.start()
            self._threads.append(th)
        if self.n_threads:
            time.sleep(0.5)                              # let the load settle
        return self

    def __exit__(self, *exc):
        self._stop.set()
        for th in self._threads:
            th.join(timeout=2.0)
        return False


def flat_index(vectors: np.ndarray):
    import faiss
    ix = faiss.IndexFlatIP(vectors.shape[1])
    ix.add(np.ascontiguousarray(vectors))
    return ix


# ---------------------------------------------------------------- the service

class RetrievalService:
    """One request = one user's top-K. The four stages L2 wants priced separately.

    Deliberately *not* batched. The batch matmul the rest of the repo uses is the
    right shape for an offline sweep and is unavailable to a service that must
    answer one user at a time inside a latency budget -- which is exactly the
    offline/online distinction L1 draws and this script measures.
    """

    def __init__(self, emb: np.ndarray, universe: np.ndarray, histories: list[np.ndarray],
                 top_k: int = TOPK, build_index: bool = True, sub: np.ndarray | None = None):
        self.emb = emb
        self.universe = universe
        # `sub` is shareable across services: it is 386 MB at catalogue scale and
        # rebuilding it per shard count was most of part C's wall time.
        self.sub = sub if sub is not None else np.ascontiguousarray(
            l2_normalise(emb[universe].astype(np.float32)))
        self.index = flat_index(self.sub) if build_index else None
        self.hist = histories
        self.top_k = top_k
        self.seen = [set(h.tolist()) for h in histories]

    # ---- stages -------------------------------------------------------

    def s_user_vector(self, i: int) -> np.ndarray:
        h = self.hist[i]
        if h.size == 0:
            return np.zeros((1, self.emb.shape[1]), dtype=np.float32)
        v = self.emb[h[h < self.emb.shape[0]]]
        if v.shape[0] == 0:
            return np.zeros((1, self.emb.shape[1]), dtype=np.float32)
        q = v.mean(0, keepdims=True)
        return np.ascontiguousarray(l2_normalise(q.astype(np.float32)))

    def s_search(self, q: np.ndarray):
        return self.index.search(q, self.top_k)

    def s_drop_seen(self, idx: np.ndarray, i: int) -> np.ndarray:
        seen = self.seen[i]
        keep = [d for d in self.universe[idx[0]].tolist() if d not in seen]
        return np.asarray(keep[:self.top_k], dtype=np.int64)

    def handle(self, i: int) -> np.ndarray:
        q = self.s_user_vector(i)
        _, idx = self.s_search(q)
        return self.s_drop_seen(idx, i)


class ShardedService(RetrievalService):
    """Doc-sharded scatter/gather: n leaves, each an exact index over its slice.

    Merging exact per-shard top-K reproduces the unsharded exact top-K exactly, so
    the only thing sharding changes here is *when* the answer arrives.
    """

    def __init__(self, emb, universe, histories, n_shards: int, top_k: int = TOPK,
                 pool: ThreadPoolExecutor | None = None, sub: np.ndarray | None = None,
                 straggler_p: float = 0.0, straggler_ms: float = 0.0,
                 bounds: np.ndarray | None = None):
        super().__init__(emb, universe, histories, top_k, build_index=False, sub=sub)
        # An *idiosyncratic* straggler: one leaf pausing for a reason local to it
        # (a GC, a compaction, a descheduled thread), which is the regime hedging
        # was designed for. It cannot be produced by loading a single box -- there
        # every leaf slows together -- so when this is non-zero the tail is
        # injected, and every table built from it says so.
        self.straggler_p = straggler_p
        self.straggler_s = straggler_ms / 1e3
        if bounds is None:
            bounds = np.linspace(0, self.sub.shape[0], n_shards + 1).astype(int)
        self.slices = [(int(bounds[j]), int(bounds[j + 1])) for j in range(len(bounds) - 1)]
        self.n_shards = len(self.slices)
        self.leaves = [flat_index(self.sub[a:b]) for a, b in self.slices]
        self.pool = pool
        self.leaf_times: list[float] = []
        self._lock = threading.Lock()

    def _leaf(self, j: int, q: np.ndarray, k: int, deadline: float | None,
              sink: list | None = None):
        """One leaf probe. Returns None when the request's budget has already
        expired -- a leaf that starts late is a leaf whose answer is worthless.

        `sink` collects this request's own leaf latencies. Per-request is the
        granularity the tail-at-scale argument needs: "P(at least one slow leaf)"
        is a statement about the leaves of one query, not about the leaf
        population, and a global list cannot answer it.
        """
        if deadline is not None and time.perf_counter() >= deadline:
            return None
        t0 = time.perf_counter()
        a, _ = self.slices[j]
        s, i = self.leaves[j].search(q, min(k, self.leaves[j].ntotal))
        if self.straggler_p and random.random() < self.straggler_p:
            time.sleep(self.straggler_s)
        dt = time.perf_counter() - t0
        self.leaf_times.append(dt)                       # list.append is atomic
        if sink is not None:
            sink.append(dt)
        return s[0], i[0] + a

    def scatter_gather(self, q: np.ndarray, k: int, deadline: float | None = None,
                       hedge_after: float | None = None):
        """Root fan-out. `deadline` caps the whole request (partial results allowed);
        `hedge_after` re-issues a still-outstanding leaf once it passes that delay."""
        pool = self.pool
        t_start = time.perf_counter()
        sink: list[float] = []
        self.last_leaf_times = sink
        pending = {pool.submit(self._leaf, j, q, k, deadline, sink): j
                   for j in range(self.n_shards)}
        answered: dict[int, tuple] = {}
        hedged = 0
        while pending:
            # block until something finishes, or until the next scheduled event --
            # the request deadline, or the moment a hedge is due.
            events = [t for t in (deadline,
                                  None if hedge_after is None else t_start + hedge_after)
                      if t is not None]
            timeout = max(0.0, min(events) - time.perf_counter()) if events else None
            done, _ = futures_wait(set(pending), timeout=timeout,
                                   return_when=FIRST_COMPLETED)
            for f in done:
                j = pending.pop(f)
                res = f.result()
                if res is not None and j not in answered:
                    answered[j] = res
            # First reply wins: once a shard has answered, stop waiting on its
            # twin. Without this the hedge is pure overhead -- the backup returns
            # early and the root still blocks on the straggler it was meant to
            # route around, which is a silent way to measure "hedging does not
            # work" when what does not work is the gather.
            for f, j in list(pending.items()):
                if j in answered:
                    f.cancel()
                    pending.pop(f)
            if not pending or len(answered) == self.n_shards:
                break
            now = time.perf_counter()
            if deadline is not None and now >= deadline:
                # give up on the stragglers: a partial answer now beats a whole one late
                for f in pending:
                    f.cancel()
                break
            if hedge_after is not None and (now - t_start) >= hedge_after:
                # one backup per outstanding leaf, first reply wins (Dean & Barroso)
                for f, j in list(pending.items()):
                    pending[pool.submit(self._leaf, j, q, k, None, sink)] = j
                    hedged += 1
                hedge_after = None
        missing = self.n_shards - len(answered)
        got = list(answered.values())
        if not got:
            return np.zeros(0, dtype=np.int64), missing, hedged
        scores = np.concatenate([g[0] for g in got])
        ids = np.concatenate([g[1] for g in got])
        order = np.argsort(-scores)[:k]
        return ids[order], missing, hedged

    def handle_sharded(self, i: int, deadline_s: float | None = None,
                       hedge_after: float | None = None):
        q = self.s_user_vector(i)
        ids, missing, hedged = self.scatter_gather(q, self.top_k, 
                                                   None if deadline_s is None else time.perf_counter() + deadline_s,
                                                   hedge_after)
        seen = self.seen[i]
        keep = [d for d in self.universe[ids].tolist() if d not in seen][:self.top_k]
        return np.asarray(keep, dtype=np.int64), missing, hedged


# ---------------------------------------------------------------- part A

def part_a(svc: RetrievalService, qidx: np.ndarray, concurrencies: list[int],
           seconds: float, reps: int) -> dict:
    """Service demand from measured stage times, then a saturation test."""
    print(f"\n== A. service demand | {svc.sub.shape[0]:,} candidates x {svc.sub.shape[1]}d "
          f"| top-{svc.top_k}")
    rng = np.random.default_rng(0)
    sample = rng.choice(qidx, size=min(reps, qidx.size), replace=False)

    # ---- S_i per stage, in isolation. V_i = 1: every request visits every stage once.
    # Warm every sampled user, not just the first few. The embedding matrix is
    # 386 MB at catalogue scale and a user vector is a random gather into it, so a
    # once-only visit measures the cache miss rather than the stage; the closed
    # loop below revisits the same users constantly and would then disagree with
    # its own prediction by 3x for no reason but cache temperature.
    t_uv, t_se, t_ds = [], [], []
    for _ in range(2):
        for i in sample:
            svc.handle(int(i))
    for i in sample:
        i = int(i)
        t0 = time.perf_counter(); q = svc.s_user_vector(i); t_uv.append(time.perf_counter() - t0)
        t0 = time.perf_counter(); _, idx = svc.s_search(q);  t_se.append(time.perf_counter() - t0)
        t0 = time.perf_counter(); svc.s_drop_seen(idx, i);   t_ds.append(time.perf_counter() - t0)

    stages = {"user_vector": t_uv, "ann_search": t_se, "drop_seen": t_ds}
    demand = {}
    for name, ts in stages.items():
        s = float(np.mean(ts))
        demand[name] = {"V": 1, "S_ms": round(s * 1e3, 4), "D_ms": round(s * 1e3, 4),
                        **{k: v for k, v in stats_ms(ts).items() if k in ("p50_ms", "p99_ms", "cv")}}
    d_max_name = max(demand, key=lambda k: demand[k]["D_ms"])
    d_max = demand[d_max_name]["D_ms"] / 1e3
    d_total = sum(v["D_ms"] for v in demand.values()) / 1e3
    x_max_pred = 1.0 / d_max
    for name, v in demand.items():
        print(f"  {name:14s} S={v['S_ms']:8.4f} ms  D={v['D_ms']:8.4f} ms  "
              f"p99={v['p99_ms']:8.4f} ms  CV={v['cv']:.2f}")
    # Two different ceilings, and L2 is careful about the difference:
    #   1/D_max  -- the queueing-network ceiling, reachable only if the stages are
    #               separate stations that can work on different requests at once
    #   1/sum(D) -- one station running all three stages serially, which is what
    #               this process actually is
    x_serial = 1.0 / d_total
    print(f"  bottleneck = {d_max_name} (D_max={d_max*1e3:.4f} ms) -> "
          f"X_max = 1/D_max = {x_max_pred:,.0f} q/s (pipelined stations)")
    print(f"  R_min = sum D = {d_total*1e3:.4f} ms -> "
          f"1/sum(D) = {x_serial:,.0f} q/s (one serial server -- this process)")

    # ---- closed loop: raise concurrency until throughput stops rising
    loop = []
    for c in concurrencies:
        X, R, lat = closed_loop(svc, qidx, c, seconds)
        n_little = X * R
        loop.append({"concurrency": c, "X_qps": round(X, 1), "R_ms": round(R * 1e3, 4),
                     "little_N": round(n_little, 2),
                     "little_error_pct": round(100 * (n_little - c) / c, 2),
                     **stats_ms(lat)})
        print(f"  N={c:3d}  X={X:9,.0f} q/s  R={R*1e3:8.3f} ms  "
              f"p99={pct(lat,99):8.3f} ms  Little N=X*R={n_little:6.2f} "
              f"({100*(n_little-c)/c:+.1f}%)")
    x_obs = max(r["X_qps"] for r in loop)
    x_1 = [r for r in loop if r["concurrency"] == 1][0]["X_qps"]
    return {"stage_demand": demand, "bottleneck": d_max_name,
            "D_max_ms": round(d_max * 1e3, 4), "D_total_ms": round(d_total * 1e3, 4),
            "X_max_pipelined_qps": round(x_max_pred, 1),
            "X_max_serial_predicted_qps": round(x_serial, 1),
            "X_observed_1srv_qps": round(x_1, 1),
            "serial_prediction_error_pct": round(100 * (x_1 - x_serial) / x_serial, 2),
            "X_max_observed_qps": round(x_obs, 1),
            "speedup_vs_1_server": round(x_obs / max(x_1, 1e-9), 2),
            "closed_loop": loop}


def closed_loop(svc: RetrievalService, qidx: np.ndarray, concurrency: int, seconds: float):
    """N fixed, measure X and R -- the Slurm-shaped regime: capacity fixed, W floats."""
    stop = time.perf_counter() + seconds
    out, lock = [], threading.Lock()

    def worker(wid: int):
        rng = random.Random(wid)
        local = []
        while time.perf_counter() < stop:
            i = int(qidx[rng.randrange(qidx.size)])
            t0 = time.perf_counter()
            svc.handle(i)
            local.append(time.perf_counter() - t0)
        with lock:
            out.extend(local)

    t0 = time.perf_counter()
    with ThreadPoolExecutor(concurrency) as ex:
        list(ex.map(worker, range(concurrency)))
    wall = time.perf_counter() - t0
    lat = np.asarray(out)
    return len(lat) / wall, float(lat.mean()), lat


# ---------------------------------------------------------------- part B

def part_b(svc: RetrievalService, qidx: np.ndarray, s_mean: float,
           rhos: list[float], seconds: float) -> dict:
    """Open-loop Poisson arrivals into one server -- the K8s-shaped regime.

    Response time W = queue wait + service, measured from *arrival*, not from the
    moment a worker picks the request up. Measuring from pickup is the classic way
    to make a saturated queue look healthy.
    """
    mu = 1.0 / s_mean
    print(f"\n== B. utilization cliff | 1 server | S={s_mean*1e3:.4f} ms -> mu={mu:,.0f} q/s")
    rows = []
    for rho in rhos:
        lam = rho * mu
        w, dropped, u_obs, lam_obs = open_loop(svc, qidx, lam, seconds)
        if w.size < 30:
            print(f"  rho={rho:.2f}  too few samples ({w.size}) -- skipped")
            continue
        # M/M/1 (exponential service) vs M/D/1 (deterministic) -- the two textbook
        # bounds our near-deterministic service should fall between.
        w_mm1 = s_mean / max(1 - rho, 1e-6)
        w_md1 = s_mean * (1 + rho / (2 * max(1 - rho, 1e-6)))
        rows.append({"rho_target": rho, "lambda_target_qps": round(lam, 1),
                     "lambda_achieved_qps": round(lam_obs, 1),
                     "rho_achieved": round(lam_obs / mu, 4),
                     "U_observed": round(u_obs, 4), "dropped": dropped,
                     "W_mm1_ms": round(w_mm1 * 1e3, 4), "W_md1_ms": round(w_md1 * 1e3, 4),
                     **stats_ms(w)})
        print(f"  rho={rho:.2f} (achieved {lam_obs/mu:.2f}, U={u_obs:.2f}) "
              f"W p50={pct(w,50):8.3f} p99={pct(w,99):9.3f} ms | "
              f"M/D/1 {w_md1*1e3:8.3f}  M/M/1 {w_mm1*1e3:9.3f} ms")
    return {"S_ms": round(s_mean * 1e3, 4), "mu_qps": round(mu, 1), "rows": rows}


def open_loop(svc: RetrievalService, qidx: np.ndarray, rate: float, seconds: float,
              max_queue: int = 20000):
    """Poisson arrivals, one server, FIFO. Returns response times from arrival."""
    import queue
    q: "queue.Queue" = queue.Queue(maxsize=max_queue)
    done, dropped = [], [0]
    busy = [0.0]

    def server():
        while True:
            item = q.get()
            if item is None:
                break
            i, t_arr = item
            t0 = time.perf_counter()
            svc.handle(i)
            t1 = time.perf_counter()
            busy[0] += t1 - t0
            done.append(t1 - t_arr)

    th = threading.Thread(target=server, daemon=True)
    th.start()
    rng = random.Random(7)
    t_start = time.perf_counter()
    t_end = t_start + seconds
    nxt = t_start
    while True:
        now = time.perf_counter()
        if now >= t_end:
            break
        if nxt > now:
            d = nxt - now
            if d > 5e-4:                                    # sleep coarsely, spin finely
                time.sleep(d * 0.8)
            continue
        i = int(qidx[rng.randrange(qidx.size)])
        try:
            q.put_nowait((i, time.perf_counter()))
        except Exception:
            dropped[0] += 1
        nxt += rng.expovariate(rate)
    q.put(None)
    th.join()
    wall = time.perf_counter() - t_start
    d = np.asarray(done)
    return d, dropped[0], busy[0] / wall, d.size / wall


# ---------------------------------------------------------------- part C

def part_c(emb: np.ndarray, universe: np.ndarray, hists: list[np.ndarray],
           qidx: np.ndarray, shard_counts: list[int], n_req: int,
           leaf_threads: int, bg_threads: int) -> dict:
    """Fan-out amplification: root latency vs leaf latency as n grows.

    The measurement that matters is *per request*: fix p as the per-leaf
    probability of being slow, and count the requests in which at least one leaf
    crossed that bar. Comparing root latency to the leaf population's p99 -- the
    obvious thing to do -- is not the same statement and saturates at 1.0 for any
    n above about 4, because the root is a max over n draws by construction.

    Two thresholds, because they answer different questions:
      * each n against **its own** leaf p99  -> tests 1-(1-p)^n directly, p=0.01
      * every n against the **n=1** leaf p99 -> what a fixed SLO actually sees as
        the system is sharded finer
    """
    print(f"\n== C. fan-out | corpus {universe.size:,} | {n_req:,} requests per n "
          f"| leaf pool {leaf_threads} | background load {bg_threads} thread(s)")
    rows = []
    rng = np.random.default_rng(1)
    sample = rng.choice(qidx, size=min(n_req, qidx.size), replace=False)
    sub = np.ascontiguousarray(l2_normalise(emb[universe].astype(np.float32)))
    ref_leaf_p99 = None
    with BackgroundLoad(sub[: min(len(sub), 20000)], bg_threads, sub.shape[1]), \
         ThreadPoolExecutor(leaf_threads) as pool:
        for n in shard_counts:
            svc = ShardedService(emb, universe, hists, n, pool=pool, sub=sub)
            for i in sample[:32]:
                svc.handle_sharded(int(i))                  # warm
            svc.leaf_times.clear()
            root, per_req = [], []
            for i in sample:
                t0 = time.perf_counter()
                svc.handle_sharded(int(i))
                root.append(time.perf_counter() - t0)
                per_req.append(list(svc.last_leaf_times))
            leaf = np.asarray(svc.leaf_times)
            r = np.asarray(root)
            worst = np.asarray([max(x) if x else np.nan for x in per_req])
            own_p99 = pct(leaf, 99) / 1e3
            if ref_leaf_p99 is None:
                ref_leaf_p99 = own_p99
            # the slide's arithmetic: per-leaf P(slow)=1% -> P(>=1 slow) = 1-(1-p)^n
            pred = 1 - 0.99 ** n
            obs_own = float(np.nanmean(worst > own_p99))
            obs_ref = float(np.nanmean(worst > ref_leaf_p99))
            rows.append({"n_shards": n, "docs_per_shard": int(universe.size / n),
                         "root": stats_ms(r), "leaf": stats_ms(leaf),
                         "root_p50_over_leaf_p50": round(pct(r, 50) / max(pct(leaf, 50), 1e-9), 2),
                         "P_slow_leaf_own_p99_observed": round(obs_own, 4),
                         "P_slow_leaf_predicted_1_minus_0.99_pow_n": round(pred, 4),
                         "P_slow_leaf_vs_n1_p99_observed": round(obs_ref, 4),
                         "leaf_p99_ms_reference_n1": round(ref_leaf_p99 * 1e3, 4)})
            print(f"  n={n:4d} ({int(universe.size/n):>7,}/shard)  root p50={pct(r,50):8.3f} "
                  f"p99={pct(r,99):9.3f} | leaf p50={pct(leaf,50):7.3f} p99={pct(leaf,99):8.3f} ms "
                  f"| P(>=1 slow) own {obs_own:.3f} vs 1-(0.99)^n {pred:.3f} "
                  f"| vs n=1 bar {obs_ref:.3f}")
    return {"corpus": int(universe.size), "n_requests": int(sample.size),
            "leaf_threads": leaf_threads, "background_threads": bg_threads, "rows": rows}


# ---------------------------------------------------------------- part D

def part_d(fs: FeatureStore, split: str, emb: np.ndarray, universe: np.ndarray,
           hists: list[np.ndarray], uids: np.ndarray, qidx: np.ndarray,
           n_shards: int, n_req: int, leaf_threads: int, budgets_ms: list[float],
           cold_thr: int, bg_threads: int, straggler_p: float = 0.0,
           straggler_ms: float = 0.0) -> dict:
    """The two fixes L2 names, each priced in the currency we are graded in.

    Hedging buys p99 with extra load and costs no recall (the answer is the same,
    it just arrives sooner). Budgets buy p99 with *dropped shards*, so they cost
    recall -- and the whole point is to state how much.
    """
    print(f"\n== D. hedging and budgets | n={n_shards} shards | {n_req:,} requests "
          f"| background load {bg_threads} thread(s)"
          + (f" | injected straggler p={straggler_p} +{straggler_ms:.0f}ms"
             if straggler_p else ""))
    rng = np.random.default_rng(2)
    sample = rng.choice(qidx, size=min(n_req, qidx.size), replace=False)
    out = {"n_shards": n_shards, "n_requests": int(sample.size),
           "background_threads": bg_threads, "straggler_p": straggler_p,
           "straggler_ms": straggler_ms, "rows": []}

    sub = np.ascontiguousarray(l2_normalise(emb[universe].astype(np.float32)))

    def run(deadline_s=None, hedge_after=None, label=""):
        with BackgroundLoad(sub[: min(len(sub), 20000)], bg_threads, sub.shape[1]), \
             ThreadPoolExecutor(leaf_threads) as pool:
            svc = ShardedService(emb, universe, hists, n_shards, pool=pool, sub=sub,
                                 straggler_p=straggler_p, straggler_ms=straggler_ms)
            for i in sample[:32]:
                svc.handle_sharded(int(i))
            svc.leaf_times.clear()
            root, miss, hed = [], 0, 0
            topk = np.full((sample.size, TOPK), NO_DOC, dtype=np.uint32)
            for r, i in enumerate(sample):
                t0 = time.perf_counter()
                ids, m, h = svc.handle_sharded(int(i), deadline_s, hedge_after)
                root.append(time.perf_counter() - t0)
                topk[r, :ids.size] = ids
                miss += m
                hed += h
            leaf = np.asarray(svc.leaf_times)
        rr = np.asarray(root)
        df = evaluate_recall(fs, split, uids[sample], topk, KS,
                             cold_threshold=cold_thr, only_users=True)
        s = summarise(df, KS)
        row = {"mode": label, "root": stats_ms(rr), "leaf": stats_ms(leaf),
               "shards_missed_per_req": round(miss / sample.size, 4),
               "hedges_per_req": round(hed / sample.size, 4),
               "extra_leaf_load_pct": round(100 * hed / max(sample.size * n_shards, 1), 2),
               **{f"recall@{k}": round(s[f"recall@{k}"], 5) for k in KS}}
        print(f"  {label:26s} p50={row['root']['p50_ms']:8.3f} p99={row['root']['p99_ms']:9.3f} ms "
              f"| miss/req {row['shards_missed_per_req']:5.2f} hedge/req {row['hedges_per_req']:5.2f} "
              f"| r@100 {row['recall@100']:.5f}")
        return row

    base = run(label="baseline (no fix)")
    out["rows"].append(base)
    # Hedge on the *leaf's* p95, not the root's. The root is a max over n leaves,
    # so its p95 already lies inside the straggler; a backup issued then arrives
    # after the original and buys nothing. Dean & Barroso hedge the operation, and
    # the operation here is one leaf probe.
    for q in (95, 99):
        d = base["leaf"][f"p{q}_ms"] / 1e3
        out["rows"].append(run(hedge_after=d,
                               label=f"hedge after leaf p{q}={d*1e3:.2f}ms"))
    for b in budgets_ms:
        out["rows"].append(run(deadline_s=b / 1e3, label=f"budget {b:.1f} ms"))
    r0 = base["recall@100"]
    for r in out["rows"]:
        r["recall@100_vs_baseline_pct"] = round(100 * (r["recall@100"] - r0) / max(r0, 1e-12), 3)
        r["p99_vs_baseline_pct"] = round(
            100 * (r["root"]["p99_ms"] - base["root"]["p99_ms"]) / max(base["root"]["p99_ms"], 1e-12), 2)
    return out


# ---------------------------------------------------------------- part F

def skewed_bounds(n_docs: int, n_shards: int, skew: float) -> np.ndarray:
    """Shard boundaries with sizes falling off as 1/rank^skew.

    `skew = 0` is the equal split every other part of this script uses. Anything
    above it is the case L2 names and nothing here has tested: "the challenge is
    skewed data or traffic -- hotspots, no clean auto-(re)balancing". A key-ranged
    shard map over a real corpus skews for free, because the keys are Zipf.
    """
    if skew <= 0:
        return np.linspace(0, n_docs, n_shards + 1).astype(int)
    w = 1.0 / np.arange(1, n_shards + 1, dtype=np.float64) ** skew
    sizes = np.maximum(1, np.round(w / w.sum() * n_docs).astype(int))
    sizes[0] += n_docs - int(sizes.sum())                # absorb rounding in the big one
    return np.concatenate([[0], np.cumsum(sizes)]).astype(int)


def part_f(emb: np.ndarray, universe: np.ndarray, hists: list[np.ndarray],
           qidx: np.ndarray, configs: list[tuple[int, float]], n_req: int,
           leaf_threads: int, bg_threads: int) -> dict:
    """Data skew, and whether micro-sharding is the fix L2 says it is.

    Scatter/gather makes traffic skew impossible -- every query touches every
    shard -- so the skew that matters here is *data* skew, and it bites through
    the same mechanism as the tail: the root waits for the largest leaf, so an
    unbalanced map throws away exactly the parallelism sharding was bought for.
    L2's prescription is to micro-shard, i.e. cut into many more, smaller pieces
    and let the pool balance them. That is a prediction, so it is measured here.
    """
    print(f"\n== F. shard skew | corpus {universe.size:,} | {n_req:,} requests per config "
          f"| leaf pool {leaf_threads} | background {bg_threads}")
    rng = np.random.default_rng(3)
    sample = rng.choice(qidx, size=min(n_req, qidx.size), replace=False)
    sub = np.ascontiguousarray(l2_normalise(emb[universe].astype(np.float32)))
    rows = []
    with BackgroundLoad(sub[: min(len(sub), 20000)], bg_threads, sub.shape[1]), \
         ThreadPoolExecutor(leaf_threads) as pool:
        for n, skew in configs:
            b = skewed_bounds(sub.shape[0], n, skew)
            sizes = np.diff(b)
            svc = ShardedService(emb, universe, hists, n, pool=pool, sub=sub, bounds=b)
            for i in sample[:32]:
                svc.handle_sharded(int(i))
            svc.leaf_times.clear()
            root = []
            for i in sample:
                t0 = time.perf_counter()
                svc.handle_sharded(int(i))
                root.append(time.perf_counter() - t0)
            r = np.asarray(root)
            leaf = np.asarray(svc.leaf_times)
            rows.append({"n_shards": n, "skew": skew,
                         "largest_shard": int(sizes.max()), "mean_shard": int(sizes.mean()),
                         "imbalance": round(float(sizes.max() / sizes.mean()), 2),
                         "largest_shard_share": round(float(sizes.max() / sizes.sum()), 4),
                         "root": stats_ms(r), "leaf": stats_ms(leaf)})
            print(f"  n={n:4d} skew={skew:.1f}  largest {sizes.max():>7,} "
                  f"({100*sizes.max()/sizes.sum():5.1f}% of corpus, {sizes.max()/sizes.mean():5.2f}x mean)"
                  f"  root p50={pct(r,50):8.3f} p99={pct(r,99):8.3f} ms")
    base = {(r["n_shards"], r["skew"]): r for r in rows}
    for r in rows:
        ref = base.get((r["n_shards"], 0.0))
        r["root_p50_vs_balanced_pct"] = (
            round(100 * (r["root"]["p50_ms"] - ref["root"]["p50_ms"]) / ref["root"]["p50_ms"], 1)
            if ref else None)
    return {"corpus": int(universe.size), "n_requests": int(sample.size),
            "leaf_threads": leaf_threads, "background_threads": bg_threads, "rows": rows}


# ---------------------------------------------------------------- part E

def encode_cost(fs: FeatureStore, n_sample: int, model: str, device: str) -> dict | None:
    """Time the embedding pass itself.

    L1 lists "embedding generation is a big cost" beside "what can be done
    offline?", and the repo has never priced it: EB-NeRD ships four sets of
    vectors and MIND's are encoded once at build time and cached, so the number
    exists nowhere. Encoding a sample and extrapolating is enough to put it on the
    offline side of the ledger next to the index build.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        return None
    texts = fs.texts()
    if not texts:
        return None
    n = min(n_sample, len(texts))
    m = SentenceTransformer(model, device=device)
    m.encode(texts[:32], batch_size=32, convert_to_numpy=True)   # warm the graph
    t0 = time.perf_counter()
    m.encode(texts[:n], batch_size=256, convert_to_numpy=True,
             normalize_embeddings=True, show_progress_bar=False)
    dt = time.perf_counter() - t0
    rate = n / dt
    return {"model": model, "device": device, "n_sampled": int(n),
            "seconds": round(dt, 3), "articles_per_s": round(rate, 1),
            "corpus_articles": int(len(texts)),
            "full_corpus_seconds": round(len(texts) / rate, 1)}


def part_e(emb: np.ndarray, universe: np.ndarray, part_a_res: dict,
           budget_ms: float, fs: FeatureStore | None = None,
           encode_sample: int = 0, encode_model: str = "sentence-transformers/all-MiniLM-L6-v2",
           encode_device: str = "cuda") -> dict:
    """The offline/online split, and what fits in a latency budget.

    L1's principle is "push heavy work offline -- latency there is hidden", and
    L2's budget slide then spends the online half a millisecond at a time. Both
    are arithmetic once the two sides are measured, and neither has been stated
    anywhere in this repo. The question a design note has to answer is not "is the
    index fast" but "how much of the budget has the retriever already spent, and
    what is left for the ranker" -- because M3's re-ranker has to fit in what
    remains.
    """
    print(f"\n== E. offline vs online | budget {budget_ms:.0f} ms")
    sub = np.ascontiguousarray(l2_normalise(emb[universe].astype(np.float32)))
    import faiss
    t0 = time.perf_counter()
    flat_index(sub)
    t_flat = time.perf_counter() - t0
    t0 = time.perf_counter()
    hn = faiss.IndexHNSWFlat(sub.shape[1], 16, faiss.METRIC_INNER_PRODUCT)
    hn.hnsw.efConstruction = 200
    hn.add(sub)
    t_hnsw = time.perf_counter() - t0

    online_ms = part_a_res["D_total_ms"]
    stages = part_a_res["stage_demand"]
    x_obs = part_a_res["X_max_observed_qps"]
    # how long the one-off build takes to amortise: at the observed serving rate,
    # the number of queries served in the time the index took to build
    amortise_flat = t_flat * x_obs
    amortise_hnsw = t_hnsw * x_obs
    res = {"budget_ms": budget_ms,
           "offline": {"index_build_flat_s": round(t_flat, 3),
                       "index_build_hnsw_M16_efC200_s": round(t_hnsw, 3),
                       "vectors": int(sub.shape[0]), "dim": int(sub.shape[1]),
                       "embedding_matrix_mb": round(sub.nbytes / 2**20, 1)},
           "online": {"R_min_ms": online_ms,
                      "budget_used_pct": round(100 * online_ms / budget_ms, 2),
                      "budget_left_ms": round(budget_ms - online_ms, 3),
                      "stages": {k: v["S_ms"] for k, v in stages.items()}},
           "amortisation_queries": {"flat": round(amortise_flat, 0),
                                    "hnsw": round(amortise_hnsw, 0)}}
    if encode_sample and fs is not None:
        enc = encode_cost(fs, encode_sample, encode_model, encode_device)
        if enc:
            res["offline"]["encode"] = enc
            print(f"  encode : {enc['articles_per_s']:,.0f} articles/s on {enc['device']} "
                  f"-> {enc['full_corpus_seconds']:,.0f}s for all "
                  f"{enc['corpus_articles']:,} articles ({enc['model'].split('/')[-1]})")
    print(f"  offline: flat build {t_flat:.2f}s | HNSW(M=16,efC=200) build {t_hnsw:.2f}s | "
          f"vectors {sub.nbytes/2**20:,.0f} MB")
    print(f"  online : R_min {online_ms:.3f} ms = {100*online_ms/budget_ms:.1f}% of a "
          f"{budget_ms:.0f} ms budget -> {budget_ms - online_ms:.2f} ms left for ranking")
    print(f"  a {t_hnsw:.0f}s HNSW build costs the wall-clock of "
          f"{amortise_hnsw:,.0f} served queries at {x_obs:,.0f} q/s")
    return res


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="ebnerd")
    ap.add_argument("--variant", default="large")
    ap.add_argument("--split", default="test")
    ap.add_argument("--embedding", default="contrastive")
    ap.add_argument("--parts", default="abcd")
    ap.add_argument("--scope", default="catalogue", choices=["universe", "catalogue"],
                    help="'universe' = the 7-day live window the assignment retrieves over; "
                         "'catalogue' = every article, i.e. the 10x system")
    ap.add_argument("--concurrency", default="1,2,4,8,16,32,48,64")
    ap.add_argument("--rhos", default="0.3,0.5,0.6,0.7,0.8,0.9,0.95")
    ap.add_argument("--shards", default="1,2,4,8,16,32,64,100")
    ap.add_argument("--budgets-ms", default="")
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--stage-reps", type=int, default=2000)
    ap.add_argument("--requests", type=int, default=2000)
    ap.add_argument("--leaf-threads", type=int, default=48)
    ap.add_argument("--background-threads", type=int, default=0,
                    help="competing search threads, so the leaf tail is real queueing")
    ap.add_argument("--max-users", type=int, default=20000)
    ap.add_argument("--encode-sample", type=int, default=0,
                    help="part E: time encoding this many article texts (0 = skip)")
    ap.add_argument("--encode-model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--encode-device", default="cuda")
    ap.add_argument("--skews", default="0,0.5,1.0",
                    help="part F: shard-size skew exponents (0 = equal split)")
    ap.add_argument("--straggler-p", type=float, default=0.0,
                    help="inject an idiosyncratic per-leaf pause with this probability")
    ap.add_argument("--straggler-ms", type=float, default=50.0,
                    help="length of the injected pause")
    ap.add_argument("--budget-ms", type=float, default=200.0,
                    help="the online latency budget the retriever must fit inside")
    ap.add_argument("--tag", default="", help="suffix so one scope can be run at several load levels")
    ap.add_argument("--out", default="reports/l2")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    fs = FeatureStore(a.dataset, a.variant)
    emb = article_embeddings(fs, a.embedding)
    uids, hists = user_histories(fs, a.split)
    if a.max_users and uids.size > a.max_users:
        keep = np.sort(np.random.default_rng(0).choice(uids.size, a.max_users, replace=False))
        uids, hists = uids[keep], [hists[i] for i in keep]
    if a.scope == "universe":
        universe = candidate_universe_for_split(fs, a.split, 7)
    else:
        universe = np.arange(emb.shape[0], dtype=np.int64)
    qidx = np.arange(len(hists), dtype=np.int64)
    cold_thr = 5 if a.dataset == "mind" else 10
    tag = f"{a.dataset}_{a.variant}_{a.scope}" + (f"_{a.tag}" if a.tag else "")

    print(f"L2 serving harness | {a.dataset}/{a.variant} {a.split} | scope={a.scope} "
          f"({universe.size:,} candidates) | {len(hists):,} users | emb={a.embedding}")

    ctx = pin_threads(1)                                    # one thread per leaf, always
    try:
        res = {"dataset": a.dataset, "variant": a.variant, "split": a.split,
               "embedding": a.embedding, "scope": a.scope,
               "candidates": int(universe.size), "dim": int(emb.shape[1]),
               "n_users": int(len(hists)), "top_k": TOPK,
               "leaf_threads": a.leaf_threads,
               "background_threads": a.background_threads,
               "cores": __import__("os").cpu_count(), "tag": a.tag}
        svc = RetrievalService(emb, universe, hists)
        if "a" in a.parts:
            res["part_a_service_demand"] = part_a(
                svc, qidx, [int(x) for x in a.concurrency.split(",")],
                a.seconds, a.stage_reps)
        if "b" in a.parts:
            s_mean = (res.get("part_a_service_demand", {}).get("D_total_ms")
                      or None)
            if s_mean is None:
                t = [ ]
                for i in qidx[:500]:
                    t0 = time.perf_counter(); svc.handle(int(i)); t.append(time.perf_counter() - t0)
                s_mean = float(np.mean(t)) * 1e3
            res["part_b_utilization_cliff"] = part_b(
                svc, qidx, s_mean / 1e3, [float(x) for x in a.rhos.split(",")], a.seconds)
        if "c" in a.parts:
            res["part_c_fanout"] = part_c(
                emb, universe, hists, qidx, [int(x) for x in a.shards.split(",")],
                a.requests, a.leaf_threads, a.background_threads)
        if "f" in a.parts:
            cfgs = []
            for n in [int(x) for x in a.shards.split(",")]:
                for sk in [float(x) for x in a.skews.split(",")]:
                    cfgs.append((n, sk))
            res["part_f_skew"] = part_f(emb, universe, hists, qidx, cfgs, a.requests,
                                        a.leaf_threads, a.background_threads)
        if "e" in a.parts:
            pa = res.get("part_a_service_demand")
            if pa is None:
                print("part E needs part A's stage timings; add 'a' to --parts")
            else:
                res["part_e_offline_online"] = part_e(
                    emb, universe, pa, a.budget_ms, fs, a.encode_sample,
                    a.encode_model, a.encode_device)
        if "d" in a.parts:
            shards = [int(x) for x in a.shards.split(",")]
            n = max(shards)
            if a.budgets_ms:
                budgets = [float(x) for x in a.budgets_ms.split(",")]
            else:
                base_p99 = (res.get("part_c_fanout", {}).get("rows") or [{}])[-1]
                p99 = base_p99.get("root", {}).get("p99_ms", 10.0)
                p50 = base_p99.get("root", {}).get("p50_ms", 5.0)
                budgets = [round(x, 3) for x in (p99, (p99 + p50) / 2, p50, p50 * 0.75)]
            res["part_d_fixes"] = part_d(
                fs, a.split, emb, universe, hists, uids, qidx, n,
                min(a.requests, 1500), a.leaf_threads, budgets, cold_thr,
                a.background_threads, a.straggler_p, a.straggler_ms)
    finally:
        if ctx is not None:
            ctx.__exit__(None, None, None)

    p = out / f"l2_serving_{tag}.json"
    p.write_text(json.dumps(res, indent=2))
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
