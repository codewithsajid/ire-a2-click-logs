# Learning from Click-Logs on EB-NeRD and MIND

CS4.406 Information Retrieval & Extraction — Assignment 2. Continues
[Assignment 1](https://github.com/codewithsajid/ire-a1-news-retrieval) (seeded from
commit `8f7b52e`), which built the lexical and semantic retrieval this one re-ranks.

Behavioural features from the click logs → a LambdaMART re-ranker over A1's candidate
generator → the NRMS starter baseline reproduced and beaten → serving and scale →
submissions to both Codabench leaderboards. Every number in this README and in the
design note is read out of a JSON written by the code that measured it, so the prose
cannot drift from the run.

Design note: [`reports/a2_design_note.pdf`](reports/a2_design_note.pdf) (6 pages).
Per-question reports: [`reports/q1/q1_features.md`](reports/q1/q1_features.md),
[`q2/q2_reranker.md`](reports/q2/q2_reranker.md),
[`q3/q3_baseline.md`](reports/q3/q3_baseline.md),
[`q4/q4_serving.md`](reports/q4/q4_serving.md),
[`q5/q5_evaluation.md`](reports/q5/q5_evaluation.md).

---

## 1. Results

### Re-ranking, in-impression — the question both leaderboards score

Test split, dev scale. *Before* is A1's best single signal on identical rows, which is
the honest baseline for what **learning a combination** buys rather than what it buys
over nothing. 95% bootstrap CIs in the generated reports.

| | A1 emb (before) | re-ranked (shipped) | | re-ranked (production) |
|---|---|---|---|---|
| EB-NeRD AUC | 0.5453 | **0.7302** | +33.9% | *0.7812* |
| EB-NeRD nDCG@10 | 0.4660 | **0.6041** | +29.6% | *0.6528* |
| MIND AUC | 0.6368 | **0.6452** | +1.3% | *0.6990* |
| MIND nDCG@10 | 0.3938 | **0.4021** | +2.1% | *0.4495* |

`shipped` uses only features computable on a split with **no labels in it**, so it is
what can go to Codabench. `production` adds the rolling click counters a live feature
store would hold and the withheld test labels deny us. The gap is what the evaluation
setup costs, not a leaderboard result.

The 20× difference between the two datasets is coverage, not tuning: 87.7% of MIND's
test users have no dated prior click and MIND ships no sessions at all, so its
behavioural columns are structurally empty and the model falls back on content.

### Stage one — the ceiling on the cascade

| | recall@100 | recall@200 | impressions retrieving no click |
|---|---|---|---|
| EB-NeRD | 0.1189 | 0.2074 | 88.0% → 79.2% |
| MIND | 0.0441 | 0.0676 | 93.6% → 90.3% |

Stage two cannot rank what stage one never retrieved. This is why the in-impression
numbers above are the ones that carry, and the corpus-wide cascade is reported
separately rather than blended into them.

### Baseline reproduced, then beaten

NRMS-docvec from `ebnerd-benchmark`, ported to PyTorch layer for layer.

| | NRMS | + freshness (bounded) | Δ AUC (paired, 95% CI) |
|---|---|---|---|
| EB-NeRD | 0.6041 | 0.6097 | +0.0056 [+0.0046, +0.0068] |
| MIND | 0.6467 | 0.6513 | +0.0045 [+0.0029, +0.0063] |

The same feature change on the shipped GBDT is worth far more — **+0.0509**
[+0.0501, +0.0517] on EB-NeRD and **+0.0538** [+0.0522, +0.0557] on MIND — which is
why the headline claim rests there. Architecture and feature axes are reported
separately; conflating them is what Q3.3 forbids.

---

## 2. Design decisions, and the alternative each one beat

| decision | alternative | why, measured |
|---|---|---|
| GBDT over hand-crafted features | neural ranker on the same matrix | the matrix is full of *structural* nulls — MIND has no sessions, cold users have no click times — and every imputation choice would be a confound in the ablations |
| families split by **availability** (`shipped` / `production`) | split by usefulness | the Codabench split has no labels, so rolling *click* counters are uncomputable there while rolling *exposure* counters are not; the split means the submitted model needs no surgery |
| train stage two on the **shown** lists | on stage-one's retrieved sets | the ordering inverts between framings; retrieved-set training needs clicks unioned in, and the union *is* the label — 0.99 AUC on its own retrieved sets, **below random** in-impression |
| `position` quarantined | used as a de-biasing feature | measured: neither dataset stores candidates in rendered order, so it carries nothing |
| bounded freshness head (α·tanh) | unbounded additive head | unbounded wins big where freshness dominates and loses big where content does; bounded is positive on both |
| early stopping on nDCG for **every** objective | metric follows the objective | otherwise the objective comparison changes two things at once — on MIND that showed as 148 trees against 13 |
| rolling counters read strictly before the request | a fixed per-split window | every fixed window is either stale by the end of the split or leaky at its start; an as-of join with `allow_exact_matches=False` is neither |
| retrieval per **user**, not per impression | per impression | the query is the click history, and EB-NeRD's test split has 244,647 impressions over 15,342 users |

---

## 3. Ablations

| ablation | swept over | headline |
|---|---|---|
| feature family, leave-one-out | user / article / match / context | `−article` **−0.1453** AUC; `−match` −0.0279; `−context` −0.0145; `−user` −0.0015 — all CIs exclude zero |
| the improvement | frozen prior vs rolling | **+0.0509** (EB-NeRD) / **+0.0538** (MIND); dead slots 84.2% → 3.7% and 54.9% → 3.7% |
| serving-unavailable features (Q9) | dwell, scroll, rendered position | **+0.0002, CI spans zero** — worth nothing measurable |
| objective arc | `binary` / `rank_xendcg` / `lambdarank` | predicted ordering **holds on EB-NeRD, inverts on MIND** |
| ↳ LambdaRank truncation | 10 / 30 / 100 / 300 | optimum tracks list length; explains ⅔ of MIND's inversion |
| training candidate set | shown / retrieved / retrieved+union | each wins only on the framing it matches; the ordering inverts |
| neural head form | unbounded / bounded | +0.0874 vs −0.0053 (EB-NeRD), −0.0971 vs +0.0045 (MIND) |
| position bias | raw vs length-stratified vs article-fixed | **absent**: raw 6.4× decay is a 1/L artifact |
| head vs tail | content vs article-log arms | content wins **both** slices — the lecture's prediction falsified, with the conditioning confound stated |
| MIND history direction | newest-last / newest-first | unknowable from the data; worth 0.5% AUC, so priced rather than guessed |

Bugs the harness caught before they became results: an expanding mean that used row
order as a tiebreak on second-resolution timestamps; a user lookup that clipped an
out-of-range id onto a real, different user; a stage-1 recall that computed to exactly
1.0 once the union was off; a cascade scored on the contaminated set it was meant to
expose; an out-of-bounds gather that surfaced as an unrelated cuBLAS error; and a
`push --delete` that overwrote fresh remote results with stale local ones.

---

## 4. Where this breaks at 10×

| what breaks | why | the fix |
|---|---|---|
| the frozen feature store | 84.2% of EB-NeRD candidate slots already carry a zero prior click count, and that grows with churn | the rolling counters, served from a streaming aggregate rather than a batch job |
| exact ANN search | A1 measured the flat/HNSW crossover at N≈8,000 | HNSW — 9.4× at 125K articles for 96.5% fidelity; its 42 s build becomes the bottleneck |
| the BM25 matmul | `H·TF` stops fitting at 10× users × 10× vocabulary | blocked product, or WAND once the *universe* grows ~5× |
| the re-rank feature join | peak RSS is linear in rows; the cascade build OOMed at K=200 over 200k impressions | chunked scoring — what makes the 205,925,868-slot EB-NeRD submission possible |
| thread oversubscription | LightGBM and polars each take every core; measured at load 77 on 48 cores, an ablation that takes 139 s had not finished in 59 minutes | explicit caps — 25× faster at 12 threads |
| single node | one box, no replication | the one item a single machine cannot measure, stated as a gap |

---

## 5. Reproducing

```bash
uv sync        # environment
make a2        # Q1 -> Q9 at dev scale, every report
```

Per-question targets if you want one piece: `make a2-q1 a2-q2 a2-q3 a2-q4 a2-q5`,
plus `a2-test`, `a2-note`, `a2-log`, and `a2-submit` for the leaderboard files at
large scale (hours). `make a2` is `scripts/run_overnight.sh`.

`run_overnight.sh` is the one-command path: one step per line, each logged and
isolated so a single failure does not abandon the rest, with thread caps and
unbuffered output. The latency benchmark is serialised by construction and
**refuses to run on a loaded box** — a p99 taken under contention measures the
contention, and nothing in the output would say so.

| target | question | writes |
|---|---|---|
| `scripts/q1_position.py` / `q1_coverage.py` / `q1_report.py` | Q1 | `reports/q1/` |
| `scripts/q2_rerank.py` | Q2 | `reports/q2/rerank_*.json` |
| `scripts/q2_twostage.py` | Q2 | cascade + training-set comparison |
| `scripts/q2_objective.py` | Q2 | the objective arc, paired CIs |
| `scripts/q3_nrms.py` | Q3.1–3.4 | NRMS, the improvement, paired CI |
| `scripts/q3_ablation.py` | Q3.3, Q9 | family LOO, rolling, grey |
| `scripts/q4_serving.py` | Q4 | memory, p99, cost/QPS |
| `scripts/q5_eval.py` | Q5 | all metrics, both slices, CIs |
| `scripts/q5_submit.py` | Q5, Q7.1 | leaderboard files, in raw row order |
| `scripts/a2_design_note.py` | Q6 | 6-page PDF, page count asserted |
| `pytest tests/` | Q1.4, Q9 | 150 passed, 10 skipped |

Data, the virtualenv and both caches are reached through symlinks and never live in
the repo. `scripts/sync.sh` mirrors the tree to the box that holds the 92 GB of data
and the GPU; results are produced there and pulled back.

---

## 6. Layout

```
src/newsrec/
  behaviour.py    Q1 design matrix: session position, dwell, freshness,
                  rolling counters, category match — all strictly prior
  bias.py         position-bias estimation, with the list-length confound removed
  rerank.py       LightGBM/LambdaMART over the five feature families
  twostage.py     stage-one retrieval and the cascade's candidate sets
  nrms.py         NRMS-docvec in PyTorch, plus the bounded freshness head
  evaluate.py     A1's metric harness + paired bootstrap (Q3.4)
  ...             config, schema, ingest, lexical, semantic, store, submit (A1)
scripts/          one entry point per question, plus run_overnight.sh
tests/            leakage, behaviour window, row order, official metrics, format
reports/          result JSONs, generated markdown, design note
```

### Deliverables map

| assignment item | where |
|---|---|
| Q1 features | `src/newsrec/behaviour.py`, `bias.py` → `reports/q1/q1_features.md` |
| Q2 re-ranker | `src/newsrec/rerank.py`, `twostage.py` → `reports/q2/q2_reranker.md` |
| Q3 baseline + improvement | `src/newsrec/nrms.py`, `scripts/q3_*.py` → `reports/q3/q3_baseline.md` |
| Q4 serving & scale | `scripts/q4_serving.py` → `reports/q4/q4_serving.md` |
| Q5 evaluation | `scripts/q5_eval.py` → `reports/q5/q5_evaluation.md` |
| Q5 submissions | `scripts/q5_submit.py` → `reports/sub/*.zip` |
| Q6 design note | `reports/a2_design_note.pdf` |
| Q7.3 leaderboard screenshots | `reports/leaderboard/` |
| Q7.4 AI usage log | `reports/ai_usage_log.md` |
| Q9 leakage test | `tests/test_behaviour_window.py` |

> `reports/q3/` and `reports/q4/` also hold **A1** artefacts (`q3_*.json`,
> `ann_ablation.md`, `q4_results.md`), kept because A2 reads A1's tuned BM25
> parameters from them. A2's own outputs are `nrms_*`, `ablation_*`, `serving_*`,
> `q3_baseline.md` and `q4_serving.md`.

---

## 7. Conventions worth knowing

**Availability decides what ships.** Not usefulness. The submit split has no labels,
so any feature derived from in-split clicks is unavailable there however much it helps
offline.

**A claim needs a control.** The probe that dated history items by article first-seen
time read 0.16 on EB-NeRD, where the answer was independently known to be ascending.
Running the known-answer control is the only reason that probe was not published.

**Paired, not independent, CIs.** Two separate intervals over the same impressions
overstate the uncertainty of their difference; the impression-to-impression variance is
common to both systems and cancels only when the resample is shared.

**Effect size next to significance.** With 244,647 and 73,152 test impressions,
thousandths of a point clear significance. That is a statement about sample size.
