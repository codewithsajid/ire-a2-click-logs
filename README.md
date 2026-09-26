# Learning from Click-Logs on EB-NeRD and MIND

CS4.406 Information Retrieval & Extraction — Assignment 2. Continues
[Assignment 1](https://github.com/codewithsajid/ire-a1-news-retrieval), which built the
lexical and semantic retrieval this one re-ranks.

A two-stage retrieve-then-rank pipeline: behavioural features from the click logs, a
LambdaMART re-ranker over A1's candidate generator, the NRMS starter baseline
reproduced and beaten, serving and scale measurements, and submissions to both
Codabench leaderboards.

**Design note:** [`reports/a2_design_note.pdf`](reports/a2_design_note.pdf) (6 pages).
**Per-question reports:** [Q1](reports/q1/q1_features.md) ·
[Q2](reports/q2/q2_reranker.md) · [Q3](reports/q3/q3_baseline.md) ·
[Q4](reports/q4/q4_serving.md) · [Q5](reports/q5/q5_evaluation.md).

---

## Results

Test split, dev scale. *Before* is A1's best single signal on the same rows, so the
delta measures what learning a combination buys. 95% bootstrap CIs are in the reports.

| | before | re-ranked (shipped) | | re-ranked (production) |
|---|---|---|---|---|
| EB-NeRD AUC | 0.5453 | **0.7302** | +33.9% | *0.7812* |
| EB-NeRD nDCG@10 | 0.4660 | **0.6041** | +29.6% | *0.6528* |
| MIND AUC | 0.6368 | **0.6452** | +1.3% | *0.6990* |
| MIND nDCG@10 | 0.3938 | **0.4021** | +2.1% | *0.4495* |

`shipped` uses only features computable on a split with no labels, so it is what goes
to Codabench. `production` adds the rolling click counters a live feature store would
hold. The gap is the cost of the evaluation setup, not a leaderboard result.

The difference between the two datasets is coverage: 87.7% of MIND's test users have
no dated prior click and MIND ships no sessions, so its behavioural columns are
largely empty and the model falls back on content.

### Stage one is the ceiling

| | recall@100 | recall@200 | impressions retrieving no click |
|---|---|---|---|
| EB-NeRD | 0.1189 | 0.2074 | 88.0% → 79.2% |
| MIND | 0.0441 | 0.0676 | 93.6% → 90.3% |

Stage two cannot rank what stage one never retrieved, so the in-impression numbers
above and the corpus-wide cascade are reported separately rather than blended.

### Baseline reproduced, then beaten

NRMS-docvec from `ebnerd-benchmark`, ported to PyTorch.

| | NRMS | + freshness (bounded) | Δ AUC (paired, 95% CI) |
|---|---|---|---|
| EB-NeRD | 0.6041 | 0.6097 | +0.0056 [+0.0046, +0.0068] |
| MIND | 0.6467 | 0.6513 | +0.0045 [+0.0029, +0.0063] |

The same feature change on the GBDT is worth more — **+0.0509** [+0.0501, +0.0517] on
EB-NeRD, **+0.0538** [+0.0522, +0.0557] on MIND — which is where the headline claim
sits. The architecture and feature axes are reported separately.

---

## Design choices

| choice | why |
|---|---|
| GBDT over hand-crafted features | the matrix is full of structural nulls (MIND has no sessions, cold users have no click times); axis-aligned splits take them without imputation |
| families split by availability (`shipped` / `production`) | the submit split has no labels, so rolling *click* counters are uncomputable there while rolling *exposure* counters are not |
| train stage two on the shown lists | training on retrieved sets needs clicks unioned back in, and the union is the label — 0.99 AUC on its own retrieved sets, below random in-impression |
| `position` excluded | neither dataset stores candidates in rendered order, so it carries nothing |
| bounded freshness head (α·tanh) | the unbounded form wins where freshness dominates and loses where content does; bounded is positive on both |
| early stopping on nDCG for every objective | otherwise the objective comparison changes two things at once — on MIND, 148 trees against 13 |
| rolling counters read strictly before the request | a fixed window is either stale by the end of the split or leaky at its start |
| retrieval per user, not per impression | the query is the click history, and EB-NeRD's test split has 244,647 impressions over 15,342 users |

---

## Ablations

| ablation | swept over | result |
|---|---|---|
| feature family, leave-one-out | user / article / match / context | `−article` **−0.1453** AUC; `−match` −0.0279; `−context` −0.0145; `−user` −0.0015; all CIs exclude zero |
| the improvement | frozen vs rolling prior | **+0.0509** (EB-NeRD) / **+0.0538** (MIND); dead slots 84.2% → 3.7% and 54.9% → 3.7% |
| serving-unavailable features (Q9) | dwell, scroll, rendered position | +0.0002 AUC, CI spans zero |
| objective arc | `binary` / `rank_xendcg` / `lambdarank` | the predicted ordering holds on EB-NeRD, inverts on MIND |
| ↳ LambdaRank truncation | 10 / 30 / 100 / 300 | the optimum tracks list length; explains ⅔ of MIND's inversion |
| training candidate set | shown / retrieved / retrieved+union | each wins on the framing it matches |
| neural head form | unbounded / bounded | +0.0874 vs −0.0053 (EB-NeRD); −0.0971 vs +0.0045 (MIND) |
| position bias | raw / length-stratified / article-fixed | absent; the raw 6.4× decay is a 1/L artifact |
| head vs tail | content vs article-log arms | the verdict flips with the framing: content wins all four slices in-impression, but through the cascade MIND inverts and article-log wins both |
| MIND history direction | newest-last / newest-first | not determinable from the data; worth 0.5% AUC |

---

## Where this breaks at 10×

| what breaks | why | the fix |
|---|---|---|
| the frozen feature store | 84.2% of EB-NeRD candidate slots carry a zero prior click count, and that grows with churn | rolling counters from a streaming aggregate |
| exact ANN search | A1 measured the flat/HNSW crossover at N≈8,000 | HNSW — 9.4× at 125K articles for 96.5% fidelity; its 42 s build becomes the bottleneck |
| the BM25 matmul | `H·TF` stops fitting at 10× users × 10× vocabulary | blocked product, or WAND once the universe grows ~5× |
| the re-rank feature join | peak RSS is linear in rows; the cascade build OOMed at K=200 over 200k impressions | chunked scoring |
| thread oversubscription | LightGBM and polars each take every core; at load 77 on 48 cores an ablation that takes 139 s had not finished in 59 minutes | explicit caps — 25× faster at 12 threads |
| single node | one box, no replication | not measurable here; stated as a gap |

---

## Reproducing

```bash
uv sync        # environment
make a2        # Q1 -> Q9 at dev scale, every report
```

`make a2` runs in order: design matrix, Q1, Q2, Q3, Q5, Q4, tests, design note, AI
usage log. Per-question targets: `make a2-q1 a2-q2 a2-q3 a2-q4 a2-q5`, plus `a2-test`,
`a2-note`, `a2-log`, and `a2-submit` for the leaderboard files at large scale (hours).

Q4 runs last and refuses to measure on a loaded box — a p99 taken under contention
measures the contention. `make a2-overnight` is the unattended Q3 → Q5 subset and
assumes the design matrix and cascade already exist.

| script | question | writes |
|---|---|---|
| `q1_position.py` / `q1_coverage.py` / `q1_report.py` | Q1 | `reports/q1/` |
| `q2_rerank.py` | Q2 | `reports/q2/rerank_*.json` |
| `q2_twostage.py` | Q2 | cascade + training-set comparison |
| `q2_objective.py` | Q2 | objective arc, paired CIs |
| `q3_nrms.py` | Q3.1–3.4 | NRMS, the improvement, paired CI |
| `q3_ablation.py` | Q3.3, Q9 | family LOO, rolling, grey |
| `q4_serving.py` | Q4 | memory, p99, cost/QPS |
| `q5_eval.py` | Q5 | all metrics, both slices, CIs |
| `q5_submit.py` | Q5, Q7.1 | leaderboard files, in raw row order |
| `a2_design_note.py` | Q6 | the 6-page PDF |
| `pytest tests/` | Q1.4, Q9 | 177 passed, 10 skipped |

Data, the virtualenv and the caches are reached through symlinks and are not in the
repo. `scripts/sync.sh` mirrors the tree to the box holding the data and the GPU.

---

## Layout

```
src/newsrec/
  behaviour.py    Q1 design matrix: session position, dwell, freshness,
                  rolling counters, category match — all strictly prior
  bias.py         position-bias estimation with the list-length confound removed
  rerank.py       LightGBM/LambdaMART over the five feature families
  twostage.py     stage-one retrieval and the cascade's candidate sets
  nrms.py         NRMS-docvec in PyTorch, plus the bounded freshness head
  evaluate.py     A1's metric harness + paired bootstrap (Q3.4)
  ...             config, schema, ingest, lexical, semantic, store, submit (A1)
scripts/          one entry point per question
tests/            leakage, behaviour window, row order, official metrics, format
reports/          result JSONs, generated markdown, design note
```

| assignment item | where |
|---|---|
| Q1 features | `behaviour.py`, `bias.py` → `reports/q1/q1_features.md` |
| Q2 re-ranker | `rerank.py`, `twostage.py` → `reports/q2/q2_reranker.md` |
| Q3 baseline + improvement | `nrms.py`, `scripts/q3_*.py` → `reports/q3/q3_baseline.md` |
| Q4 serving & scale | `scripts/q4_serving.py` → `reports/q4/q4_serving.md` |
| Q5 evaluation | `scripts/q5_eval.py` → `reports/q5/q5_evaluation.md` |
| Q5 submissions | `scripts/q5_submit.py` → `reports/sub/*.zip` |
| Q6 design note | `reports/a2_design_note.pdf` |
| Q7.3 leaderboard screenshots | `reports/leaderboard/` |
| Q7.4 AI usage log | `reports/ai_usage_log.md` |
| Q9 leakage test | `tests/test_behaviour_window.py` |

`reports/q3/` and `reports/q4/` also hold A1 artefacts (`q3_*.json`,
`ann_ablation.md`, `q4_results.md`), kept because A2 reads A1's tuned BM25 parameters
from them. A2's own outputs are `nrms_*`, `ablation_*`, `serving_*`, `q3_baseline.md`
and `q4_serving.md`.
