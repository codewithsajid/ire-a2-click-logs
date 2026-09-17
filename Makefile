# CS4.406 IRE Assignment 2 -- Learning from click-logs on EB-NeRD + MIND
# (A1's targets are kept below: A2 reads A1's tuned BM25 parameters and its
#  feature store, so `make data` and `make q2` remain the way to rebuild them.)
# Raw/derived data lives outside $HOME (home space is tight on gvlab2):
#   data/ -> /home/resources/ire_a1_data     (override with DATA_ROOT=...)
.PHONY: setup download ebnerd mind-small mind-large verify status kernel clean-zips \
        data data-large test bench bench-gpu baseline q2 q3 ann q4 q4-ablation figures \
        query index-ablation multi universe l2 l3 l4 l5 \
        reproduce all ai-log note threshold userrep features


# ============================================================ Assignment 2
.PHONY: a2 a2-features a2-q1 a2-q2 a2-q3 a2-q4 a2-q5 a2-note a2-log a2-submit a2-test
PY2 ?= .venv/bin/python
DS2 ?= ebnerd mind
V2  ?= small
# LightGBM and polars each default to every core; two concurrent steps then ask
# for twice the machine and spend the difference on context switches.
export OMP_NUM_THREADS ?= 12
export POLARS_MAX_THREADS ?= 12
export PYTHONUNBUFFERED = 1

a2:                          ## A2 one-command reproduce: Q1 -> Q9 at dev scale
	bash scripts/run_overnight.sh

a2-features:                 ## Q1/Q2 design matrix + first re-ranker (both datasets)
	@for d in $(DS2); do $(PY2) scripts/q2_rerank.py --dataset $$d --variant $(V2) --tag shipped; \
	  $(PY2) scripts/q2_rerank.py --dataset $$d --variant $(V2) \
	    --families user,article,match,context,rolling --tag production; done

a2-q1:                       ## Q1 position bias, coverage, report
	@for d in $(DS2); do $(PY2) scripts/q1_position.py --dataset $$d --variant $(V2); \
	  $(PY2) scripts/q1_coverage.py --dataset $$d --variant $(V2) --splits test; done
	$(PY2) scripts/verify_history.py
	$(PY2) scripts/q1_report.py

a2-q2:                       ## Q2 two-stage cascade + objective arc + report
	@for d in $(DS2); do $(PY2) scripts/q2_twostage.py --dataset $$d --variant $(V2); \
	  $(PY2) scripts/q2_objective.py --dataset $$d --variant $(V2) --truncation-sweep 10,30,100,300; done
	$(PY2) scripts/q2_report.py

a2-q3:                       ## Q3 NRMS baseline, improvement, ablation, paired CIs
	@for d in $(DS2); do $(PY2) scripts/q3_nrms.py --dataset $$d --variant $(V2) --epochs 12 --patience 3; \
	  $(PY2) scripts/q3_nrms.py --dataset $$d --variant $(V2) --epochs 12 --patience 3 --extra-mode add; \
	  $(PY2) scripts/q3_ablation.py --dataset $$d --variant $(V2); done
	$(PY2) scripts/q3_report.py

a2-q4:                       ## Q4 serving + scale. REFUSES on a loaded box, by design
	@for d in $(DS2); do $(PY2) scripts/q4_serving.py --dataset $$d --variant $(V2); done
	$(PY2) scripts/q4_report.py

a2-q5:                       ## Q5 all metrics, both slices, CIs
	@for d in $(DS2); do $(PY2) scripts/q5_eval.py --dataset $$d --variant $(V2); done
	$(PY2) scripts/q5_report.py

a2-submit:                   ## Q5/Q7.1 leaderboard files at large scale (hours)
	@for d in $(DS2); do $(PY2) scripts/q5_submit.py --dataset $$d --variant large; done

a2-test:                     ## Q1.4/Q9 behaviour-window + A1's leakage suite
	$(PY2) -m pytest tests/ -q

a2-note:                     ## Q6 design note (6 pages, page count asserted)
	$(PY2) scripts/a2_design_note.py

a2-log:                      ## Q7.4 AI usage log, from the session transcripts
	$(PY2) scripts/ai_usage_log.py \
	  $${TRANSCRIPTS:-$$HOME/.claude/projects/-home-gokboru-Documents-Course-Work-M26-IRE-Assignments} \
	  reports/ai_usage_log.md

setup:                       ## venv + deps (uv; torch cu128 for the RTX 5090)
	uv sync

download: ebnerd mind-small  ## everything reachable without gated access

ebnerd:                      ## all EB-NeRD bundles + embedding artifacts (resumable)
	bash scripts/fetch_ebnerd.sh 24
mind-small:                  ## MIND-small (official zips if authorised, else ungated mirrors)
	bash scripts/download_mind.sh small
mind-large:                  ## MINDlarge_{train,dev,test} -- needs accepted HF gate
	bash scripts/download_mind.sh large

# ---- Q1: raw -> feature store -------------------------------------------
data:                        ## build the dev-scale stores (demo + small)
	.venv/bin/python -u -m newsrec.build --config configs/ebnerd_demo.yaml
	.venv/bin/python -u -m newsrec.build --config configs/ebnerd_small.yaml
	.venv/bin/python -u -m newsrec.build --config configs/mind_small.yaml

data-large:                  ## build the Codabench-scale stores (hours)
	.venv/bin/python -u -m newsrec.build --config configs/ebnerd_large.yaml
	.venv/bin/python -u -m newsrec.build --config configs/mind_large.yaml

test:                        ## behaviour-window / leakage assertions (Q9)
	.venv/bin/python -m pytest tests/ -q

bench:                       ## CPU timings for the hot operators
	.venv/bin/python -m newsrec.bench --dataset mind --variant small --engines polars \
	  --out reports/bench/mind_small_cpu.json

bench-gpu:                   ## three-way: polars CPU vs cudf-polars vs native cuDF
	.venv/bin/python -m newsrec.bench --dataset mind --variant small \
	  --engines polars,gpu,cudf --out reports/bench/mind_small_all.json

# ---- Q2 / Q3: retrieval -------------------------------------------------
baseline:                    ## leaderboard submissions (built against the RAW files, in their row order)
	.venv/bin/python -u -m newsrec.submit --dataset mind --variant large --split submit \
	  --raw-root data/raw/mind   --out reports/sub/prediction.txt
	.venv/bin/python -u -m newsrec.submit --dataset ebnerd --variant large --split submit \
	  --raw-root data/raw/ebnerd --out reports/sub/predictions.txt
	.venv/bin/python -m pytest tests/test_submission_format.py -q

q2:                          ## BM25 + (k1,b) ablation
	.venv/bin/python -u scripts/q2_bm25.py --dataset ebnerd --variant small
	.venv/bin/python -u scripts/q2_bm25.py --dataset mind   --variant small

q3:                          ## semantic ANN + lexical-vs-semantic comparison
	.venv/bin/python -u scripts/q3_semantic.py --dataset ebnerd --variant small \
	  --embeddings contrastive,xlm_roberta,word2vec,bert_multilingual
	.venv/bin/python -u scripts/q3_semantic.py --dataset mind --variant small \
	  --embeddings sentence-transformers/all-MiniLM-L6-v2
	.venv/bin/python -u scripts/q3_compare.py

threshold:                   ## Q3 ablation: similarity cutoffs vs a matched fixed top-K budget
	.venv/bin/python -u scripts/q3_threshold.py --dataset ebnerd --variant demo  --embedding contrastive
	.venv/bin/python -u scripts/q3_threshold.py --dataset ebnerd --variant small --embedding contrastive
	.venv/bin/python -u scripts/q3_threshold.py --dataset mind   --variant small --embedding all-MiniLM-L6-v2
	.venv/bin/python -u scripts/plots_threshold.py

userrep:                     ## Q3 ablation: how much history to read, and whether to decay it
	.venv/bin/python -u scripts/q3_userrep.py --dataset ebnerd --variant small --embedding contrastive
	.venv/bin/python -u scripts/q3_userrep.py --dataset mind   --variant small --embedding all-MiniLM-L6-v2

features:                    ## Q4 ablation: which stored user/article features help the ranking
	.venv/bin/python -u scripts/q4_features.py --dataset ebnerd --variant small --embedding contrastive
	.venv/bin/python -u scripts/q4_features.py --dataset mind --variant small \
	  --embedding all-MiniLM-L6-v2
	.venv/bin/python -u scripts/plots_features.py

query:                       ## Q2 ablation: query-side term saturation (k3)
	.venv/bin/python -u scripts/q2_query.py --dataset ebnerd --variant small
	.venv/bin/python -u scripts/q2_query.py --dataset mind   --variant small

index-ablation:              ## Q2 ablation: indexed field, stopwords, stemming
	.venv/bin/python -u scripts/q2_index.py --dataset ebnerd --variant small
	.venv/bin/python -u scripts/q2_index.py --dataset mind   --variant small

multi:                       ## Q3 ablation: multi-interest user representation
	.venv/bin/python -u scripts/q3_multiinterest.py --dataset ebnerd --variant small --embedding contrastive
	.venv/bin/python -u scripts/q3_multiinterest.py --dataset mind   --variant small --embedding all-MiniLM-L6-v2

universe:                    ## Q1/Q3 ablation: how much of recall is the candidate window
	.venv/bin/python -u scripts/q1_universe.py --dataset ebnerd --variant small --embedding contrastive
	.venv/bin/python -u scripts/q1_universe.py --dataset mind   --variant small --embedding all-MiniLM-L6-v2

ann:                         ## Q3 ablation: ANN index family + parameters, and scale sweep
	.venv/bin/python -u scripts/q3_ann.py --dataset ebnerd --variant small \
	  --embedding contrastive --part a --threads 1
	.venv/bin/python -u scripts/q3_ann.py --dataset mind --variant small \
	  --embedding all-MiniLM-L6-v2 --part a --threads 1 --max-queries 10000
	.venv/bin/python -u scripts/q3_ann.py --dataset ebnerd --variant small \
	  --embedding contrastive --part b --scale-variant large --threads 1
	.venv/bin/python -u scripts/ann_report.py

# ---- L2: the serving side (service demand, the cliff, fan-out, the tail) --
# Defaults to the large bundle: fan-out over a 1,677-article window is not a
# measurement of anything. `make L2_VARIANT=small l2` runs it at dev scale.
L2_VARIANT ?= large
l2:                          ## L2 ablation: service demand, utilization cliff, fan-out, skew and the tail
	.venv/bin/python -u scripts/l2_serving.py --dataset ebnerd --variant $(L2_VARIANT) \
	  --scope universe  --parts abe --seconds 8 --stage-reps 3000
	.venv/bin/python -u scripts/l2_serving.py --dataset ebnerd --variant $(L2_VARIANT) \
	  --scope catalogue --parts abe --seconds 8 --stage-reps 2000
	.venv/bin/python -u scripts/l2_serving.py --dataset mind --variant small \
	  --embedding all-MiniLM-L6-v2 --scope catalogue --parts abe --seconds 8 \
	  --stage-reps 2000 --rhos 0.3,0.5,0.7,0.9,0.95 --encode-sample 8000
	.venv/bin/python -u scripts/l2_serving.py --dataset ebnerd --variant $(L2_VARIANT) \
	  --scope catalogue --parts cd --shards 1,2,4,8,16,32,64,128 --requests 400 --tag idle
	.venv/bin/python -u scripts/l2_serving.py --dataset ebnerd --variant $(L2_VARIANT) \
	  --scope catalogue --parts cd --shards 1,2,4,8,16,32,64,128 --requests 400 \
	  --background-threads 32 --tag loaded
	.venv/bin/python -u scripts/l2_serving.py --dataset ebnerd --variant $(L2_VARIANT) \
	  --scope catalogue --parts f --shards 16,64,128 --skews 0,0.5,1.0 --requests 400 --tag skew
	.venv/bin/python -u scripts/l2_serving.py --dataset ebnerd --variant $(L2_VARIANT) \
	  --scope catalogue --parts d --shards 16 --requests 400 --straggler-p 0.01 \
	  --straggler-ms 50 --budgets-ms 60,40,25 --tag straggler_n16
	.venv/bin/python -u scripts/l2_report.py

# ---- L3: storage, access patterns and the RUM triangle -------------------
L3_VARIANT ?= large
l3:                          ## L3 ablation: hierarchy, random access, bytes-vs-throughput, the update axis
	.venv/bin/python -u scripts/l3_storage.py --dataset ebnerd --variant $(L3_VARIANT) \
	  --scope catalogue --parts abcd --gathers 100,1000,10000,50000 \
	  --dims 768,384,192,96,48 --queries 2000 --repeats 3 --days 7 --merge-every 4
	.venv/bin/python -u scripts/l3_storage.py --dataset mind --variant small \
	  --embedding all-MiniLM-L6-v2 --scope catalogue --parts abcd \
	  --gathers 100,1000,10000 --dims 384,192,96,48 --queries 2000 --repeats 3 \
	  --days 7 --merge-every 4
	.venv/bin/python -u scripts/l3_report.py

# ---- L4: corpus laws, near-duplicate detection and the sketch family -----
# Wants the large bundle for two reasons: a held-out Heaps prediction needs two
# corpus sizes, and near-duplicate stories only exist in a catalogue big enough
# to have republished any.
L4_VARIANT ?= large
l4:                          ## L4 ablation: Zipf/Heaps, shingling, MinHash/LSH, Bloom/CM/HLL
	.venv/bin/python -u scripts/l4_dedup.py --dataset ebnerd --variant $(L4_VARIANT) \
	  --small-variant small --split test --parts abcdef --sample 5000 --users 3000
	.venv/bin/python -u scripts/l4_dedup.py --dataset mind --variant $(L4_VARIANT) \
	  --small-variant small --split test --parts abcdef --sample 5000 --users 3000
	.venv/bin/python -u scripts/l4_report.py

# ---- L5: posting anatomy, compression and top-k query processing ---------
L5_VARIANT ?= large
l5:                          ## L5 ablation: postings, d-gaps/v-byte, TAAT/DAAT/WAND, skips, tiering, SPIMI
	.venv/bin/python -u scripts/l5_postings.py --dataset ebnerd --variant $(L5_VARIANT) \
	  --split test --parts abcdefg --users 2000 --queries 60
	.venv/bin/python -u scripts/l5_postings.py --dataset mind --variant $(L5_VARIANT) \
	  --split test --parts abcdefg --users 2000 --queries 60
	.venv/bin/python -u scripts/l5_report.py

# ---- Q4: offline evaluation harness (and the Q9 leakage ablation) -------
q4:                          ## AUC / MRR / nDCG + beyond-accuracy + slices + CIs
	.venv/bin/python -u scripts/q4_eval.py --dataset ebnerd --variant small --split test \
	  --embedding contrastive
	.venv/bin/python -u scripts/q4_eval.py --dataset mind --variant small --split test \
	  --embedding all-MiniLM-L6-v2
	.venv/bin/python -u scripts/q4_report.py

q4-ablation:                 ## same harness on augmented history (Q9 with/without)
	.venv/bin/python -u scripts/q4_eval.py --dataset ebnerd --variant small --split test \
	  --embedding contrastive --history-mode augmented
	.venv/bin/python -u scripts/q4_eval.py --dataset mind --variant small --split test \
	  --embedding all-MiniLM-L6-v2 --history-mode augmented

figures:                     ## report figures
	.venv/bin/python -u scripts/plots.py
	.venv/bin/python -u scripts/plots_ann.py
	.venv/bin/python -u scripts/plots_q4.py
	.venv/bin/python -u scripts/plots_threshold.py
	.venv/bin/python -u scripts/plots_features.py
	.venv/bin/python -u scripts/plots_ablations.py
	.venv/bin/python -u scripts/examples.py --dataset ebnerd --variant small --embedding contrastive

# ---- one command, raw files -> every number in the design note ----------
reproduce: data test q2 q3 ann threshold userrep query index-ablation multi universe q4 q4-ablation features baseline figures note results-page ai-log  ## everything at dev scale
	@echo "reports/{q2,q3,q4,figures,sub} are rebuilt."

all: download reproduce      ## download first, then the above (hours)

note:                        ## Q6 -- design note HTML + PDF, checks the 4-page limit
	.venv/bin/python -u scripts/design_note.py

results-page:                ## browsable results page (self-contained HTML)
	.venv/bin/python -u scripts/results_page.py

ai-log:                      ## Q7.4 -- regenerate the AI usage log from session transcripts
	.venv/bin/python -u scripts/ai_usage_log.py

verify:                      ## open every raw bundle and print headline counts
	.venv/bin/python scripts/verify_raw.py

normalize:                   ## strip __MACOSX/.DS_Store, flatten nested bundle dirs
	bash scripts/normalize_raw.sh

kernel:                      ## register the venv as a Jupyter kernel
	.venv/bin/python -m ipykernel install --user --name newsrec-a1 --display-name "IRE A1 (newsrec)"

status:                      ## what is on disk / still downloading
	@du -sh data/raw/* 2>/dev/null || true
	@echo "--- in flight ---"
	@ls data/raw/ebnerd/*.chunks.json 2>/dev/null | sed 's|.*/||;s|.chunks.json||' || echo "  none"
	@df -h /home | tail -1

clean-zips:                  ## drop archives once extracted (frees ~6 GB)
	rm -f data/raw/ebnerd/*.zip data/raw/mind/*.zip
