#!/usr/bin/env bash
# Re-run every headline number, dev scale first then Codabench scale.
#
# Use this whenever something upstream of the reports changes -- the feature
# store schema, a default, a retriever -- so the whole report set stays one
# vintage. Mixing vintages has cost real debugging time here: a Q2 report that
# predated a store rebuild and a Q3 report that predated a baseline rewrite both
# looked like regressions until the timestamps were checked.
#
# Dev scale runs first so a mistake surfaces in minutes rather than after the
# Codabench-scale steps have burned an hour. Each step is logged and isolated:
# one failure does not abandon the rest.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
mkdir -p logs
LOG=logs/pipeline_run.log

run() {
  local name=$1; shift
  echo "[$(date '+%F %T')] START  $name" | tee -a "$LOG"
  if /usr/bin/time -f "%e s  %M KB peak" -o /tmp/_t_$name "$@" >> "logs/step_$name.log" 2>&1; then
    echo "[$(date '+%F %T')] OK     $name  ($(cat /tmp/_t_$name))" | tee -a "$LOG"
  else
    echo "[$(date '+%F %T')] FAIL   $name  rc=$? -- see logs/step_$name.log" | tee -a "$LOG"
  fi
}

echo "=== pipeline run started $(date '+%F %T') ===" | tee -a "$LOG"

# ---- 1. dev scale ----------------------------------------------------------
run q2_demo_ebnerd   $PY -u scripts/q2_bm25.py --dataset ebnerd --variant demo
run q2_small_ebnerd  $PY -u scripts/q2_bm25.py --dataset ebnerd --variant small
run q2_small_mind    $PY -u scripts/q2_bm25.py --dataset mind   --variant small

run q3_small_ebnerd  $PY -u scripts/q3_semantic.py --dataset ebnerd --variant small \
    --embeddings contrastive,xlm_roberta,word2vec,bert_multilingual
run q3_small_mind    $PY -u scripts/q3_semantic.py --dataset mind --variant small \
    --embeddings sentence-transformers/all-MiniLM-L6-v2
run q3-compare             $PY -u scripts/q3_compare.py

run thr_demo_ebnerd  $PY -u scripts/q3_threshold.py --dataset ebnerd --variant demo  --embedding contrastive
run thr_small_ebnerd $PY -u scripts/q3_threshold.py --dataset ebnerd --variant small --embedding contrastive
run thr_small_mind   $PY -u scripts/q3_threshold.py --dataset mind   --variant small --embedding all-MiniLM-L6-v2

run feat_ebnerd      $PY -u scripts/q4_features.py --dataset ebnerd --variant small --embedding contrastive
run feat_mind        $PY -u scripts/q4_features.py --dataset mind --variant small \
    --embedding all-MiniLM-L6-v2

run q2q_ebnerd       $PY -u scripts/q2_query.py --dataset ebnerd --variant small
run q2q_mind         $PY -u scripts/q2_query.py --dataset mind   --variant small
run q2i_ebnerd       $PY -u scripts/q2_index.py --dataset ebnerd --variant small
run q2i_mind         $PY -u scripts/q2_index.py --dataset mind   --variant small
run multi_ebnerd     $PY -u scripts/q3_multiinterest.py --dataset ebnerd --variant small --embedding contrastive
run multi_mind       $PY -u scripts/q3_multiinterest.py --dataset mind   --variant small --embedding all-MiniLM-L6-v2

run uni_ebnerd       $PY -u scripts/q1_universe.py --dataset ebnerd --variant small --embedding contrastive
run uni_mind         $PY -u scripts/q1_universe.py --dataset mind   --variant small --embedding all-MiniLM-L6-v2

run ann_a_ebnerd     $PY -u scripts/q3_ann.py --dataset ebnerd --variant small \
    --embedding contrastive --part a --threads 1
run ann_a_mind       $PY -u scripts/q3_ann.py --dataset mind --variant small \
    --embedding all-MiniLM-L6-v2 --part a --threads 1 --max-queries 10000

for hm in shipped augmented; do
  run q4_small_ebnerd_$hm $PY -u scripts/q4_eval.py --dataset ebnerd --variant small \
      --split test --embedding contrastive --history-mode $hm
  run q4_small_mind_$hm   $PY -u scripts/q4_eval.py --dataset mind --variant small \
      --split test --embedding all-MiniLM-L6-v2 --history-mode $hm
done

echo "DEV_SCALE_DONE" | tee -a "$LOG"

# ---- 2. Codabench scale ----------------------------------------------------
run q2_large_ebnerd $PY -u scripts/q2_bm25.py --dataset ebnerd --variant large --grid-users 100000
run q2_large_mind   $PY -u scripts/q2_bm25.py --dataset mind   --variant large --grid-users 100000

run q3_large_ebnerd $PY -u scripts/q3_semantic.py --dataset ebnerd --variant large \
    --embeddings contrastive,xlm_roberta
run q3_large_mind   $PY -u scripts/q3_semantic.py --dataset mind --variant large \
    --embeddings sentence-transformers/all-MiniLM-L6-v2

run ann_b_ebnerd $PY -u scripts/q3_ann.py --dataset ebnerd --variant small \
    --embedding contrastive --part b --scale-variant large --threads 1

run q4_large_ebnerd $PY -u scripts/q4_eval.py --dataset ebnerd --variant large --split test \
    --embedding contrastive --max-impressions 600000 --n-boot 200
run q4_large_mind   $PY -u scripts/q4_eval.py --dataset mind --variant large --split test \
    --embedding all-MiniLM-L6-v2 --max-impressions 200000 --n-boot 200

echo "LARGE_SCALE_DONE" | tee -a "$LOG"

# ---- 3. reports ------------------------------------------------------------
run q4_report    $PY -u scripts/q4_report.py
run ann_report   $PY -u scripts/ann_report.py
run examples     $PY -u scripts/examples.py --dataset ebnerd --variant small --embedding contrastive
run plots        $PY -u scripts/plots.py
run plots_ann    $PY -u scripts/plots_ann.py
run plots_q4     $PY -u scripts/plots_q4.py
run plots_thresh $PY -u scripts/plots_threshold.py
run plots_feat   $PY -u scripts/plots_features.py
run plots_abl    $PY -u scripts/plots_ablations.py

echo "=== pipeline run finished $(date '+%F %T') ===" | tee -a "$LOG"
echo PIPELINE_RUN_DONE | tee -a "$LOG"
