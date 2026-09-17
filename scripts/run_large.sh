#!/usr/bin/env bash
# Overnight: refresh the dev-scale results with the new metrics, then run the
# identical pipeline at Codabench scale (EB-NeRD large: 125,541 articles /
# 12.5M labelled test impressions; MIND large: 130,379 / 376,471).
#
# Each step is logged and isolated -- one failure does not abandon the rest,
# because a nine-hour run that dies on step 3 and reports nothing is worse than
# one that finishes seven of nine steps and says which two are missing.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
mkdir -p logs reports/q4
LOG=logs/large_run.log

run() {
  local name=$1; shift
  echo "[$(date '+%F %T')] START  $name" | tee -a "$LOG"
  if /usr/bin/time -f "%e s  %M KB peak" -o /tmp/_t "$@" >> "logs/$name.log" 2>&1; then
    echo "[$(date '+%F %T')] OK     $name  ($(cat /tmp/_t))" | tee -a "$LOG"
  else
    echo "[$(date '+%F %T')] FAIL   $name  rc=$? -- see logs/$name.log" | tee -a "$LOG"
  fi
}

echo "=== run started $(date '+%F %T') ===" | tee -a "$LOG"

# ---- 1. dev scale, refreshed for the cutoff/rank-profile metrics -----------
# (skipped if already regenerated -- the shipped pair was done before the rerun)
for hm in shipped augmented; do
  run "q4_small_ebnerd_$hm" $PY -u scripts/q4_eval.py --dataset ebnerd --variant small \
      --split test --embedding contrastive --history-mode $hm
  run "q4_small_mind_$hm"   $PY -u scripts/q4_eval.py --dataset mind --variant small \
      --split test --embedding all-MiniLM-L6-v2 --history-mode $hm
done

# ---- 2. Codabench scale ----------------------------------------------------
run q2_large_ebnerd $PY -u scripts/q2_bm25.py --dataset ebnerd --variant large --grid-users 100000
run q2_large_mind   $PY -u scripts/q2_bm25.py --dataset mind   --variant large --grid-users 100000

run q3_large_ebnerd $PY -u scripts/q3_semantic.py --dataset ebnerd --variant large \
    --embeddings contrastive,xlm_roberta
run q3_large_mind   $PY -u scripts/q3_semantic.py --dataset mind --variant large \
    --embeddings sentence-transformers/all-MiniLM-L6-v2

# EB-NeRD large test is 12.5M impressions over 150M candidate rows. The harness
# peaks at ~74 KB per impression (measured at dev scale), so the cap is set by
# RAM, not by statistics: 600K impressions is ~45 GB peak and a 95% CI half-width
# of ~0.0006 -- an order of magnitude below the gaps between rankers.
run q4_large_ebnerd $PY -u scripts/q4_eval.py --dataset ebnerd --variant large --split test \
    --embedding contrastive --max-impressions 600000 --n-boot 200
run q4_large_mind   $PY -u scripts/q4_eval.py --dataset mind --variant large --split test \
    --embedding all-MiniLM-L6-v2 --max-impressions 200000 --n-boot 200

# ---- 2b. the serving ablation (L2) -----------------------------------------
# Needs the large bundle: fan-out over a 1,677-article window measures nothing.
# Each invocation is isolated so a bad measurement does not cost the rest.
run l2_universe  $PY -u scripts/l2_serving.py --dataset ebnerd --variant large \
    --scope universe  --parts abe --seconds 8 --stage-reps 3000
run l2_catalogue $PY -u scripts/l2_serving.py --dataset ebnerd --variant large \
    --scope catalogue --parts abe --seconds 8 --stage-reps 2000
run l2_fanout_idle   $PY -u scripts/l2_serving.py --dataset ebnerd --variant large \
    --scope catalogue --parts cd --shards 1,2,4,8,16,32,64,128 --requests 400 --tag idle
run l2_fanout_loaded $PY -u scripts/l2_serving.py --dataset ebnerd --variant large \
    --scope catalogue --parts cd --shards 1,2,4,8,16,32,64,128 --requests 400 \
    --background-threads 32 --tag loaded
run l2_skew      $PY -u scripts/l2_serving.py --dataset ebnerd --variant large \
    --scope catalogue --parts f --shards 16,64,128 --skews 0,0.5,1.0 --requests 400 --tag skew
run l2_straggler $PY -u scripts/l2_serving.py --dataset ebnerd --variant large \
    --scope catalogue --parts d --shards 16 --requests 400 --straggler-p 0.01 \
    --straggler-ms 50 --budgets-ms 60,40,25 --tag straggler_n16
run l2_mind      $PY -u scripts/l2_serving.py --dataset mind --variant small \
    --embedding all-MiniLM-L6-v2 --scope catalogue --parts abe --seconds 8 \
    --stage-reps 2000 --rhos 0.3,0.5,0.7,0.9,0.95 --encode-sample 8000

# ---- 2c. storage, access patterns and the RUM triangle (L3) ----------------
run l3_ebnerd $PY -u scripts/l3_storage.py --dataset ebnerd --variant large \
    --scope catalogue --parts abcd --gathers 100,1000,10000,50000 \
    --dims 768,384,192,96,48 --queries 2000 --repeats 3 --days 7 --merge-every 4
run l3_mind   $PY -u scripts/l3_storage.py --dataset mind --variant small \
    --embedding all-MiniLM-L6-v2 --scope catalogue --parts abcd \
    --gathers 100,1000,10000 --dims 384,192,96,48 --queries 2000 --repeats 3 \
    --days 7 --merge-every 4

# ---- 2d. corpus laws, near-duplicate detection and sketches (L4) ------------
# Zipf/Heaps need the large bundle for a held-out prediction; the dedup pass needs
# a catalogue big enough to contain genuinely republished stories.
run l4_ebnerd $PY -u scripts/l4_dedup.py --dataset ebnerd --variant large \
  --small-variant small --split test --parts abcdef --sample 5000 --users 3000
run l4_mind   $PY -u scripts/l4_dedup.py --dataset mind --variant large \
  --small-variant small --split test --parts abcdef --sample 5000 --users 3000

# ---- 2e. postings, compression and top-k processing (L5) -------------------
run l5_ebnerd $PY -u scripts/l5_postings.py --dataset ebnerd --variant large \
  --split test --parts abcdefg --users 2000 --queries 60
run l5_mind   $PY -u scripts/l5_postings.py --dataset mind --variant large \
  --split test --parts abcdefg --users 2000 --queries 60

# ---- 3. reports ------------------------------------------------------------
run q4_report   $PY -u scripts/q4_report.py
run ann_report  $PY -u scripts/ann_report.py
run l2_report   $PY -u scripts/l2_report.py
run l3_report   $PY -u scripts/l3_report.py
run l4_report   $PY -u scripts/l4_report.py
run l5_report   $PY -u scripts/l5_report.py
run plots_q4    $PY -u scripts/plots_q4.py
run plots_ann   $PY -u scripts/plots_ann.py
run design_note $PY -u scripts/design_note.py
run ai_log      $PY -u scripts/ai_usage_log.py

echo "=== run finished $(date '+%F %T') ===" | tee -a "$LOG"
echo LARGE_RUN_DONE | tee -a "$LOG"
