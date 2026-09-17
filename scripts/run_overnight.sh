#!/usr/bin/env bash
# Q3 -> Q9 unattended. One step per line, each logged and isolated, so a single
# failure does not abandon the rest -- the same discipline as A1's run_large.sh.
#
#   bash scripts/run_overnight.sh            everything
#   bash scripts/run_overnight.sh q4         one stage
#
# Steps are ordered by dependency. The latency benchmark (q4) runs ALONE and is
# never parallel with anything: a timing taken under contention measures the
# contention. Everything else may overlap freely.
set -uo pipefail

# Unbuffered: python buffers stdout when it is not a tty, so a long step goes
# dark for minutes and a status check cannot tell progress from a hang.
export PYTHONUNBUFFERED=1
PY="${PY:-.venv/bin/python}"
LOG="${LOG:-logs}"
mkdir -p "$LOG" reports/q3 reports/q4 reports/q5
DATASETS="${DATASETS:-ebnerd mind}"
VARIANT="${VARIANT:-small}"
ONLY="${1:-all}"

step() {
  local name="$1"; shift
  if [ "$ONLY" != "all" ] && [[ "$name" != $ONLY* ]]; then return 0; fi
  local t0=$SECONDS
  printf '\n=== %-28s %s\n' "$name" "$(date '+%H:%M:%S')"
  if "$@" > "$LOG/$name.log" 2>&1; then
    printf '    ok   %4ds   %s\n' "$((SECONDS - t0))" "$LOG/$name.log"
  else
    printf '    FAIL %4ds   %s  (continuing)\n' "$((SECONDS - t0))" "$LOG/$name.log"
    tail -5 "$LOG/$name.log" | sed 's/^/      /'
  fi
}

echo "overnight run: $(date)  datasets='$DATASETS' variant=$VARIANT filter=$ONLY"

# ---------------------------------------------------------------- Q3
for d in $DATASETS; do
  step "q3_nrms_$d"       $PY scripts/q3_nrms.py --dataset "$d" --variant "$VARIANT" \
                                                 --epochs 12 --patience 3
  # the unbounded head, kept as the ablation that shows why the shipped one is
  # bounded: on MIND it drove validation AUC from 0.62 to 0.52 in one epoch
  step "q3_nrms_add_$d"   $PY scripts/q3_nrms.py --dataset "$d" --variant "$VARIANT" \
                                                 --epochs 12 --patience 3 --extra-mode add
  step "q3_ablation_$d" $PY scripts/q3_ablation.py --dataset "$d" --variant "$VARIANT"
done

# ---------------------------------------------------------------- Q5
for d in $DATASETS; do
  step "q5_eval_$d"     $PY scripts/q5_eval.py     --dataset "$d" --variant "$VARIANT"
done

# ---------------------------------------------------------------- Q2 extras
for d in $DATASETS; do
  step "q2_objective_$d" $PY scripts/q2_objective.py --dataset "$d" --variant "$VARIANT" \
                                                     --truncation-sweep 10,30,100,300
done

# ---------------------------------------------------------------- Q4 (ALONE)
# Nothing above may still be running here. The loop is sequential and every step
# blocks, so this holds by construction -- stated because it is a property of the
# script that a future edit could silently break.
for d in $DATASETS; do
  step "q4_serving_$d"  $PY scripts/q4_serving.py  --dataset "$d" --variant "$VARIANT"
done

# ---------------------------------------------------------------- reports
step "report_q1" $PY scripts/q1_report.py
step "report_q2" $PY scripts/q2_report.py
step "report_q3" $PY scripts/q3_report.py
step "report_q4" $PY scripts/q4_report.py
step "report_q5" $PY scripts/q5_report.py

echo
echo "done: $(date)"
ls -la reports/q*/*.md 2>/dev/null | awk '{print "  " $9, $5" bytes"}'
