#!/usr/bin/env bash
# Mirror the working tree to the box that holds the data and the GPU.
#
# The datasets are 92 GB and live at /home/resources/ire_a1_data on gvlab2; the
# only copy of the built feature stores is there too. So code is authored here,
# pushed, and run there -- `pull` brings back the small artefacts (JSON results,
# generated markdown, figures) that the design note reads.
#
#   scripts/sync.sh push          code -> gvlab2
#   scripts/sync.sh pull          reports/ -> here
#   scripts/sync.sh run "make q1" push, then run a command remotely
set -euo pipefail

HOST="${A2_HOST:-gvlab2}"
# The campus resolver intermittently fails on *.iiit.ac.in while the hosts stay
# reachable, so the address can be pinned without editing ~/.ssh/config.
SSH_OPTS=()
[ -n "${A2_IP:-}" ] && SSH_OPTS+=(-o "HostName=$A2_IP")
REMOTE="${A2_REMOTE:-ire_a2}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# .venv, data and external/ are provisioned on the far side and must never be
# overwritten by the local tree (external/ebnerd-benchmark is a checkout, and
# the venv holds Linux+CUDA wheels this laptop has no use for).
EXCLUDES=(--exclude .git --exclude .venv --exclude data --exclude artifacts
          --exclude external --exclude logs --exclude __pycache__
          --exclude '.pytest_cache' --exclude '*.parquet' --exclude '*.npy'
          --exclude '*.zip' --exclude '*.faiss')

case "${1:-push}" in
  push)
    rsync -az --delete -e "ssh ${SSH_OPTS[*]}" "${EXCLUDES[@]}" "$HERE/" "$HOST:$REMOTE/"
    echo "pushed -> $HOST:$REMOTE"
    ;;
  pull)
    rsync -az -e "ssh ${SSH_OPTS[*]}" --exclude '*.npy' --exclude '*.parquet' \
          "$HOST:$REMOTE/reports/" "$HERE/reports/"
    echo "pulled reports/ <- $HOST:$REMOTE"
    ;;
  run)
    shift
    rsync -az --delete -e "ssh ${SSH_OPTS[*]}" "${EXCLUDES[@]}" "$HERE/" "$HOST:$REMOTE/"
    ssh "${SSH_OPTS[@]}" "$HOST" "bash -lc 'cd $REMOTE && $*'"
    ;;
  *)
    echo "usage: $0 {push|pull|run <cmd>}" >&2; exit 2
    ;;
esac
