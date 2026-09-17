#!/usr/bin/env bash
# Keep scripts/fetch_ebnerd.sh alive until every EB-NeRD bundle is complete.
# The S3 link drops connections and systemd-resolved occasionally fails to
# resolve the bucket, so a run can die half-way; par_download.py resumes from
# its per-chunk checkpoints, and this loop just restarts it.
set -uo pipefail
PROJ="$HOME/ire_a1"
cd "$PROJ"
for _ in $(seq 1 200); do
  if ! pgrep -f 'scripts/fetch_ebnerd.sh' > /dev/null; then
    # any .chunks.json left means at least one file is still incomplete
    if ! ls data/raw/ebnerd/*.chunks.json > /dev/null 2>&1 \
       && [ -d data/raw/ebnerd/ebnerd_large ] && [ -d data/raw/ebnerd/ebnerd_testset ]; then
      echo "[watchdog] all complete $(date)"; exit 0
    fi
    echo "[watchdog] restarting fetch $(date)"
    bash scripts/fetch_ebnerd.sh 24 >> logs/fetch_ebnerd.log 2>&1
  fi
  sleep 60
done
echo "[watchdog] gave up after 200 checks $(date)"
