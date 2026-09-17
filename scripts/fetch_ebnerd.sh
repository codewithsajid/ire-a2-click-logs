#!/usr/bin/env bash
# Download the EB-NeRD bundles (RecSys 2024 Challenge, Ekstra Bladet).
#
# The S3 bucket throttles each connection to ~30 KB/s from gvlab2, so files are
# pulled with scripts/par_download.py (many HTTP Range connections, per-chunk
# resume).  Re-running is safe: complete files are skipped, partial ones resume.
#
# Data lands in $DATA_ROOT (default: the data/ symlink -> /home/resources/ire_a1_data).
# Usage: scripts/fetch_ebnerd.sh [conns]
set -uo pipefail
PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-$PROJ/data}"
RAW="$DATA_ROOT/raw/ebnerd"
S3="https://ebnerd-dataset.s3.eu-west-1.amazonaws.com"
CONNS="${1:-24}"
mkdir -p "$RAW" "$PROJ/logs"

# priority order: dev sets, then the Codabench-required large bundles, then the
# optional pre-trained embedding artifacts.
# NB: articles_large_only.zip sits at the bucket root, not under artifacts/ as
# the assignment PDF states (that path 404s).
FILES=(
  ebnerd_demo.zip
  ebnerd_small.zip
  ebnerd_testset.zip
  ebnerd_large.zip
  articles_large_only.zip
  artifacts/Ekstra_Bladet_word2vec.zip
  artifacts/Ekstra_Bladet_contrastive_vector.zip
  artifacts/google_bert_base_multilingual_cased.zip
  artifacts/FacebookAI_xlm_roberta_base.zip
)

for f in "${FILES[@]}"; do
  python3 "$PROJ/scripts/par_download.py" "$S3/$f" "$RAW/$(basename "$f")" \
      --conns "$CONNS" --chunk-mb 8 || echo "[warn] failed: $(basename "$f")"
done

# Integrity before extraction. `--retry`/`-C -` resume a broken transfer, but a
# resumed download that ends up wrong still produces a complete-looking file, and
# a silently truncated bundle shows up much later as "missing articles". `unzip -t`
# verifies every member's CRC, which is the strongest check available -- the
# publisher ships no checksums. The sha256 goes in a sidecar so a rebuild can say
# which bundle changed (Config._raw_fingerprint uses sizes for the same reason).
for z in "$RAW"/*.zip; do
  [ -e "$z" ] || continue
  d="$RAW/$(basename "$z" .zip)"
  if [ -d "$d" ] && [ -f "$d/.verified" ]; then
    echo "[ebnerd] already extracted and verified: $(basename "$d")"; continue
  fi
  echo "[ebnerd] verifying $(basename "$z")"
  if ! unzip -tqq "$z" >/dev/null 2>&1; then
    echo "[FAIL] corrupt archive, not extracting: $z  (delete it and re-run to refetch)"
    continue
  fi
  [ -f "$z.sha256" ] || sha256sum "$z" | awk '{print $1}' > "$z.sha256"
  echo "[ebnerd] unzip $(basename "$z")"
  mkdir -p "$d" && unzip -q -o "$z" -d "$d" && date -u +%FT%TZ > "$d/.verified" \
    || echo "[warn] extraction failed: $z"
done

bash "$PROJ/scripts/normalize_raw.sh"
echo "[ebnerd] finished $(date)"
