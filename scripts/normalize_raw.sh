#!/usr/bin/env bash
# Tidy the extracted raw trees (both datasets).
#
# Several EB-NeRD archives were zipped on macOS, so they carry __MACOSX/ and
# .DS_Store cruft, and a few bundles -- ebnerd_testset, the embedding artifacts,
# and the MINDlarge_* zips -- unpack into a redundant nested dir of the same
# name. Flatten those so every bundle looks like
#   <bundle>/{articles.parquet,train/,validation/,test/}   (EB-NeRD)
#   <bundle>/{behaviors.tsv,news.tsv,*.vec}                (MIND)
set -uo pipefail
PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW="${DATA_ROOT:-$PROJ/data}/raw"

find "$RAW" -name '__MACOSX' -type d -prune -exec rm -rf {} + 2>/dev/null
find "$RAW" -name '.DS_Store' -delete 2>/dev/null

for d in "$RAW"/*/*/; do
  d="${d%/}"; inner="$d/$(basename "$d")"
  if [ -d "$inner" ]; then
    echo "[normalize] flattening $(basename "$d")"
    mv "$inner"/* "$d"/ 2>/dev/null && rmdir "$inner"
  fi
done

echo "[normalize] layout:"
for d in "$RAW"/*/*/; do
  [ -d "$d" ] || continue
  printf "  %-42s %s\n" "$(basename "${d%/}")" "$(ls "$d" | head -6 | tr '\n' ' ')"
done
