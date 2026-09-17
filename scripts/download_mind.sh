#!/usr/bin/env bash
# Download MIND (Microsoft News Dataset) into $DATA_ROOT/raw/mind.
#
# The official mirror `yjw1029/MIND` is HF-gated ("gated: auto") and is the only
# source of MINDlarge_test.zip, which the Codabench MIND leaderboard needs.
#   * authorised token -> official zips
#   * otherwise        -> MINDsmall_{train,dev} rebuilt from two ungated mirrors
#       huyva/MIND-small        : behaviors.tsv + news.tsv (byte-identical to official)
#       chuong090703/MIND-small : entity_embedding.vec + relation_embedding.vec
#     (in chuong's copy `test/` is the official dev split, and `train/`+`dev/`
#      are the official train split cut in two -- we take only its .vec files.)
#
# A token alone is not enough: accept the terms once at
# https://huggingface.co/datasets/yjw1029/MIND ("Agree and access repository"),
# else every resolve returns 403 GatedRepo.
#
# Usage: scripts/download_mind.sh [small|large|all]
set -uo pipefail
PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-$PROJ/data}"
RAW="$DATA_ROOT/raw/mind"
WHAT="${1:-small}"
TOKEN="${HF_TOKEN:-$(cat "$HOME/.cache/huggingface/token" 2>/dev/null)}"
mkdir -p "$RAW"

authorised() {  # is the token allowed to read the gated repo?
  [ -n "$TOKEN" ] || return 1
  [ "$(curl -s -o /dev/null -w '%{http_code}' -H "Authorization: Bearer $TOKEN" \
        https://huggingface.co/api/datasets/yjw1029/MIND/auth-check)" = "200" ]
}

hfget() {  # hfget <repo> <path-in-repo> <dest>
  local repo="$1" path="$2" dest="$3"
  [ -s "$dest" ] && { echo "[mind] have ${dest#$RAW/}"; return 0; }
  mkdir -p "$(dirname "$dest")"
  echo "[mind] $repo :: $path"
  curl -fL --retry 10 --retry-delay 5 -C - --progress-bar \
    ${TOKEN:+-H "Authorization: Bearer $TOKEN"} \
    -o "$dest" "https://huggingface.co/datasets/$repo/resolve/main/$path"
}

official() {
  for z in "$@"; do
    hfget yjw1029/MIND "$z" "$RAW/$z" || return 1
    # `-C -` resumes a broken transfer, but a resumed download that ends up wrong
    # is still a complete-looking file. `unzip -t` checks every member's CRC.
    if ! unzip -tqq "$RAW/$z" >/dev/null 2>&1; then
      echo "[FAIL] corrupt archive, not extracting: $RAW/$z (delete and re-run)" >&2
      return 1
    fi
    [ -f "$RAW/$z.sha256" ] || sha256sum "$RAW/$z" | awk '{print $1}' > "$RAW/$z.sha256"
    d="$RAW/${z%.zip}"
    if [ ! -f "$d/.source_official" ]; then   # (re)extract, overwriting mirror-built copies
      mkdir -p "$d"; unzip -q -o "$RAW/$z" -d "$d"; touch "$d/.source_official"
    fi
  done
}

mirror_small() {
  for s in train dev; do
    hfget huyva/MIND-small "$s/behaviors.tsv" "$RAW/MINDsmall_$s/behaviors.tsv"
    hfget huyva/MIND-small "$s/news.tsv"      "$RAW/MINDsmall_$s/news.tsv"
  done
  hfget chuong090703/MIND-small "MIND-small/train/entity_embedding.vec"   "$RAW/MINDsmall_train/entity_embedding.vec"
  hfget chuong090703/MIND-small "MIND-small/train/relation_embedding.vec" "$RAW/MINDsmall_train/relation_embedding.vec"
  hfget chuong090703/MIND-small "MIND-small/test/entity_embedding.vec"    "$RAW/MINDsmall_dev/entity_embedding.vec"
  hfget chuong090703/MIND-small "MIND-small/test/relation_embedding.vec"  "$RAW/MINDsmall_dev/relation_embedding.vec"
}

gate_hint() {
  echo "[mind] no authorised HF token."
  echo "       1. open https://huggingface.co/datasets/yjw1029/MIND and click 'Agree and access repository'"
  echo "       2. hf auth login   (or export HF_TOKEN=hf_...)"
}

case "$WHAT" in
  small)
    if authorised && official MINDsmall_train.zip MINDsmall_dev.zip; then
      echo "[mind] small: official zips"
    else
      gate_hint; echo "[mind] falling back to ungated mirrors for MIND-small"
      mirror_small
    fi ;;
  large)
    authorised || { gate_hint; echo "[mind] MINDlarge is gated-only -- cannot continue"; exit 1; }
    official MINDlarge_train.zip MINDlarge_dev.zip MINDlarge_test.zip ;;
  all) "$0" small; "$0" large ;;
  *) echo "usage: $0 [small|large|all]"; exit 2 ;;
esac

bash "$PROJ/scripts/normalize_raw.sh"
echo "[mind] done -> $RAW"
