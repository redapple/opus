#!/usr/bin/env bash
# Compare a stock and a patched opus_demo build:
#   - encoded and decoded output must be byte-identical (fails otherwise)
#   - on Linux, instruction counts (valgrind/callgrind) are reported
# Usage: compare.sh BASE_BUILD_DIR PATCH_BUILD_DIR
set -euo pipefail

BASE_DIR=$1
PATCH_DIR=$2
SUM=${GITHUB_STEP_SUMMARY:-/dev/null}
HERE=$(cd "$(dirname "$0")" && pwd)
WORK=$(mktemp -d)

sha() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | cut -d' ' -f1
  else shasum -a 256 "$1" | cut -d' ' -f1; fi
}

python3 "$HERE/gen_audio.py" "$WORK/clip.raw" 10
head -c $((48000 * 4 * 3)) "$WORK/clip.raw" > "$WORK/short.raw"

fail=0
{
  echo "### Bit-exactness (stock vs patched)"
  echo
  echo "| bitrate | frame (ms) | encoded | decoded |"
  echo "|---|---|---|---|"
} >> "$SUM"

for br in 24000 64000 128000; do
  for fs in 20 2.5; do
    for v in base patch; do
      d=$BASE_DIR; [ "$v" = patch ] && d=$PATCH_DIR
      "$d/opus_demo" -e audio 48000 2 "$br" -framesize "$fs" "$WORK/clip.raw" "$WORK/$v.opus" >/dev/null 2>&1
      "$d/opus_demo" -d 48000 2 "$WORK/$v.opus" "$WORK/$v.pcm" >/dev/null 2>&1
    done
    e=same; dd=same
    [ "$(sha "$WORK/base.opus")" = "$(sha "$WORK/patch.opus")" ] || { e=DIFFERENT; fail=1; }
    [ "$(sha "$WORK/base.pcm")" = "$(sha "$WORK/patch.pcm")" ] || { dd=DIFFERENT; fail=1; }
    echo "bitrate=$br frame=$fs encoded=$e decoded=$dd"
    echo "| $((br / 1000)) kbps | $fs | $e | $dd |" >> "$SUM"
  done
done

if command -v valgrind >/dev/null 2>&1; then
  {
    echo
    echo "### Instructions executed, encoder, 3 s stereo clip, 20 ms frames (callgrind)"
    echo
    echo "| bitrate | stock | patched | change |"
    echo "|---|---|---|---|"
  } >> "$SUM"
  for br in 24000 64000 128000; do
    ir=""
    for v in base patch; do
      d=$BASE_DIR; [ "$v" = patch ] && d=$PATCH_DIR
      n=$(valgrind --tool=callgrind --callgrind-out-file="$WORK/cg.$v.out" \
            "$d/opus_demo" -e audio 48000 2 "$br" -framesize 20 "$WORK/short.raw" "$WORK/x.opus" 2>&1 \
          | grep -o 'Collected : [0-9]*' | awk '{print $3}')
      ir="$ir $n"
    done
    set -- $ir
    pct=$(awk -v a="$1" -v p="$2" 'BEGIN { printf "%.2f", (p - a) * 100 / a }')
    echo "ir bitrate=$br stock=$1 patched=$2 change=${pct}%"
    echo "| $((br / 1000)) kbps | $1 | $2 | ${pct}% |" >> "$SUM"
  done
fi

exit $fail
