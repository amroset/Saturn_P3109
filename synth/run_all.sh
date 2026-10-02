#!/usr/bin/env bash
# Synthesize the vector FP units of every config given, in parallel, and print
# each unit's area and critical path (synth_unit.sh).
#
#   run_all.sh <CONFIG>...
#
# Each job logs to runs/<CONFIG>/<unit>.log. Exits non-zero if any job fails.
#
# Environment overrides: UNITS (default "FPConvPipe FPFMAPipe"), and those of
# synth_unit.sh.
set -eo pipefail

[ $# -ge 1 ] || { echo "usage: $0 <CONFIG>..." >&2; exit 2; }
HERE=$(cd "$(dirname "$0")" && pwd)
UNITS=${UNITS:-"FPConvPipe FPFMAPipe"}

for C in "$@"; do "$HERE/gen_verilog.sh" "$C"; done
pids=() jobs=()
for C in "$@"; do
  mkdir -p "$HERE/runs/$C"
  for U in $UNITS; do
    "$HERE/synth_unit.sh" "$C" "$U" > "$HERE/runs/$C/$U.log" 2>&1 &
    pids+=($!) jobs+=("$C $U")
  done
done

fail=0
for i in "${!pids[@]}"; do
  read -r C U <<< "${jobs[$i]}"
  if wait "${pids[$i]}"; then
    cat "$HERE/runs/$C/$U/summary.txt"
  else
    echo "$C $U FAILED, see runs/$C/$U.log"
    fail=1
  fi
done
exit $fail
