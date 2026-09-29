#!/usr/bin/env bash
# Synthesize the vector FP units for every config given, in parallel.
#
#   run_all.sh <CONFIG>...
HERE=$(cd "$(dirname "$0")" && pwd)
UNITS=${UNITS:-"FPConvPipe FPFMAPipe"}
for C in "$@"; do "$HERE/gen_verilog.sh" "$C" || exit 1; done
for C in "$@"; do
  for U in $UNITS; do
    "$HERE/synth_unit.sh" "$C" "$U" > /dev/null 2>&1 &
  done
done
wait
for C in "$@"; do for U in $UNITS; do cat "$HERE/runs/$C/$U/summary.txt" 2>/dev/null || echo "$C $U FAILED"; done; done
