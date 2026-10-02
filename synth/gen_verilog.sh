#!/usr/bin/env bash
# Re-emit a config's Verilog for Yosys.
#
#   gen_verilog.sh <CONFIG>
#
# Yosys cannot parse SystemVerilog packed arrays, which firtool emits by
# default. This reruns firtool on the FIRRTL that `make verilog` already left in
# sims/verilator, with the same flags plus disallowPackedArrays -- the option
# Chipyard itself adds under ENABLE_YOSYS_FLOW (common.mk) -- and writes the
# result here, so the simulator's Verilog is left alone.
#
# Environment override: FIRTOOL.
set -euo pipefail

[ $# -eq 1 ] || { echo "usage: $0 <CONFIG>" >&2; exit 2; }
CONFIG=$1
HERE=$(cd "$(dirname "$0")" && pwd)
CY=$(cd "$HERE/../../.." && pwd)
LONG=chipyard.harness.TestHarness.$CONFIG
BUILD=$CY/sims/verilator/generated-src/$LONG
OUT=$HERE/verilog/$CONFIG
FIRTOOL=${FIRTOOL:-$CY/.conda-env/riscv-tools/bin/firtool}

[ -f "$BUILD/$LONG.fir" ] || { echo "error: no FIRRTL for $CONFIG; run make -C sims/verilator verilog CONFIG=$CONFIG first" >&2; exit 1; }
if [ "$OUT/.done" -nt "$BUILD/$LONG.fir" ]; then exit 0; fi   # already up to date

rm -rf "$OUT"
mkdir -p "$OUT"
"$FIRTOOL" \
  --format=fir \
  --disable-annotation-classless \
  --disable-annotation-unknown \
  --lowering-options=emittedLineLength=2048,noAlwaysComb,disallowLocalVariables,verifLabels,disallowPortDeclSharing,locationInfoStyle=wrapInAtSquareBracket,disallowPackedArrays \
  --repl-seq-mem --repl-seq-mem-file="$OUT/mems.conf" \
  --annotation-file="$BUILD/$LONG.appended.anno.json" \
  --split-verilog \
  -o "$OUT" \
  "$BUILD/$LONG.fir" > "$OUT/firtool.log" 2>&1
touch "$OUT/.done"
