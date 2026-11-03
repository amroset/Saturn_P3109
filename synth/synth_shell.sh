#!/usr/bin/env bash
# Synthesize one module's own logic, its submodules kept as black boxes.
#
#   synth_shell.sh <CONFIG> <module> [clock period ns]
#
# Same flow and caveats as synth_unit.sh, but only <module>.sv is mapped; every
# other file of its hierarchy is read with -lib, so submodule instances count
# as empty boxes. Use it for a module whose submodules cannot be synthesized
# here (VectorBackend instantiates the register file's memory macros), to
# measure the logic that lives in the module itself (e.g. the scale buffers).
set -euo pipefail
export LC_ALL=C

[ $# -ge 2 ] || { echo "usage: $0 <CONFIG> <module> [clock period ns]" >&2; exit 2; }
CONFIG=$1
TOP=$2
PERIOD=${3:-20}

HERE=$(cd "$(dirname "$0")" && pwd)
CY=$(cd "$HERE/../../.." && pwd)
GEN=$HERE/verilog/$CONFIG
OUT=$HERE/runs/$CONFIG/$TOP-shell
PY=${PYTHON:-$CY/.conda-env/bin/python}
YOSYS=${YOSYS:-$HOME/.conda-yosys/bin/yosys}
OPENROAD=${OPENROAD:-$HOME/.conda-openroad/bin/openroad}
PDK=${PDK:-$HOME/.conda-sky130/share/pdk/sky130A/libs.ref/sky130_fd_sc_hd}
LIB=$PDK/lib/sky130_fd_sc_hd__tt_025C_1v80.lib

"$HERE/gen_verilog.sh" "$CONFIG"
mkdir -p "$OUT"
rm -f "$OUT/summary.txt"
$PY "$HERE/../models/collect_hier.py" "$GEN" "$TOP" > "$OUT/files.txt"
SELF=$(grep "/$TOP.sv$" "$OUT/files.txt")
LIBS=$(grep -v "/$TOP.sv$" "$OUT/files.txt" | tr '\n' ' ')

cat > "$OUT/synth.ys" <<EOF
read_verilog -sv -DSYNTHESIS -lib $LIBS
read_verilog -sv -DSYNTHESIS $SELF
hierarchy -top $TOP
synth -top $TOP
dfflibmap -liberty $LIB
abc -D $((PERIOD * 1000)) -liberty $LIB
opt_clean -purge
tee -o $OUT/area.txt stat -liberty $LIB
EOF
$YOSYS -q -l "$OUT/yosys.log" "$OUT/synth.ys" >/dev/null 2>&1

CELLS=$(grep -m1 -oP "Number of cells:\s+\K[0-9]+" "$OUT/area.txt")
AREA=$(grep -m1 -oP "Chip area for module .*: \K[0-9.]+" "$OUT/area.txt")
FLOPS=$( (grep -oP "sky130_fd_sc_hd__(df|ed|sd)[a-z0-9_]+\s+\K[0-9]+" "$OUT/area.txt" || true) | paste -sd+ | bc)
printf "%-36s %-18s cells %7s  flops %6s  area %9s um^2 (own logic, submodules excluded)\n" \
  "$CONFIG" "$TOP" "$CELLS" "${FLOPS:-0}" "$AREA" | tee "$OUT/summary.txt"
