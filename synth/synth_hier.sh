#!/usr/bin/env bash
# Per-module area of one Saturn unit on Sky130 (no timing).
#
#   synth_hier.sh <CONFIG> <top module>
#
# The same mapping as synth_unit.sh, but the hierarchy is kept, so Yosys
# reports cell area per module and how many times each is instantiated. Area
# is before buffering, and optimizations across module boundaries are lost,
# so the total runs a little above the flattened figure. Use it to see where
# area sits, and synth_unit.sh for the totals.
set -euo pipefail
export LC_ALL=C

CONFIG=$1
TOP=$2
HERE=$(cd "$(dirname "$0")" && pwd)
CY=$(cd "$HERE/../../.." && pwd)
GEN=$HERE/verilog/$CONFIG
OUT=$HERE/runs/$CONFIG/$TOP.hier
PY=$CY/.conda-env/bin/python
YOSYS=${YOSYS:-$HOME/.conda-yosys/bin/yosys}
PDK=${PDK:-$HOME/.conda-sky130/share/pdk/sky130A/libs.ref/sky130_fd_sc_hd}
LIB=$PDK/lib/sky130_fd_sc_hd__tt_025C_1v80.lib

"$HERE/gen_verilog.sh" "$CONFIG"
mkdir -p "$OUT"
$PY "$HERE/../models/collect_hier.py" "$GEN" "$TOP" > "$OUT/files.txt"

cat > "$OUT/synth.ys" <<EOF
read_verilog -sv -DSYNTHESIS $(tr '\n' ' ' < "$OUT/files.txt")
hierarchy -check -top $TOP
synth -top $TOP
dfflibmap -liberty $LIB
abc -D 20000 -liberty $LIB
opt_clean -purge
tee -o $OUT/area.txt stat -liberty $LIB -top $TOP
EOF
$YOSYS -q -l "$OUT/yosys.log" "$OUT/synth.ys" >/dev/null 2>&1
echo "$CONFIG $TOP done"
