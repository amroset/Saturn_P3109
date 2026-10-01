#!/usr/bin/env bash
# Synthesize one Saturn unit onto Sky130 and time it.
#
#   synth_unit.sh <CONFIG> <top module> [clock period ns]
#
# Yosys maps the unit (flattened) onto sky130_fd_sc_hd at the typical corner.
# OpenROAD then buffers high-fanout nets (repair_design: Yosys leaves a control
# bit driving thousands of gates unbuffered, which on its own reads as ~50 ns)
# and reports cell area and the worst register-to-register path.
#
# Inputs and outputs are treated as registered elsewhere (zero external delay),
# so the numbers are the unit's own logic. No place-and-route: area is cell
# area, not die area, and timing uses the library's wire-load model rather
# than extracted wires.
set -euo pipefail
export LC_ALL=C                  # decimal points, not commas, in awk/printf

CONFIG=$1
TOP=$2
PERIOD=${3:-20}

HERE=$(cd "$(dirname "$0")" && pwd)
CY=$(cd "$HERE/../../.." && pwd)
GEN=$HERE/verilog/$CONFIG
OUT=$HERE/runs/$CONFIG/$TOP
PY=$CY/.conda-env/bin/python
YOSYS=${YOSYS:-$HOME/.conda-yosys/bin/yosys}
OPENROAD=${OPENROAD:-$HOME/.conda-openroad/bin/openroad}
PDK=${PDK:-$HOME/.conda-sky130/share/pdk/sky130A/libs.ref/sky130_fd_sc_hd}
LIB=$PDK/lib/sky130_fd_sc_hd__tt_025C_1v80.lib

"$HERE/gen_verilog.sh" "$CONFIG"
mkdir -p "$OUT"
$PY "$HERE/../models/collect_hier.py" "$GEN" "$TOP" > "$OUT/files.txt"

cat > "$OUT/synth.ys" <<EOF
read_verilog -sv -DSYNTHESIS $(tr '\n' ' ' < "$OUT/files.txt")
hierarchy -check -top $TOP
synth -flatten -top $TOP
dfflibmap -liberty $LIB
abc -D $((PERIOD * 1000)) -liberty $LIB
opt_clean -purge
tee -o $OUT/area.txt stat -liberty $LIB
write_verilog -noattr $OUT/netlist.v
EOF
$YOSYS -q -l "$OUT/yosys.log" "$OUT/synth.ys" >/dev/null 2>&1

cat > "$OUT/sta.tcl" <<EOF
read_lef $PDK/techlef/sky130_fd_sc_hd__nom.tlef
read_lef $PDK/lef/sky130_fd_sc_hd.lef
read_liberty $LIB
read_verilog $OUT/netlist.v
link_design $TOP
create_clock -period $PERIOD [get_ports clock]
set_input_delay 0 -clock clock [delete_from_list [all_inputs] [get_ports clock]]
set_output_delay 0 -clock clock [all_outputs]
set_max_fanout 16 [current_design]
repair_design
report_checks -path_delay max -fields {fanout} -digits 3
report_wns
report_design_area
EOF
$OPENROAD -no_init -exit "$OUT/sta.tcl" > "$OUT/timing.txt" 2>&1

CELLS=$(grep -m1 -oP "Number of cells:\s+\K[0-9]+" "$OUT/area.txt")
AREA=$(grep -oP "Design area \K[0-9.]+" "$OUT/timing.txt")
SLACK=$(grep -m1 -oP "^\s+\K-?[0-9.]+(?=\s+slack)" "$OUT/timing.txt")
DELAY=$(awk -v p="$PERIOD" -v s="$SLACK" 'BEGIN { printf "%.2f", p - s }')
printf "%-36s %-12s cells %7s  area %9s um^2  critical path %6s ns\n" \
  "$CONFIG" "$TOP" "$CELLS" "$AREA" "$DELAY" | tee "$OUT/summary.txt"
