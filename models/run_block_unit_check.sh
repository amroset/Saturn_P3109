#!/usr/bin/env bash
# Check block scaling in the real conversion unit, with a different scale per lane.
#
#   ./run_block_unit_check.sh <block CONFIG> <workdir> <finite: 0|1>
#
# Takes FPConvBlock from the config's generated Verilog (make verilog first)
# and drives io_scale directly -- FPConvPipe ties it to 2^0, since no
# instruction delivers a scale yet. Expected values: block_vectors.py.
set -eo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cy=$(cd "$here/../../.." && pwd)
config=$1; work=$2; finite=$3
gen=$cy/sims/verilator/generated-src/chipyard.harness.TestHarness.$config/gen-collateral
py=${PY:-$HOME/venvs/gfloat/bin/python}
mkdir -p "$work"
[ -f "$work/conv_block.bin" ] || "$py" "$here/block_vectors.py" "$work" "$finite"

echo "### FPConvBlock with per-lane scales ($config)"
obj="$work/obj_block"
rm -rf "$obj"
"$cy/.conda-env/bin/python" "$cy/generators/saturn/synth/collect_hier.py" "$gen" FPConvBlock > "$work/block.f"
verilator --cc -f "$work/block.f" --exe "$here/tb_sweep_unit.cpp" \
    --top-module FPConvBlock --Mdir "$obj" -o sim -Wno-fatal -Wno-lint -Wno-style \
    -CFLAGS "-DVTOP=VFPConvBlock -DVTOP_HEADER='\"VFPConvBlock.h\"' -DUNIT_CONV -DCONV_BLOCK -DNO_FMT_HI -O2" \
    --build -j 8 >"$work/build_block.log" 2>&1 || { echo "   build failed, see $work/build_block.log"; exit 1; }
"$obj/sim" "$work/conv_block.bin"
