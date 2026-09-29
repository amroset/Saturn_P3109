#!/usr/bin/env bash
# Check a precision-sweep build's conversion and multiply-add units, in place.
#
#   ./run_sweep_check.sh <CONFIG> <workdir> <P for code 0> <P for code 1> [...]
#
# Takes FPConvBlock and SegmentedFMAPipe from the config's generated Verilog
# (make verilog CONFIG=... first), drives their format-code pins directly --
# no instruction can reach the extra formats yet -- and compares every output
# with sweep_vectors.py's expected values (gfloat / rto_ref / fma_ref).
set -eo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cy=$(cd "$here/../../.." && pwd)
config=$1; work=$2; shift 2
gen=$cy/sims/verilator/generated-src/chipyard.harness.TestHarness.$config/gen-collateral
py=${PY:-$HOME/venvs/gfloat/bin/python}
mkdir -p "$work"

if [ ! -f "$work/conv.bin" ] || [ ! -f "$work/fma.bin" ]; then
    "$py" "$here/sweep_vectors.py" "$work" "$@"
fi

# With only the pair there are no upper format-code bits, so no port for them.
nofmthi=$([ $# -le 2 ] && echo "-DNO_FMT_HI" || true)

rc=0
for unit in conv fma; do
    top=$([ $unit = conv ] && echo FPConvBlock || echo SegmentedFMAPipe)
    def=$([ $unit = conv ] && echo UNIT_CONV || echo UNIT_FMA)
    echo "### $top ($config, formats: $*)"
    obj="$work/obj_$unit"
    rm -rf "$obj"
    "$cy/.conda-env/bin/python" "$cy/generators/saturn/synth/collect_hier.py" "$gen" "$top" > "$work/$unit.f"
    verilator --cc -f "$work/$unit.f" --exe "$here/tb_sweep_unit.cpp" \
        --top-module "$top" --Mdir "$obj" -o sim -Wno-fatal -Wno-lint -Wno-style \
        -CFLAGS "-DVTOP=V$top -DVTOP_HEADER='\"V$top.h\"' -D$def $nofmthi -O2" \
        --build -j 8 >"$work/build_$unit.log" 2>&1 || { echo "   build failed, see $work/build_$unit.log"; rc=1; continue; }
    "$obj/sim" "$work/$unit.bin" || rc=1
    echo
done
exit $rc
