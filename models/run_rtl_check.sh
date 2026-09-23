#!/usr/bin/env bash
# Check the generated P3109 rounder against the Python model, with Verilator.
#
#   ./run_rtl_check.sh <dir with the .sv and expected_*.bin files>
#
# Verilates each wrapper on its own -- no Chipyard simulation involved, so the
# whole thing takes seconds.
set -eo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
work=${1:-.}
cd "$work"

rc=0
for dom in ext fin; do
    top=$([ "$dom" = ext ] && echo P3109ConvExt || echo P3109ConvFin)
    echo "### $top  (${dom}ended domain)"
    rm -rf "obj_$dom"
    verilator --cc "$top.sv" --exe "$here/tb_p3109_rounder.cpp" \
        --top-module "$top" --Mdir "obj_$dom" -o "sim_$dom" \
        -CFLAGS "-DVTOP=V$top -DVTOP_HEADER='\"V$top.h\"' -O2" \
        --build -j 4 >/dev/null
    "./obj_$dom/sim_$dom" "expected_$dom.bin" || rc=1
    echo
done
exit $rc
