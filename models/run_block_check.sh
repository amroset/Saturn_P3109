#!/usr/bin/env bash
# Check the generated P3109 block conversions against the gfloat reference.
#
#   ./run_block_check.sh <dir with the .sv and expected_*.bin files>
#
# Verilates each wrapper on its own -- no Chipyard simulation involved.
# ConvertFromBlock is exhaustive (both operands are 8 bits); ConvertToBlock
# sweeps every BF16 pattern against every scale.
set -eo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
work=${1:-.}
cd "$work"

rc=0
for dom in ext fin; do
    [ "$dom" = ext ] && label="extended" || label="finite"

    for dir in from to; do
        if [ "$dir" = from ]; then
            top=$([ "$dom" = ext ] && echo P3109FromBlockExt || echo P3109FromBlockFin)
            tb="$here/tb_p3109_from_block.cpp"
            what="ConvertFromBlock"
        else
            top=$([ "$dom" = ext ] && echo P3109ToBlockExt || echo P3109ToBlockFin)
            tb="$here/tb_p3109_to_block.cpp"
            what="ConvertToBlock"
        fi

        echo "### $what  ($label domain)"
        obj="obj_${dir}_${dom}"
        rm -rf "$obj"
        verilator --cc "$top.sv" --exe "$tb" \
            --top-module "$top" --Mdir "$obj" -o "sim_${dir}_${dom}" \
            -CFLAGS "-DVTOP=V$top -DVTOP_HEADER='\"V$top.h\"' -O2" \
            --build -j 4 >/dev/null
        "./$obj/sim_${dir}_${dom}" "expected_${dir}_${dom}.bin" || rc=1
        echo
    done
done
exit $rc
