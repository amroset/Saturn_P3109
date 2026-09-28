#!/usr/bin/env bash
# Check the FMA's 8-bit rounder, at every core shape, against the exact reference.
#
#   ./run_fma_check.sh <dir with the .sv and fma_*.bin files>
#
# An 8-bit multiply-add can run on any of the five FMA core sizes, and each
# hands the rounder a raw number of a different shape -- so each shape gets its
# own Verilated wrapper and its own run. No Chipyard simulation involved.
set -eo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
work=${1:-.}
cd "$work"

# core -> exponent width of that core's type
declare -A EXPW=( [FP64]=11 [FP32]=8 [FP16]=5 [BF16]=8 [E5M3]=5 )

rc=0
for dom in ext fin; do
    D=$([ "$dom" = ext ] && echo Ext || echo Fin)
    for core in FP64 FP32 FP16 BF16 E5M3; do
        top="P3109FmaRound${core}${D}"
        sexpw=$(( ${EXPW[$core]} + 2 ))
        echo "### FMA rounder, $core core  ($([ "$dom" = ext ] && echo extended || echo finite) domain)"
        obj="obj_fma_${core}_${dom}"
        rm -rf "$obj"
        verilator --cc "$top.sv" --exe "$here/tb_p3109_fma_round.cpp" \
            --top-module "$top" --Mdir "$obj" -o sim \
            -CFLAGS "-DVTOP=V$top -DVTOP_HEADER='\"V$top.h\"' -DSEXP_W=$sexpw -O2" \
            --build -j 4 >/dev/null
        "./$obj/sim" "fma_${core}_${dom}.bin" || rc=1
        echo
    done
done
exit $rc
