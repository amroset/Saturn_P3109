#!/usr/bin/env bash
#
# Check a precision-sweep build's FPConvBlock and SegmentedFMAPipe, codes and
# exception flags, with Verilator. The format code is driven on the units'
# pins, since no instruction reaches the extra formats yet.
#
#   ./run_sweep_check.sh <CONFIG> <workdir>
#
# CONFIG is one of the P3109Sweep*V256D128ShuttleConfig configs, or
# P3109V256D128ShuttleConfig for the pair on its usual reader. Generate its
# Verilog first:
#   make -C sims/verilator verilog CONFIG=<CONFIG>
#
# Environment overrides:
#   GFLOAT_PYTHON  interpreter that has gfloat installed
set -eo pipefail

usage() {
	echo "usage: $0 <CONFIG> <workdir>" >&2
	exit 2
}
[ $# -eq 2 ] || usage

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cydir=$(cd "$here/../../.." && pwd)
config=$1
work=$(mkdir -p "$2" && cd "$2" && pwd)
gen=$cydir/sims/verilator/generated-src/chipyard.harness.TestHarness.$config/gen-collateral
GFLOAT_PYTHON=${GFLOAT_PYTHON:-$HOME/venvs/gfloat/bin/python}

# Precisions in format-code order: 4 and 3, then the extras in the config's name
case "$config" in
	P3109V256D128ShuttleConfig) precisions="4 3" ;;
	P3109Sweep*V256D128ShuttleConfig)
		digits=${config#P3109Sweep}
		digits=${digits%V256D128ShuttleConfig}
		precisions="4 3 $(echo "$digits" | tr -d 34 | sed 's/./& /g')" ;;
	*) usage ;;
esac
if [ ! -d "$gen" ]; then
	echo "error: no $gen" >&2
	echo "       run: make -C $cydir/sims/verilator verilog CONFIG=$config" >&2
	exit 1
fi
if ! "$GFLOAT_PYTHON" -c 'import gfloat' 2>/dev/null; then
	echo "error: no gfloat in $GFLOAT_PYTHON; see benchmarks/common-data-gen/README.md" >&2
	exit 1
fi

# With only the pair there are no format-code bits above altfmt, so no port
nofmthi=$([ "$(echo $precisions | wc -w)" -le 2 ] && echo -DNO_FMT_HI || true)

echo "### $config: precisions $precisions"
"$GFLOAT_PYTHON" "$here/sweep_vectors.py" "$work" $precisions
rc=0
for unit in conv fma; do
	top=$([ $unit = conv ] && echo FPConvBlock || echo SegmentedFMAPipe)
	def=$([ $unit = conv ] && echo UNIT_CONV || echo UNIT_FMA)
	echo "### $top"
	"$GFLOAT_PYTHON" "$here/collect_hier.py" "$gen" "$top" > "$work/$unit.f"
	verilator --cc -f "$work/$unit.f" --exe "$here/tb_sweep_unit.cpp" --top-module "$top" \
		--Mdir "$work/obj_$unit" -o sim -Wno-fatal -Wno-lint -Wno-style \
		-CFLAGS "-O2 -DVTOP=V$top -DVTOP_HEADER='\"V$top.h\"' -D$def $nofmthi" --build -j "$(nproc)" \
		> "$work/build_$unit.log" 2>&1 || { echo "error: see $work/build_$unit.log" >&2; rc=1; continue; }
	"$work/obj_$unit/sim" "$work/$unit.bin" || rc=1
done
exit $rc
