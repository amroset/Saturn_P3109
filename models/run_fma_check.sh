#!/usr/bin/env bash
#
# Check SegmentedFMAPipe in scaled mode (docs/scaled-mode.md) with Verilator:
# every code pair of both formats on the widened lanes, as fma, add and mul
# with 8-bit results and as widening mul, add and fma with BF16 results, with
# sampled accumulators and scales, all five rounding modes; see fma_vectors.py.
#
#   ./run_fma_check.sh <CONFIG> <workdir>
#
# CONFIG is a config with P3109Formats.block, e.g. P3109BlockV256D128ShuttleConfig.
# Generate its Verilog first:
#   make -C sims/verilator verilog CONFIG=<CONFIG>
#
# Environment overrides:
#   GFLOAT_PYTHON  interpreter that has gfloat installed
#   JOBS           processes for the vectors (default 8)
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

case "$config" in
	*Block*Finite*|*Finite*Block*) std=p3109-finite ;;
	*Block*) std=p3109 ;;
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

"$GFLOAT_PYTHON" "$here/fma_vectors.py" "$work" "$std"
"$GFLOAT_PYTHON" "$here/collect_hier.py" "$gen" SegmentedFMAPipe > "$work/fma.f"
verilator --cc -f "$work/fma.f" --exe "$here/tb_fma_lane.cpp" --top-module SegmentedFMAPipe \
	--Mdir "$work/obj_fma" -o sim -Wno-fatal -Wno-lint -Wno-style \
	-CFLAGS "-O2" --build -j "${JOBS:-8}" \
	> "$work/build_fma.log" 2>&1 || { echo "error: see $work/build_fma.log" >&2; exit 1; }
"$work/obj_fma/sim" "$work/fma.bin" "$work/fma.cats"
