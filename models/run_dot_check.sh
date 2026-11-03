#!/usr/bin/env bash
#
# Check one ScaledDotLane (vfdot4, docs/vfdot4.md) with Verilator: every code
# pair in one position, random and cancelling four-term sums, scales at FP32's
# overflow and underflow thresholds, both formats, all five rounding modes;
# see dot_vectors.py.
#
#   ./run_dot_check.sh <CONFIG> <workdir>
#
# CONFIG is a config with P3109Formats.dot, e.g. P3109BlockDotV256D128ShuttleConfig.
# Generate its Verilog first:
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

case "$config" in
	*Finite*Dot*|*Dot*Finite*) std=p3109-finite ;;
	*Dot*) std=p3109 ;;
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

"$GFLOAT_PYTHON" "$here/dot_vectors.py" "$work" "$std"
"$GFLOAT_PYTHON" "$here/collect_hier.py" "$gen" ScaledDotLane > "$work/dot.f"
verilator --cc -f "$work/dot.f" --exe "$here/tb_dot_lane.cpp" --top-module ScaledDotLane \
	--Mdir "$work/obj_dot" -o sim -Wno-fatal -Wno-lint -Wno-style \
	-CFLAGS "-O2" --build -j "$(nproc)" \
	> "$work/build_dot.log" 2>&1 || { echo "error: see $work/build_dot.log" >&2; exit 1; }
"$work/obj_dot/sim" "$work/dot.bin"
