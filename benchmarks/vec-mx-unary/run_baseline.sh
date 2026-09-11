#!/usr/bin/env bash
#
# Reproduce the FP8 conversion test end to end: validate the reference model,
# regenerate the vectors, cross-compile, and run on Verilator.
#
#   ./run_baseline.sh              # OCP FP8 baseline
#   ./run_baseline.sh p3109        # P3109, both formats in the extended domain
#   ./run_baseline.sh p3109-finite # P3109, both formats in the finite domain
#   ./run_baseline.sh ocp --gen-only     # skip the 18-minute simulator step
#
# Every run is logged to results/<std>-<config>-<timestamp>.log, and
# results/latest.log points at the most recent one.
#
# Environment overrides:
#   GFLOAT_PYTHON  interpreter that has gfloat installed (needs >= 3.12)
#   CONFIG         Chipyard config whose simulator binary to run. By default it
#                  follows the standard: ocp -> MXV256D128ShuttleConfig,
#                  p3109 -> P3109V256D128ShuttleConfig,
#                  p3109-finite -> P3109FiniteV256D128ShuttleConfig
#   N              elements per array
#
# Note: do NOT add `set -u`. Chipyard's env.sh sources the conda hook, which
# reads unset variables and would abort.
set -eo pipefail

STD=ocp
GEN_ONLY=0
for arg in "$@"; do
	case "$arg" in
		ocp|p3109|p3109-finite) STD=$arg ;;
		--gen-only) GEN_ONLY=1 ;;
		*) echo "usage: $0 [ocp|p3109|p3109-finite] [--gen-only]" >&2; exit 2 ;;
	esac
done

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
bmarks=$(dirname "$here")
cydir=$(cd "$bmarks/../../.." && pwd)

# Each standard needs the simulator built for it: P3109 vectors run against the
# OCP hardware would "fail" for reasons that have nothing to do with the RTL.
case "$STD" in
	ocp)          default_config=MXV256D128ShuttleConfig ;;
	p3109)        default_config=P3109V256D128ShuttleConfig ;;
	p3109-finite) default_config=P3109FiniteV256D128ShuttleConfig ;;
esac
CONFIG=${CONFIG:-$default_config}
N=${N:-256}
GFLOAT_PYTHON=${GFLOAT_PYTHON:-$HOME/venvs/gfloat/bin/python}
sim=$cydir/sims/verilator/simulator-chipyard.harness-$CONFIG
results=$here/results
log=$results/$STD-$CONFIG-$(date +%Y%m%d-%H%M%S).log
mkdir -p "$results"

if ! "$GFLOAT_PYTHON" -c 'import gfloat' 2>/dev/null; then
	echo "error: no gfloat in $GFLOAT_PYTHON" >&2
	echo "       gfloat needs Python >= 3.12; see benchmarks/common-data-gen/README-gfloat.md" >&2
	exit 1
fi

exec > >(tee "$log") 2>&1
trap 'ln -sfn "$(basename "$log")" "$results/latest.log"' EXIT

echo "# $STD / $CONFIG / N=$N / $(date -Is)"
echo "# saturn $(git -C "$cydir/generators/saturn" describe --always --dirty 2>/dev/null)"
echo

echo "### 1/4  validate gfloat against the Spike golden file"
"$GFLOAT_PYTHON" "$here/gen_data/validate_gfloat.py"

echo
echo "### 2/4  generate data.S  (--std $STD, N=$N)"
"$GFLOAT_PYTHON" "$here/gen_data/gen_data.py" --std "$STD" -n "$N" -o "$here/data.S"
head -1 "$here/data.S"

echo
echo "### 3/4  cross-build the benchmark"
source "$cydir/env.sh"
make -C "$bmarks" vec-mx-unary.riscv

if [ "$GEN_ONLY" = 1 ]; then
	echo
	echo "--gen-only: stopping before the simulator."
	echo
	echo "log: $log"
	exit 0
fi

if [ ! -x "$sim" ]; then
	echo "error: no simulator at $sim" >&2
	echo "       build it with: make -C $cydir/sims/verilator CONFIG=$CONFIG" >&2
	exit 1
fi

echo
echo "### 4/4  run on Verilator  (config $CONFIG, ~18 min)"
time "$sim" "$bmarks/vec-mx-unary.riscv"

echo
echo "log: $log"
