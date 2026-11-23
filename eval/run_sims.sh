#!/usr/bin/env bash
#
# Run benchmarks on a config's Verilator simulator.
#
#   eval/run_sims.sh <CONFIG> <binary>...
#
# Each binary's output goes to <binary without .riscv>.<CONFIG>.log, next to
# it. JOBS simulations run at once (default 4); several at once slow each
# other down. Build the simulator first:
#   make -C sims/verilator CONFIG=<CONFIG>
#
# Run from the Chipyard root.
set -eo pipefail

[ $# -ge 2 ] || { echo "usage: $0 <CONFIG> <binary>..." >&2; exit 2; }
cfg=$1; shift
cy=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
sim=$cy/sims/verilator/simulator-chipyard.harness-$cfg
[ -x "$sim" ] || { echo "no simulator $sim" >&2; exit 1; }

run() {
	b=$(realpath "$1"); log=${b%.riscv}.$cfg.log
	# +loadmem loads the program directly; through the host interface it takes hours
	timeout 43200 "$sim" +permissive +loadmem="$b" +permissive-off "$b" > "$log" 2>&1
	echo "$(basename "$log") rc=$?"
}
export -f run; export sim cfg
printf '%s\n' "$@" | xargs -P "${JOBS:-4}" -I{} bash -c 'run {}'
