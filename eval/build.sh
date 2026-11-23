#!/usr/bin/env bash
#
# Build one benchmark with extra compile flags.
#
#   eval/build.sh <benchmark> <out.riscv> [flags...]
#
# For example: eval/build.sh opu-scaled-gemm /tmp/osg.riscv -DSCALEDMODE -DFP4
# Run from the Chipyard root after `source env.sh`.
set -eo pipefail

[ $# -ge 2 ] || { echo "usage: $0 <benchmark> <out.riscv> [flags...]" >&2; exit 2; }
bm=$1; mkdir -p "$(dirname "$2")"; out=$(cd "$(dirname "$2")" && pwd)/$(basename "$2"); shift 2
B=$(cd "$(dirname "${BASH_SOURCE[0]}")/../benchmarks" && pwd)
cmd=$(make -C "$B" -n -B "$bm.riscv" 2>/dev/null | grep -m1 -- "-o $bm.riscv") ||
	{ echo "no compile command for $bm (did you source env.sh?)" >&2; exit 1; }
(cd "$B" && eval "${cmd/-o $bm.riscv/$* -o $out}")
echo "built $out"
