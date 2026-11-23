#!/usr/bin/env bash
#
# Build one benchmark at the eight code layouts behind every timing result.
#
#   eval/build_layouts.sh <benchmark> <outdir> <name> [flags...]
#
# Writes <outdir>/<name>-l0.riscv ... <name>-l7.riscv. The flags (for example
# -DREPS=4 -DSHARED) are added to the benchmark's normal compile command.
#
# Layout i inserts PAD[i] bytes before the kernels and, where the benchmark
# reads it, moves its arrays by DOFF = 576 * i bytes. METHOD selects the
# remaining flags:
#   aligned (default)  every label at 16 bytes and the arrays at a fixed address
#                      (-falign-labels=16, .bss.bench at 0x80100000)
#   pad                PAD only, as in opu-timing and the runs before 2026-11-10
#
# Run from the Chipyard root after `source env.sh`.
set -eo pipefail

[ $# -ge 3 ] || { echo "usage: $0 <benchmark> <outdir> <name> [flags...]" >&2; exit 2; }
bm=$1 dir=$2 name=$3; shift 3
mkdir -p "$dir"; out=$(cd "$dir" && pwd)
B=$(cd "$(dirname "${BASH_SOURCE[0]}")/../benchmarks" && pwd)
PADS=(0 4 12 20 28 36 48 60)

cmd=$(make -C "$B" -n -B "$bm.riscv" 2>/dev/null | grep -m1 -- "-o $bm.riscv") ||
	{ echo "no compile command for $bm (did you source env.sh?)" >&2; exit 1; }

for i in 0 1 2 3 4 5 6 7; do
	f="-DPAD=${PADS[$i]} -fno-toplevel-reorder"
	case ${METHOD:-aligned} in
		aligned) f="$f -DDOFF=$((576 * i)) -falign-labels=16 -Wl,--section-start=.bss.bench=0x80100000" ;;
		pad) ;;
		*) echo "METHOD must be aligned or pad" >&2; exit 2 ;;
	esac
	(cd "$B" && eval "${cmd/-o $bm.riscv/$* $f -o $out/$name-l$i.riscv}")
done
echo "built $out/$name-l{0..7}.riscv"
