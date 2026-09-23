"""Exhaustively check the unified rounder against gfloat.

All 65,536 BF16 patterns x 5 rounding modes x 2 formats x 2 domains x
{sat off, sat on} = 5,242,880 conversions, compared against gfloat_ref.convert
-- the same reference the Verilator tests use.
"""
import os
import sys

sys.path.insert(0, "/home/amroset/Thesis/Chipyard_P3109/generators/saturn/benchmarks/common-data-gen")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gfloat.formats import format_info_p3109
from gfloat.types import Domain, Signedness
from gfloat_ref import BF16, FRM, convert
from unified_rounder import convert_bf16

total_bad = 0
for dom, finite in ((Domain.Extended, False), (Domain.Finite, True)):
    for fmt, P in (("p4", 4), ("p3", 3)):
        fi = format_info_p3109(8, P, Signedness.Signed, dom)
        for sat in (False, True):
            for name, (frm, rnd) in FRM.items():
                bad = []
                for b in range(1 << 16):
                    got = convert_bf16(b, fmt, frm, sat, finite)
                    want = convert(BF16, fi, b, rnd, sat)
                    if got != want:
                        bad.append((b, got, want))
                total_bad += len(bad)
                extra = ""
                if bad:
                    extra = "   e.g. bf16 0x%04X got 0x%02X want 0x%02X (%d cases)" % (
                        bad[0][0], bad[0][1], bad[0][2], len(bad))
                print(f"{fi.name:16} {'sat' if sat else '   '} {name}: "
                      f"{65536 - len(bad):5}/65536{extra}", flush=True)

print("\nTOTAL MISMATCHES:", total_bad)
sys.exit(1 if total_bad else 0)
