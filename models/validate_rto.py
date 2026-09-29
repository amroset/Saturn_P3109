"""Check the unified rounder model's round-to-odd against the standard.

Round-to-odd is the mode vfncvt.rod passes to the conversion rounder (code 6).
It is not an frm value in RISC-V, so it is kept out of gfloat_ref.FRM, which
the benchmark generators loop over.

Every BF16 pattern x {binary8p4, binary8p3} x {extended, finite} x {SatNone,
SatFinite}, compared against rto_ref.py (a transcription of P3109 4.7.3-4.7.5).

The reference is also checked against round-to-odd's defining property: an
inexact result inside the finite range has an odd last bit.
"""
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "benchmarks", "common-data-gen"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gfloat import decode_float                      # noqa: E402
from gfloat.formats import format_info_p3109         # noqa: E402
from gfloat.types import Domain, Signedness          # noqa: E402
from gfloat_ref import BF16                          # noqa: E402
from rto_ref import convert_bf16_odd                 # noqa: E402
from unified_rounder import convert_bf16, RODD       # noqa: E402

total_bad, total = 0, 0
for dom, finite in ((Domain.Extended, False), (Domain.Finite, True)):
    for fmt, P in (("p4", 4), ("p3", 3)):
        fi = format_info_p3109(8, P, Signedness.Signed, dom)
        for sat in (False, True):
            bad, kinds, prop_bad = 0, {}, 0
            for b in range(1 << 16):
                want = convert_bf16_odd(b, fi, sat)
                got = convert_bf16(b, fmt, RODD, sat, finite)
                total += 1

                # the reference's own sanity: inexact and in range -> odd
                x = decode_float(BF16, b).fval
                y = decode_float(fi, want).fval
                if (math.isfinite(x) and math.isfinite(y) and x != y
                        and abs(x) <= fi.max and want & 1 == 0):
                    prop_bad += 1

                if got != want:
                    bad += 1
                    k = ("overflow" if math.isfinite(x) and abs(x) > fi.max
                         else "infinite input" if math.isinf(x) else "other")
                    kinds[k] = kinds.get(k, 0) + 1
                    if kinds[k] == 1:
                        print(f"   first {k:14} {fi.name} sat={sat}: bf16 0x{b:04X} "
                              f"({x:g}) model 0x{got:02X} "
                              f"({decode_float(fi, got).fval:g})  standard 0x{want:02X} ({y:g})")
            total_bad += bad
            print(f"{fi.name:14} sat={sat!s:5}: {65536 - bad:5}/65536 agree"
                  f"{'  ' + str(kinds) if kinds else ''}"
                  f"{'   REFERENCE PROPERTY FAILS: ' + str(prop_bad) if prop_bad else ''}",
                  flush=True)

print(f"\nTOTAL DISAGREEMENTS: {total_bad} of {total}")
sys.exit(1 if total_bad else 0)
