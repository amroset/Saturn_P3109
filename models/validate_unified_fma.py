"""Drive the unified rounder with FMA-shaped inputs.

The conversion path always hands the rounder a BF16 value, so two branches never
get exercised there:

  * doShiftSigDown1.  A BF16 raw always has its significand in [1,2).  The FMA
    core's unrounded output can be in [1,4), which is the branch hardfloat calls
    doShiftSigDown1 -- and it interacts with prec_shift, which is exactly where
    the first derivation of this model was wrong.

  * the core widths.  An 8-bit element runs on whichever core sits in its lane:
    FP64, FP32, FP16, BF16 or E5M3.  The rounder sees a different input exponent
    and significand width in each case and must give the same answer.

Three checks:

  1. cross-core agreement, exhaustive: every operand pair, every op, both
     formats -- all five cores must agree with the exact reference.
  2. mode and domain coverage on selected pairs, in both normalisations, so
     doShiftSigDown1 is checked against truth rather than against itself.
  3. the normalisation invariant: the same value presented as sig in [1,2) and
     as sig in [2,4) with the exponent one lower must round identically.
"""
import os
import sys
from fractions import Fraction

sys.path.insert(0, "/home/amroset/Thesis/Chipyard_P3109/generators/saturn/benchmarks/common-data-gen")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gfloat.formats import format_info_p3109
from gfloat.types import Domain, Signedness
from gfloat_ref import FRM
from fma_ref import OPS, exact, project, _floor_log2, binary_inputs
from unified_rounder import unified_round
from fma_raw import CORES, raw_from_exact

def model(r, fmt, core, mode, sat, finite, unnormalized=False):
    ew, sw = CORES[core]
    raw = raw_from_exact(r, ew, sw + 2, unnormalized)
    return unified_round(raw, fmt, mode, sat, finite,
                         in_exp_width=ew, in_sig_width=sw + 2,
                         sig_msb_always_zero=False)[0]


FMTS = {"p4": 4, "p3": 3}
OPNAMES = ("mul", "add", "sub")
fail = 0

# --- 1. cross-core agreement, exhaustive ----------------------------------
print("1. every operand pair, every op, both formats, all five cores  (RNE, extended)")
for fmt, P in FMTS.items():
    fi = format_info_p3109(8, P, Signedness.Signed, Domain.Extended)
    for op in OPNAMES:
        bad, n = 0, 0
        for a in range(256):
            xa = exact(fi, a)
            for b in range(256):
                r = OPS[op](xa, exact(fi, b), 0)
                want = project(fi, r, list(FRM.values())[0][1], False)
                n += 1
                for core in CORES:
                    got = model(r, fmt, core, 0, False, False)
                    if got != want:
                        bad += 1
                        if bad == 1:
                            print(f"   MISMATCH {fmt} {op} core={core} "
                                  f"a=0x{a:02X} b=0x{b:02X} "
                                  f"got 0x{got:02X} want 0x{want:02X}")
        fail += bad
        print(f"   {fi.name} {op:4}: {n} pairs x 5 cores, {bad} mismatches", flush=True)

# --- 2. modes, domains, saturation on selected pairs ----------------------
print("\n2. selected operand pairs x 5 modes x 2 domains x sat on/off x 5 cores")
for dom, finite in ((Domain.Extended, False), (Domain.Finite, True)):
    for fmt, P in FMTS.items():
        fi = format_info_p3109(8, P, Signedness.Signed, dom)
        for op in OPNAMES:
            pairs, _cats = binary_inputs(op, fi, fi, 512, seed=1)
            bad = 0
            for a, b in pairs:
                for name, (frm, rnd) in FRM.items():
                    r = OPS[op](exact(fi, a), exact(fi, b), frm)
                    for sat in (False, True):
                        want = project(fi, r, rnd, sat)
                        for core in CORES:
                            # both normalisations, against the reference: this is
                            # where doShiftSigDown1 gets checked against truth
                            for shifted in (False, True):
                                got = model(r, fmt, core, frm, sat, finite, shifted)
                                if got != want:
                                    bad += 1
                                    if bad == 1:
                                        print(f"   MISMATCH {fi.name} {op} {name} "
                                              f"sat={sat} core={core} shifted={shifted} "
                                              f"a=0x{a:02X} b=0x{b:02X} "
                                              f"got 0x{got:02X} want 0x{want:02X}")
            fail += bad
            print(f"   {fi.name} {op:4}: {len(pairs)} pairs, {bad} mismatches", flush=True)

# --- 3. the normalisation invariant ---------------------------------------
print("\n3. same value, significand in [1,2) vs [2,4)  -- the doShiftSigDown1 path")
for fmt, P in FMTS.items():
    for dom, finite in ((Domain.Extended, False), (Domain.Finite, True)):
        fi = format_info_p3109(8, P, Signedness.Signed, dom)
        bad = 0
        for op in OPNAMES:
            for a, b in binary_inputs(op, fi, fi, 512, seed=2)[0]:
                r = OPS[op](exact(fi, a), exact(fi, b), 0)
                if r[0] != "num" or r[2] == 0:
                    continue
                for name, (frm, rnd) in FRM.items():
                    for sat in (False, True):
                        for core in CORES:
                            lo = model(r, fmt, core, frm, sat, finite, unnormalized=False)
                            hi = model(r, fmt, core, frm, sat, finite, unnormalized=True)
                            if lo != hi:
                                bad += 1
                                if bad == 1:
                                    print(f"   MISMATCH {fi.name} {op} {name} core={core}: "
                                          f"normalised 0x{lo:02X} vs shifted 0x{hi:02X}")
        fail += bad
        print(f"   {fi.name}: {bad} disagreements between the two normalisations", flush=True)

print("\nTOTAL MISMATCHES:", fail)
sys.exit(1 if fail else 0)
