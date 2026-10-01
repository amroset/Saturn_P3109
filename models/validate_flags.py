"""Check the unified rounder model's exception flags against flags_ref.py.

  1. conversions: every BF16 pattern x six rounding modes (the five frm modes
     and round-to-odd) x {binary8p4, binary8p3} x {extended, finite} x
     {SatNone, SatFinite}, with a signalling-NaN input raising invalid;
  2. the FMA's rounder: every 8-bit operand pair through multiply, add and
     subtract, on every core shape, in every frm mode, with the raw result
     presented both normalised and one binade lower (doShiftSigDown1).
"""
import os
import sys
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "benchmarks", "common-data-gen"))
sys.path.insert(0, HERE)

from fractions import Fraction                         # noqa: E402
from gfloat import decode_float                         # noqa: E402
from gfloat.formats import format_info_p3109            # noqa: E402
from gfloat.types import Domain, Signedness             # noqa: E402
from gfloat_ref import BF16                             # noqa: E402
from fma_ref import OPS, exact                          # noqa: E402
from fma_raw import CORES, raw_from_exact               # noqa: E402
from flags_ref import p3109_flags                       # noqa: E402
from unified_rounder import unified_round, raw_from_fn  # noqa: E402

MODES = (0, 1, 2, 3, 4, 6)
FMTS = (("p4", 4), ("p3", 3))
NAMES = ("NV", "DZ", "OF", "UF", "NX")


def fi_of(P, finite):
    return format_info_p3109(8, P, Signedness.Signed,
                             Domain.Finite if finite else Domain.Extended)


def bf16_exact(b):
    v = decode_float(BF16, b).fval
    if v != v:
        return ("nan",)
    if v in (float("inf"), float("-inf")):
        return ("inf", 1 if v < 0 else 0)
    return ("num", 1 if b >> 15 else 0, abs(Fraction(v)))


def is_snan(b):
    return (b >> 7) & 0xFF == 0xFF and b & 0x7F != 0 and not (b >> 6) & 1


def conv_job(args):
    fmt, P, finite, mode, sat = args
    fi, bad, first = fi_of(P, finite), 0, None
    for b in range(1 << 16):
        snan = is_snan(b)
        _, got = unified_round(raw_from_fn(b, 8, 8), fmt, mode, sat, finite, invalid_exc=snan)
        want = p3109_flags(bf16_exact(b), fi, mode, invalid=snan)
        if tuple(map(bool, got)) != want:
            bad += 1
            first = first or f"bf16 0x{b:04X}: got {fmt_flags(got)} want {fmt_flags(want)}"
    return f"conv {fi.name} mode {mode} sat={sat!s:5}", 65536, bad, first


def fma_job(args):
    fmt, P, finite, op, mode = args
    fi, bad, n, first = fi_of(P, finite), 0, 0, None
    for a in range(256):
        xa = exact(fi, a)
        for b in range(256):
            r = OPS[op](xa, exact(fi, b), mode)
            want = p3109_flags(r, fi, mode)
            for core, (ew, sw) in CORES.items():
                for shifted in ((False, True) if r[0] == "num" and r[2] else (False,)):
                    raw = raw_from_exact(r, ew, sw + 2, shifted)
                    _, got = unified_round(raw, fmt, mode, False, finite,
                                           in_exp_width=ew, in_sig_width=sw + 2,
                                           sig_msb_always_zero=False)
                    n += 1
                    if tuple(map(bool, got)) != want:
                        bad += 1
                        first = first or (f"{op} a=0x{a:02X} b=0x{b:02X} core={core} shifted={shifted}: "
                                          f"got {fmt_flags(got)} want {fmt_flags(want)}")
    return f"fma  {fi.name} {op} mode {mode}", n, bad, first


def fmt_flags(f):
    return "".join(name for name, on in zip(NAMES, f) if on) or "-"


if __name__ == "__main__":
    jobs_c = [(fmt, P, fin, m, sat) for fmt, P in FMTS for fin in (False, True)
              for m in MODES for sat in (False, True)]
    jobs_f = [(fmt, P, fin, op, m) for fmt, P in FMTS for fin in (False, True)
              for op in OPS for m in MODES if m != 6]
    total = bad_total = 0
    with Pool(os.cpu_count() or 1) as pool:
        for fn, jobs in ((conv_job, jobs_c), (fma_job, jobs_f)):
            for name, n, bad, first in pool.imap(fn, jobs):
                total += n
                bad_total += bad
                print(f"{name:40} {n - bad:8}/{n} agree" + (f"   first: {first}" if first else ""),
                      flush=True)
    print(f"\nTOTAL FLAG MISMATCHES: {bad_total} of {total}")
    sys.exit(1 if bad_total else 0)
