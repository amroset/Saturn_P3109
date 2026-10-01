"""Check the rounder model on every precision a sweep build can carry.

binary8p2 .. binary8p7, every BF16 pattern x six rounding modes (the five frm
modes and round-to-odd) x {extended, finite} x {SatNone, SatFinite}, codes
against gfloat and rto_ref, flags against flags_ref. Each format runs at two
datapath widths, the narrowest that holds it (or 4, the pair's) and the widest
(7): only delta and prec_shift change between them, so the two bracket every
set the sweep builds.
"""
import os
import sys
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "benchmarks", "common-data-gen"))
sys.path.insert(0, HERE)

from gfloat_ref import BF16, FRM, convert, p3109_format           # noqa: E402
from flags_ref import flags, bf16_exact, bf16_is_snan, RODD       # noqa: E402
from rto_ref import convert_bf16_odd                              # noqa: E402
from p3109_rounder import p3109_round, raw_from_fn                # noqa: E402

MODES = [(frm, rnd) for frm, rnd in FRM.values()] + [(RODD, None)]
WIDEST = 7


def job(args):
    P, sig_int, finite, sat = args
    fi, bad, first = p3109_format(P, finite), 0, None
    for b in range(1 << 16):
        snan = bf16_is_snan(b)
        raw = raw_from_fn(b, 8, 8)
        for frm, rnd in MODES:
            got = p3109_round(raw, P, frm, sat, finite, invalid_exc=snan, sig_int=sig_int)
            want_code = convert_bf16_odd(b, fi, sat) if frm == RODD else convert(BF16, fi, b, rnd, sat)
            want_flags = flags(bf16_exact(b), fi, frm, invalid=snan)
            if got[0] != want_code or tuple(map(bool, got[1])) != want_flags:
                bad += 1
                first = first or (f"bf16 0x{b:04X} mode {frm}: got 0x{got[0]:02X} {got[1]}, "
                                  f"want 0x{want_code:02X} {want_flags}")
    return P, sig_int, finite, sat, bad, first


if __name__ == "__main__":
    jobs = [(P, s, finite, sat) for P in range(2, 8) for s in sorted({max(P, 4), WIDEST})
            for finite in (False, True) for sat in (False, True)]
    n = len(MODES) << 16
    total = 0
    with Pool(min(len(jobs), os.cpu_count() or 1)) as pool:
        for P, s, finite, sat, bad, first in pool.imap(job, jobs):
            total += bad
            print(f"binary8p{P} {'finite  ' if finite else 'extended'} sat={sat!s:5} datapath {s}: "
                  f"{n - bad:6}/{n} agree" + (f"   first: {first}" if first else ""), flush=True)
    print(f"\nTOTAL MISMATCHES: {total} of {len(jobs) * n}")
    sys.exit(1 if total else 0)
