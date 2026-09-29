"""Check the unified rounder model on every precision the sweep can build.

For binary8p2 .. binary8p7, every BF16 pattern x every rounding mode (the five
frm modes against gfloat, round-to-odd against rto_ref) x {SatNone, SatFinite}
x {extended, finite}, at two datapath widths: the narrowest one that holds the
format, and the widest (7, which holds binary8p7). A format must round the
same whatever the widest format beside it is -- only prec_shift and delta
change -- so the two widths bracket every set the sweep uses.

Runs in parallel, one job per (P, sig_int, domain, sat).
"""
import os
import sys
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "benchmarks", "common-data-gen"))
sys.path.insert(0, HERE)

from gfloat.formats import format_info_p3109          # noqa: E402
from gfloat.types import Domain, Signedness           # noqa: E402
from gfloat_ref import BF16, FRM, convert             # noqa: E402
from rto_ref import convert_bf16_odd                  # noqa: E402
from unified_rounder import unified_round, raw_from_fn, RODD   # noqa: E402

PRECISIONS = range(2, 8)
WIDEST = 7


def job(args):
    P, sig_int, finite, sat = args
    fi = format_info_p3109(8, P, Signedness.Signed,
                           Domain.Finite if finite else Domain.Extended)
    bad, first = 0, None
    for b in range(1 << 16):
        raw = raw_from_fn(b, 8, 8)
        for name, (frm, rnd) in list(FRM.items()) + [("rod", (RODD, None))]:
            got = unified_round(raw, P, frm, sat, finite, sig_int=sig_int)[0]
            want = convert_bf16_odd(b, fi, sat) if frm == RODD else convert(BF16, fi, b, rnd, sat)
            if got != want:
                bad += 1
                if first is None:
                    first = f"bf16 0x{b:04X} {name}: got 0x{got:02X} want 0x{want:02X}"
    return P, sig_int, finite, sat, bad, first


if __name__ == "__main__":
    jobs = [(P, s, finite, sat)
            for P in PRECISIONS
            for s in sorted({max(P, 4), WIDEST})
            for finite in (False, True)
            for sat in (False, True)]
    total = 0
    with Pool(min(len(jobs), os.cpu_count() or 1)) as pool:
        for P, s, finite, sat, bad, first in pool.imap_unordered(job, jobs):
            total += bad
            print(f"binary8p{P} {'finite  ' if finite else 'extended'} sat={sat!s:5} "
                  f"datapath {s}: {6 * 65536 - bad:6}/{6 * 65536} agree"
                  f"{'   first: ' + first if first else ''}", flush=True)
    print(f"\nTOTAL MISMATCHES: {total} of {len(jobs) * 6 * 65536}")
    sys.exit(1 if total else 0)
