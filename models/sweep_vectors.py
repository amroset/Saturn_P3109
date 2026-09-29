"""Test vectors for the precision sweep, driven into the real units.

    python sweep_vectors.py <outdir> <P for code 0> <P for code 1> [P for code 2 ...]

Writes two files of cycle-by-cycle stimulus with the expected output, for a
Verilator testbench that drives FPConvBlock and SegmentedFMAPipe straight from
a sweep config's generated Verilog, with the format code on their pins:

  conv.bin  FPConvBlock, one 32-byte record per cycle:
              u8 fmt, u8 frm, u8 flags, u8 in_eew, u32 0, u64 in, u64 expect, u64 mask
            flags: bit0 widen, bit1 narrow, bit2 rto, bit3 sat
    * widening: every code of every format -> BF16, four per cycle
    * narrowing: every BF16 pattern -> every format, in every rounding mode
      (the five frm modes and round-to-odd), with and without saturation

  fma.bin   SegmentedFMAPipe, one 40-byte record per cycle:
              u8 fmt, u8 frm, u8 op, u8 addsub, u8 mul, u8[3] 0, u64 a, u64 b, u64 c, u64 expect
    * every operand pair of every format, through multiply, add and subtract,
      eight pairs per cycle -- one per lane, and so one per core. In RNE each
      pair visits all eight lanes; in the other four modes each pair runs once,
      on a lane that rotates with the pair.

Every expected value comes from gfloat (conversions), rto_ref (round-to-odd) or
fma_ref (arithmetic, rounded once from the exact result) -- not from our model
of the rounder -- so the check is independent of the design.
All formats are in the extended domain, as in the sweep configs.
"""
import os
import struct
import sys
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "benchmarks", "common-data-gen"))
sys.path.insert(0, HERE)

from gfloat.formats import format_info_p3109          # noqa: E402
from gfloat.types import Domain, Signedness           # noqa: E402
from gfloat_ref import BF16, FRM, convert             # noqa: E402
from fma_ref import binary                            # noqa: E402
from rto_ref import convert_bf16_odd                  # noqa: E402

RODD = 6
FLAG_WIDEN, FLAG_NARROW, FLAG_RTO, FLAG_SAT = 1, 2, 4, 8


def fi_of(P):
    return format_info_p3109(8, P, Signedness.Signed, Domain.Extended)


def narrow_job(args):
    """All BF16 patterns for one (format code, mode, sat): four per record."""
    code, P, frm, sat = args
    fi, out = fi_of(P), bytearray()
    for base in range(0, 1 << 16, 4):
        lanes = range(base, base + 4)
        want = [convert_bf16_odd(b, fi, sat) if frm == RODD else
                convert(BF16, fi, b, dict((v[0], v[1]) for v in FRM.values())[frm], sat)
                for b in lanes]
        word = sum(b << (16 * i) for i, b in enumerate(lanes))
        exp = sum(w << (16 * i) for i, w in enumerate(want))
        flags = FLAG_NARROW | (FLAG_RTO if frm == RODD else 0) | (FLAG_SAT if sat else 0)
        out += struct.pack("<BBBBIQQQ", code, 0 if frm == RODD else frm, flags, 1, 0,
                           word, exp, 0x00FF00FF00FF00FF)
    return bytes(out)


def fma_job(args):
    """All operand pairs for one (format code, op, mode): eight per record."""
    code, P, op, frm, rotations = args
    fi, out = fi_of(P), bytearray()
    rnd = dict((v[0], v[1]) for v in FRM.values())[frm]
    pairs = [(a, b) for a in range(256) for b in range(256)]
    for rot in range(rotations):
        for base in range(0, len(pairs), 8):
            chunk = pairs[base:base + 8]
            # lane l carries pair (l + shift) of the chunk, so pairs move across cores
            shift = (rot + base // 8) % 8
            lanes = [chunk[(l + shift) % 8] for l in range(8)]
            want = [binary(op, fi, fi, a, b, rnd, False) for a, b in lanes]
            wa = sum(a << (8 * l) for l, (a, b) in enumerate(lanes))
            wb = sum(b << (8 * l) for l, (a, b) in enumerate(lanes))
            we = sum(w << (8 * l) for l, w in enumerate(want))
            if op == "mul":    # a * b (the pipe supplies c = 0)
                rec = (code, frm, 0, 0, 1, wa, wb, 0, we)
            elif op == "add":  # a * 1 + c (the pipe supplies b = 1)
                rec = (code, frm, 0, 1, 0, wa, 0, wb, we)
            else:              # a * 1 - c
                rec = (code, frm, 1, 1, 0, wa, 0, wb, we)
            out += struct.pack("<BBBBBxxxQQQQ", *rec)
    return bytes(out)


if __name__ == "__main__":
    outdir, precisions = sys.argv[1], [int(p) for p in sys.argv[2:]]
    os.makedirs(outdir, exist_ok=True)

    # Widening is small: build it here. Four codes per record, lanes 0,2,4,6.
    conv = bytearray()
    for code, P in enumerate(precisions):
        fi = fi_of(P)
        for base in range(0, 256, 4):
            lanes = range(base, base + 4)
            want = [convert(fi, BF16, c, dict((v[0], v[1]) for v in FRM.values())[0], False)
                    for c in lanes]
            word = sum(c << (16 * i) for i, c in enumerate(lanes))
            exp = sum(w << (16 * i) for i, w in enumerate(want))
            conv += struct.pack("<BBBBIQQQ", code, 0, FLAG_WIDEN, 0, 0,
                                word, exp, 0xFFFFFFFFFFFFFFFF)

    modes = [v[0] for v in FRM.values()]
    narrow = [(c, P, frm, sat) for c, P in enumerate(precisions)
              for frm in modes + [RODD] for sat in (False, True)]
    fma = [(c, P, op, frm, 8 if frm == 0 else 1) for c, P in enumerate(precisions)
           for op in ("mul", "add", "sub") for frm in modes]
    with Pool(os.cpu_count()) as pool:
        for chunk in pool.imap(narrow_job, narrow):
            conv += chunk
        with open(os.path.join(outdir, "conv.bin"), "wb") as f:
            f.write(conv)
        with open(os.path.join(outdir, "fma.bin"), "wb") as f:
            for chunk in pool.imap(fma_job, fma):
                f.write(chunk)
    print(f"conv.bin: {len(conv) // 32} cycles; "
          f"fma.bin: {os.path.getsize(os.path.join(outdir, 'fma.bin')) // 40} cycles")
