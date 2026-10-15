"""Test vectors for a precision-sweep build's conversion and FMA units.

    sweep_vectors.py <outdir> <P for code 0> <P for code 1> [P for code 2 ...]

Writes, for tb_sweep_unit.cpp, one record per cycle with the format code on
the units' pins and the expected codes and exception flags:

  conv.bin  FPConvBlock, 40 bytes, four lanes:
              u8 fmt, u8 frm, u8 flags, u8 in_eew, u32 0, u64 in, u64 expect, u64 mask, u64 exc
            flags: bit0 widen, bit1 narrow, bit2 round-to-odd, bit3 sat
    * widening: every code of every format to BF16 (exact, no flags)
    * narrowing: every BF16 pattern to every format, the five frm modes and
      round-to-odd, SatNone and SatFinite

  fma.bin   SegmentedFMAPipe, 48 bytes, eight lanes (one per core):
              u8 fmt, u8 frm, u8 op, u8 addsub, u8 mul, u8[3] 0, u64 a, u64 b, u64 c, u64 expect, u64 exc
    * every operand pair of every format through multiply, add and subtract.
      In RNE each pair visits all eight lanes; in the other modes each pair
      runs once, on a lane that rotates with the pair.

exc holds each lane's 5-bit flags, one byte per lane. Codes come from gfloat,
rto_ref and fma_ref, flags from flags_ref. Every format is in the extended
domain, as in the sweep configs.
"""
import os
import struct
import sys
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "benchmarks", "common-data-gen"))
sys.path.insert(0, HERE)

from gfloat_ref import BF16, FRM, convert, p3109_format  # noqa: E402
from fma_ref import OPS, exact, project                  # noqa: E402
from flags_ref import flags, pack, bf16_exact, bf16_is_snan, RODD  # noqa: E402
from rto_ref import convert_bf16_odd                     # noqa: E402

WIDEN, NARROW, RTO, SAT = 1, 2, 4, 8
CONV = struct.Struct("<BBBBIQQQQ")
FMA = struct.Struct("<BBBBBxxxQQQQQ")
RND = {frm: rnd for frm, rnd in FRM.values()}


def lanes_word(vals, width):
    return sum(v << (width * l) for l, v in enumerate(vals))


def narrow_job(args):
    """Every BF16 pattern, four per record, for one (code, mode, sat)."""
    code, P, frm, sat = args
    fi, out = p3109_format(P, False), bytearray()
    rec_flags = NARROW | (RTO if frm == RODD else 0) | (SAT if sat else 0)
    for base in range(0, 1 << 16, 4):
        ins = range(base, base + 4)
        codes = [convert_bf16_odd(b, fi, sat) if frm == RODD else convert(BF16, fi, b, RND[frm], sat)
                 for b in ins]
        excs = [pack(flags(bf16_exact(b), fi, frm, invalid=bf16_is_snan(b))) for b in ins]
        out += CONV.pack(code, 0 if frm == RODD else frm, rec_flags, 1, 0, lanes_word(ins, 16),
                         lanes_word(codes, 16), 0x00FF00FF00FF00FF, lanes_word(excs, 8))
    return bytes(out)


def fma_job(args):
    """Every operand pair, eight per record, for one (code, op, mode)."""
    code, P, op, frm, rotations = args
    fi, out = p3109_format(P, False), bytearray()
    pairs = [(a, b) for a in range(256) for b in range(256)]
    for rot in range(rotations):
        for base in range(0, len(pairs), 8):
            chunk = pairs[base:base + 8]
            shift = (rot + base // 8) % 8            # pair (l + shift) rides lane l
            lanes = [chunk[(l + shift) % 8] for l in range(8)]
            codes, excs = [], []
            for a, b in lanes:
                xa, xb = exact(fi, a), exact(fi, b)
                r = OPS[op](xa, xb, RND[frm])
                invalid = r[0] == "nan" and "nan" not in (xa[0], xb[0])   # 0 x Inf, Inf - Inf
                codes.append(project(fi, r, RND[frm]))
                excs.append(pack(flags(r, fi, frm, invalid=invalid)))
            wa, wb = lanes_word([a for a, _ in lanes], 8), lanes_word([b for _, b in lanes], 8)
            if op == "mul":    # a * b (the pipe supplies c = 0)
                head, abc = (code, frm, 0, 0, 1), (wa, wb, 0)
            elif op == "add":  # a * 1 + c (the pipe supplies b = 1)
                head, abc = (code, frm, 0, 1, 0), (wa, 0, wb)
            else:              # a * 1 - c
                head, abc = (code, frm, 1, 1, 0), (wa, 0, wb)
            out += FMA.pack(*head, *abc, lanes_word(codes, 8), lanes_word(excs, 8))
    return bytes(out)


if __name__ == "__main__":
    if len(sys.argv) < 4:
        sys.exit("usage: sweep_vectors.py <outdir> <P for code 0> <P for code 1> [P for code 2 ...]")
    outdir, precisions = sys.argv[1], [int(p) for p in sys.argv[2:]]
    os.makedirs(outdir, exist_ok=True)

    conv = bytearray()
    for code, P in enumerate(precisions):
        fi = p3109_format(P, False)
        for base in range(0, 256, 4):
            ins = range(base, base + 4)
            conv += CONV.pack(code, 0, WIDEN, 0, 0, lanes_word(ins, 16),
                              lanes_word([convert(fi, BF16, c) for c in ins], 16),
                              0xFFFFFFFFFFFFFFFF, 0)

    modes = list(RND)
    narrow = [(c, P, frm, sat) for c, P in enumerate(precisions)
              for frm in modes + [RODD] for sat in (False, True)]
    fma = [(c, P, op, frm, 8 if frm == 0 else 1) for c, P in enumerate(precisions)
           for op in OPS for frm in modes]
    with Pool(os.cpu_count() or 1) as pool:
        for chunk in pool.imap(narrow_job, narrow):
            conv += chunk
        with open(os.path.join(outdir, "conv.bin"), "wb") as f:
            f.write(conv)
        n_fma = 0
        with open(os.path.join(outdir, "fma.bin"), "wb") as f:
            for chunk in pool.imap(fma_job, fma):
                f.write(chunk)
                n_fma += len(chunk) // FMA.size
    print(f"{outdir}: conv.bin {len(conv) // CONV.size} cycles, fma.bin {n_fma} cycles")
