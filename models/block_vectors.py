"""Test vectors for block scaling, driven into the real conversion unit.

    python block_vectors.py <outdir> <finite: 0|1>

Writes conv_block.bin for tb_sweep_unit.cpp (built with UNIT_CONV and
CONV_BLOCK), which drives FPConvBlock from a block config's generated Verilog
with a different scale on every lane. One 40-byte record per cycle:

    u8 fmt, u8 frm, u8 flags, u8 in_eew, u32 0, u64 in, u64 scale, u64 expect, u64 mask
    flags: bit0 widen, bit1 narrow, bit3 sat

The scale convention under test: a lane's Binary8p1uf scale sits in the same
byte as its 8-bit element -- bytes 0, 2, 4, 6 of the block's 64-bit word, in
both directions (the other scale bytes are filled with junk the unit must
ignore).

  * ConvertFromBlock (widening): every code x every scale, both formats, in all
    five rounding modes.
  * ConvertToBlock (narrowing): every BF16 pattern x every scale, both formats,
    RNE without saturation -- the specification P3109 4.5 requires -- plus the
    other four modes and saturation on a spread of 8 scales.

Four lanes per record, each with its own scale, so a lane that picked up the
wrong scale byte shows as a mismatch. Expected values come from block_ref.py
(a transcription of P3109 5.4-5.5 on top of gfloat).
"""
import os
import struct
import sys
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "benchmarks", "common-data-gen"))
sys.path.insert(0, HERE)

from gfloat_ref import FRM                              # noqa: E402
from block_ref import convert_from_block, convert_to_block   # noqa: E402

WIDEN, NARROW, SAT = 1, 2, 8
JUNK = 0x5A                        # odd scale bytes: must not be read
SPREAD = (0, 1, 64, 127, 128, 129, 200, 254)     # 255 is NaN, covered exhaustively
RND = {v[0]: v[1] for v in FRM.values()}


def scale_word(scales):
    return sum((s << (16 * l)) | (JUNK << (16 * l + 8)) for l, s in enumerate(scales))


def widen_job(args):
    code, P, finite, frm = args
    out = bytearray()
    combos = [(c, s) for c in range(256) for s in range(256)]
    for i in range(0, len(combos), 4):
        lanes = combos[i:i + 4]
        want = [convert_from_block(c, s, P, finite, RND[frm]) for c, s in lanes]
        out += struct.pack("<BBBBIQQQQ", code, frm, WIDEN, 0, 0,
                           sum(c << (16 * l) for l, (c, s) in enumerate(lanes)),
                           scale_word([s for c, s in lanes]),
                           sum(w << (16 * l) for l, w in enumerate(want)),
                           0xFFFFFFFFFFFFFFFF)
    return bytes(out)


def narrow_job(args):
    code, P, finite, frm, sat, scales, bits_lo, bits_hi = args
    out = bytearray()
    for b in range(bits_lo, bits_hi):
        for i in range(0, len(scales), 4):
            ss = scales[i:i + 4]
            # four neighbouring BF16 patterns, so lanes differ in value too
            vals = [(b + l) & 0xFFFF for l in range(4)]
            want = [convert_to_block(v, s, P, finite, RND[frm], sat) for v, s in zip(vals, ss)]
            out += struct.pack("<BBBBIQQQQ", code, frm, NARROW | (SAT if sat else 0), 1, 0,
                               sum(v << (16 * l) for l, v in enumerate(vals)),
                               scale_word(ss),
                               sum(w << (16 * l) for l, w in enumerate(want)),
                               0x00FF00FF00FF00FF)
    return bytes(out)


if __name__ == "__main__":
    outdir, finite = sys.argv[1], sys.argv[2] == "1"
    os.makedirs(outdir, exist_ok=True)
    fmts = [(0, 4), (1, 3)]
    jobs_w = [(c, P, finite, frm) for c, P in fmts for frm in RND]
    # exhaustive scale sweep at RNE / SatNone, split into slices of BF16 patterns
    all_scales = list(range(256))
    jobs_n = [(c, P, finite, 0, False, all_scales, lo, lo + 2048)
              for c, P in fmts for lo in range(0, 1 << 16, 2048)]
    jobs_n += [(c, P, finite, frm, sat, list(SPREAD), 0, 1 << 16)
               for c, P in fmts for frm in RND for sat in (False, True)
               if (frm, sat) != (0, False)]
    path = os.path.join(outdir, "conv_block.bin")
    n = 0
    with Pool(os.cpu_count()) as pool, open(path, "wb") as f:
        for chunk in pool.imap(widen_job, jobs_w):
            f.write(chunk); n += len(chunk)
        for chunk in pool.imap(narrow_job, jobs_n):
            f.write(chunk); n += len(chunk)
    print(f"conv_block.bin: {n // 40} cycles")
