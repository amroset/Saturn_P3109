"""Write the expected P3109 codes for every conversion, for the Verilator testbench.

One byte per case, in this order (the testbench walks the same loops):

    for domain in (extended, finite):          # a separate file each
      for sat in (0, 1):
        for altfmt in (0, 1):                  # binary8p4, binary8p3
          for mode in (0, 1, 2, 3, 4, 6):      # RNE RTZ RDN RUP RMM, round-to-odd
            for bits in 0 .. 65535:            # every BF16 pattern

That is 65536 * 6 * 2 * 2 = 1,572,864 bytes per domain.

Mode 6 is round-to-odd, which the conversion unit passes to the rounder for
vfncvt.rod. It is not an frm value, which is why it is listed explicitly here
rather than taken from gfloat_ref.FRM.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from unified_rounder import convert_bf16

MODES = (0, 1, 2, 3, 4, 6)   # RNE RTZ RDN RUP RMM, round-to-odd

out_dir = sys.argv[1] if len(sys.argv) > 1 else "."

for domain, finite in (("ext", False), ("fin", True)):
    buf = bytearray()
    for sat in (False, True):
        for fmt in ("p4", "p3"):
            for mode in MODES:
                for bits in range(1 << 16):
                    buf.append(convert_bf16(bits, fmt, mode, sat, finite))
    path = os.path.join(out_dir, f"expected_{domain}.bin")
    with open(path, "wb") as f:
        f.write(buf)
    print(f"wrote {path}  ({len(buf)} cases)")
