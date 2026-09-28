"""Write the expected results of both block conversions, for the Verilator testbench.

Both directions are swept exhaustively over the scale, at the projection
specification P3109 4.5 actually requires -- (NearestTiesToEven, SatNone). The
rounder's other rounding modes and its saturating mode are already checked
exhaustively on their own by validate_unified.py; what is new here is the scale
path, so that is what this sweeps.

    expected_from_<dom>.bin   ConvertFromBlock, 2 bytes per case (BF16, LE)
      for fmt in (p4, p3):
        for scale in 0 .. 255:
          for code in 0 .. 255:
      = 2 * 256 * 256 * 2 bytes = 262,144 bytes

    expected_to_<dom>.bin     ConvertToBlock, 1 byte per case
      for fmt in (p4, p3):
        for scale in 0 .. 255:
          for bits in 0 .. 65535:
      = 2 * 256 * 65536 bytes = 33,554,432 bytes

The testbench walks the same loops in the same order.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from block_ref import convert_from_block, convert_to_block

out_dir = sys.argv[1] if len(sys.argv) > 1 else "."

for domain, finite in (("ext", False), ("fin", True)):
    buf = bytearray()
    for precision in (4, 3):                      # altfmt = 0, then 1
        for scale in range(256):
            for code in range(256):
                v = convert_from_block(code, scale, precision, finite)
                buf.append(v & 0xFF)
                buf.append((v >> 8) & 0xFF)
    path = os.path.join(out_dir, f"expected_from_{domain}.bin")
    with open(path, "wb") as f:
        f.write(buf)
    print(f"wrote {path}  ({len(buf)//2} cases)", flush=True)

for domain, finite in (("ext", False), ("fin", True)):
    buf = bytearray()
    for precision in (4, 3):
        for scale in range(256):
            for bits in range(1 << 16):
                buf.append(convert_to_block(bits, scale, precision, finite))
        print(f"  ...{domain} p{precision} done", flush=True)
    path = os.path.join(out_dir, f"expected_to_{domain}.bin")
    with open(path, "wb") as f:
        f.write(buf)
    print(f"wrote {path}  ({len(buf)} cases)", flush=True)
