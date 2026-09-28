// Standalone Verilator testbench for ConvertToBlock (P3109 5.5.2).
//
// Sweeps every BF16 pattern against every Binary8p1uf scale, for both 8-bit
// formats, and compares against expected_to_<domain>.bin, which
// models/dump_block_expected.py writes from the gfloat-based reference.
//
// Rounding is (NearestTiesToEven, SatNone) throughout -- the projection
// specification 4.5 requires. The rounder's other modes are already swept
// exhaustively by the plain rounder testbench; what is new here is the scale.
//
// Built by run_block_check.sh.

#include <cstdio>
#include <cstdlib>
#include <vector>
#include "verilated.h"
#include VTOP_HEADER

static const size_t N_FMT = 2, N_SCALE = 256, N_BITS = 65536;

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <expected.bin>\n", argv[0]); return 2; }

    const size_t total = N_FMT * N_SCALE * N_BITS;
    FILE* f = fopen(argv[1], "rb");
    if (!f) { perror("expected"); return 2; }
    std::vector<unsigned char> expected(total);
    if (fread(expected.data(), 1, total, f) != total) {
        fprintf(stderr, "short expected file\n"); return 2;
    }
    fclose(f);

    Verilated::commandArgs(argc, argv);
    VTOP* dut = new VTOP;
    dut->io_roundingMode = 0;   // RNE
    dut->io_sat = 0;            // SatNone

    size_t i = 0, bad = 0;
    for (int altfmt = 0; altfmt < 2; altfmt++) {
        size_t bad_here = 0;
        for (int scale = 0; scale < 256; scale++) {
            for (int bits = 0; bits < 65536; bits++, i++) {
                dut->io_in = bits;
                dut->io_scale = scale;
                dut->io_altfmt = altfmt;
                dut->eval();
                unsigned got = dut->io_out & 0xFF;
                if (got != expected[i]) {
                    if (bad_here < 4)
                        printf("   MISMATCH %s scale=%d bf16 0x%04X: got 0x%02X want 0x%02X\n",
                               altfmt ? "binary8p3" : "binary8p4", scale, bits,
                               got, expected[i]);
                    bad_here++;
                }
            }
        }
        printf("   %-10s all 256 scales x 65536 patterns: %zu/%zu ok\n",
               altfmt ? "binary8p3" : "binary8p4",
               N_SCALE * N_BITS - bad_here, N_SCALE * N_BITS);
        bad += bad_here;
    }

    delete dut;
    printf("   TOTAL MISMATCHES: %zu  (of %zu cases)\n", bad, i);
    return bad ? 1 : 0;
}
