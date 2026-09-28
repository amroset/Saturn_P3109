// Standalone Verilator testbench for ConvertFromBlock (P3109 5.5.1).
//
// Both operands are 8 bits wide -- the element and its Binary8p1uf scale -- so
// the sweep below is the entire input space: 256 x 256 for each format, twice
// over for the two domains. Nothing is sampled.
//
// Compared against expected_from_<domain>.bin, two bytes per case (BF16,
// little-endian), written by models/dump_block_expected.py.
//
// Built by run_block_check.sh.

#include <cstdio>
#include <cstdlib>
#include <vector>
#include "verilated.h"
#include VTOP_HEADER

static const size_t N_FMT = 2, N_SCALE = 256, N_CODE = 256;

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <expected.bin>\n", argv[0]); return 2; }

    const size_t cases = N_FMT * N_SCALE * N_CODE;
    FILE* f = fopen(argv[1], "rb");
    if (!f) { perror("expected"); return 2; }
    std::vector<unsigned char> expected(cases * 2);
    if (fread(expected.data(), 1, cases * 2, f) != cases * 2) {
        fprintf(stderr, "short expected file\n"); return 2;
    }
    fclose(f);

    Verilated::commandArgs(argc, argv);
    VTOP* dut = new VTOP;
    dut->io_roundingMode = 0;   // RNE

    size_t i = 0, bad = 0;
    for (int altfmt = 0; altfmt < 2; altfmt++) {
        size_t bad_here = 0;
        for (int scale = 0; scale < 256; scale++) {
            for (int code = 0; code < 256; code++, i++) {
                dut->io_in = code;
                dut->io_scale = scale;
                dut->io_altfmt = altfmt;
                dut->eval();
                unsigned got = dut->io_out & 0xFFFF;
                unsigned want = expected[2 * i] | (expected[2 * i + 1] << 8);
                if (got != want) {
                    if (bad_here < 4)
                        printf("   MISMATCH %s scale=%d code 0x%02X: got 0x%04X want 0x%04X\n",
                               altfmt ? "binary8p3" : "binary8p4", scale, code, got, want);
                    bad_here++;
                }
            }
        }
        printf("   %-10s all 256 scales x 256 codes: %zu/%zu ok\n",
               altfmt ? "binary8p3" : "binary8p4",
               N_SCALE * N_CODE - bad_here, N_SCALE * N_CODE);
        bad += bad_here;
    }

    delete dut;
    printf("   TOTAL MISMATCHES: %zu  (of %zu cases)\n", bad, i);
    return bad ? 1 : 0;
}
