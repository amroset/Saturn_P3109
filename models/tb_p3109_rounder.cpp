// Standalone Verilator testbench for the unified P3109 rounder.
//
// Sweeps every BF16 pattern through every rounding mode (round-to-odd included), both formats and both
// saturation settings, and compares the hardware against expected_<domain>.bin,
// which models/dump_expected.py writes from the Python model.
//
// Built by run_rtl_check.sh, which verilates one wrapper at a time.

#include <cstdio>
#include <cstdlib>
#include <vector>
#include "verilated.h"
#include VTOP_HEADER

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <expected.bin>\n", argv[0]); return 2; }

    FILE* f = fopen(argv[1], "rb");
    if (!f) { perror("expected"); return 2; }
    std::vector<unsigned char> expected(1572864);   // 65536 x 6 modes x 2 x 2
    if (fread(expected.data(), 1, expected.size(), f) != expected.size()) {
        fprintf(stderr, "short expected file\n"); return 2;
    }
    fclose(f);

    Verilated::commandArgs(argc, argv);
    VTOP* dut = new VTOP;

    size_t i = 0, bad = 0;
    // The five frm modes plus round-to-odd (6), which vfncvt.rod uses.
    const int   mode_codes[] = {0, 1, 2, 3, 4, 6};
    const char* modes[]      = {"rne", "rtz", "rdn", "rup", "rmm", "rod"};

    for (int sat = 0; sat < 2; sat++) {
        for (int altfmt = 0; altfmt < 2; altfmt++) {
            for (int frm = 0; frm < 6; frm++) {
                size_t bad_here = 0;
                for (int bits = 0; bits < 65536; bits++, i++) {
                    dut->io_in = bits;
                    dut->io_altfmt = altfmt;
                    dut->io_roundingMode = mode_codes[frm];
                    dut->io_sat = sat;
                    dut->eval();
                    unsigned got = dut->io_out & 0xFF;
                    if (got != expected[i]) {
                        if (bad_here == 0)
                            printf("   MISMATCH %s sat=%d %s: bf16 0x%04X got 0x%02X want 0x%02X\n",
                                   altfmt ? "binary8p3" : "binary8p4", sat, modes[frm],
                                   bits, got, expected[i]);
                        bad_here++;
                    }
                }
                printf("   %-10s sat=%d %s: %5zu/65536\n",
                       altfmt ? "binary8p3" : "binary8p4", sat, modes[frm],
                       65536 - bad_here);
                bad += bad_here;
            }
        }
    }

    delete dut;
    printf("\nTOTAL MISMATCHES: %zu  (of %zu cases)\n", bad, i);
    return bad ? 1 : 0;
}
