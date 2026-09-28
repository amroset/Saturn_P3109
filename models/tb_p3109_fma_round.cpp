// Standalone Verilator testbench for the FMA's 8-bit rounder, at one core shape.
//
// Reads the 16-byte records models/dump_fma_expected.py writes -- one raw
// number per record, exactly as that FMA core would present it -- drives the
// P3109FmaRound<core><dom> wrapper, and compares against the exact reference.
//
// Built once per core and domain by run_fma_check.sh. SEXP_W is the width of
// the core's signed exponent field (expWidth + 2): a negative exponent must be
// masked to exactly that many bits before it is driven onto the port.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
#include "verilated.h"
#include VTOP_HEADER

#pragma pack(push, 1)
struct Rec {
    uint8_t  flags;   // isNaN | isInf<<1 | isZero<<2 | sign<<3 | altfmt<<4
    uint8_t  rm;
    uint8_t  want;
    uint8_t  pad;
    int32_t  sExp;
    uint64_t sig;
};
#pragma pack(pop)
static_assert(sizeof(Rec) == 16, "record layout");

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <fma_core_dom.bin>\n", argv[0]); return 2; }

    FILE* f = fopen(argv[1], "rb");
    if (!f) { perror("records"); return 2; }
    fseek(f, 0, SEEK_END);
    size_t n = ftell(f) / sizeof(Rec);
    fseek(f, 0, SEEK_SET);
    std::vector<Rec> recs(n);
    if (fread(recs.data(), sizeof(Rec), n, f) != n) { fprintf(stderr, "short read\n"); return 2; }
    fclose(f);

    Verilated::commandArgs(argc, argv);
    VTOP* dut = new VTOP;
    const uint32_t sexp_mask = (SEXP_W >= 32) ? 0xFFFFFFFFu : ((1u << SEXP_W) - 1);

    size_t bad = 0, bad_shown = 0;
    size_t per_fmt[2] = {0, 0}, bad_fmt[2] = {0, 0};
    for (size_t i = 0; i < n; i++) {
        const Rec& r = recs[i];
        int altfmt = (r.flags >> 4) & 1;
        dut->io_isNaN        = r.flags & 1;
        dut->io_isInf        = (r.flags >> 1) & 1;
        dut->io_isZero       = (r.flags >> 2) & 1;
        dut->io_sign         = (r.flags >> 3) & 1;
        dut->io_altfmt       = altfmt;
        dut->io_roundingMode = r.rm;
        dut->io_sExp         = (uint32_t)r.sExp & sexp_mask;
        dut->io_sig          = r.sig;
        dut->eval();
        unsigned got = dut->io_out & 0xFF;
        per_fmt[altfmt]++;
        if (got != r.want) {
            bad++; bad_fmt[altfmt]++;
            if (bad_shown++ < 5)
                printf("   MISMATCH %s rm=%u flags=0x%02X sExp=%d sig=0x%llX: got 0x%02X want 0x%02X\n",
                       altfmt ? "binary8p3" : "binary8p4", r.rm, r.flags, r.sExp,
                       (unsigned long long)r.sig, got, r.want);
        }
    }
    for (int a = 0; a < 2; a++)
        printf("   %-10s %9zu/%zu ok\n", a ? "binary8p3" : "binary8p4",
               per_fmt[a] - bad_fmt[a], per_fmt[a]);

    delete dut;
    printf("   TOTAL MISMATCHES: %zu  (of %zu cases)\n", bad, n);
    return bad ? 1 : 0;
}
