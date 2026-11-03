// Drives one ScaledDotLane, taken from a vfdot4 config's generated Verilog,
// with the records dot_vectors.py writes: one record per cycle; each result
// leaves the lane three cycles after its operands (stage registers).
#include <cstdint>
#include <cstdio>
#include <memory>
#include <vector>
#include "verilated.h"
#include "VScaledDotLane.h"

#define LAT 3

struct Rec {
    uint8_t  alt, frm, sa, sb;
    uint32_t a, b, c, expect;
};
static_assert(sizeof(Rec) == 20, "record layout must match dot_vectors.py");

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <dot.bin>\n", argv[0]); return 2; }
    FILE* f = fopen(argv[1], "rb");
    if (!f) { perror(argv[1]); return 2; }
    fseek(f, 0, SEEK_END);
    long bytes = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (bytes <= 0 || bytes % sizeof(Rec)) { fprintf(stderr, "bad record file\n"); return 2; }
    std::vector<Rec> recs(bytes / sizeof(Rec));
    if (fread(recs.data(), sizeof(Rec), recs.size(), f) != recs.size()) { perror("read"); return 2; }
    fclose(f);

    auto ctx = std::make_unique<VerilatedContext>();
    auto t = std::make_unique<VScaledDotLane>(ctx.get());
    auto tick = [&] { t->clock = 0; t->eval(); t->clock = 1; t->eval(); };

    t->io_valid = 0;
    t->reset = 1;
    for (int i = 0; i < 5; i++) tick();
    t->reset = 0;

    // per altfmt: unscaled and scaled
    size_t bad = 0, n[2][2] = {{0, 0}, {0, 0}}, bad_n[2][2] = {{0, 0}, {0, 0}};
    for (size_t k = 0; k < recs.size() + LAT - 1; k++) {
        if (k < recs.size()) {
            const Rec& r = recs[k];
            t->io_valid = 1;
            t->io_altfmt = r.alt; t->io_frm = r.frm;
            t->io_a = r.a; t->io_b = r.b; t->io_c = r.c;
            t->io_sa = r.sa; t->io_sb = r.sb;
        } else {
            t->io_valid = 0;
        }
        tick();
        if (k + 1 < LAT) continue;
        const Rec& r = recs[k + 1 - LAT];
        t->eval();
        uint32_t got = t->io_out;
        int scaled = !(r.sa == 128 && r.sb == 128);
        n[r.alt][scaled]++;
        if (got != r.expect) {
            bad_n[r.alt][scaled]++;
            if (bad++ < 12)
                printf("   MISMATCH alt=%d frm=%d a=%08x b=%08x sa=%u sb=%u c=%08x: got %08x, want %08x\n",
                       r.alt, r.frm, r.a, r.b, r.sa, r.sb, r.c, got, r.expect);
        }
    }
    for (int a = 0; a < 2; a++)
        for (int s = 0; s < 2; s++)
            if (n[a][s])
                printf("   altfmt %d %s: %zu/%zu dot products agree\n", a, s ? "scaled  " : "unscaled",
                       n[a][s] - bad_n[a][s], n[a][s]);
    printf("   TOTAL MISMATCHES: %zu  (of %zu)\n", bad, recs.size());
    return bad ? 1 : 0;
}
