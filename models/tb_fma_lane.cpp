// Drives SegmentedFMAPipe, taken from a scaled-mode config's generated Verilog,
// with the records fma_vectors.py writes: one record per clock, 8 lanes of an
// 8-bit result or 4 of a BF16 result. The latency is found from the first
// records, then every lane's result (and an 8-bit result's flags) is checked.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <string>
#include <vector>
#include "verilated.h"
#include "VSegmentedFMAPipe.h"

struct Rec {
    uint8_t  cat, alt, frm, op, addsub, mul, a_eew, b_eew, c_eew, out_eew, widen, chk, pad[4];
    uint64_t a, b, c, expect, exc;
    uint8_t  scale[8][4];
};
static_assert(sizeof(Rec) == 88, "record layout must match fma_vectors.py");

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s <fma.bin> <fma.cats> [mismatches shown]\n", argv[0]); return 2; }
    size_t show = argc > 3 ? strtoul(argv[3], nullptr, 0) : 16;
    FILE* f = fopen(argv[1], "rb");
    if (!f) { perror(argv[1]); return 2; }
    fseek(f, 0, SEEK_END);
    long bytes = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (bytes <= 0 || bytes % sizeof(Rec)) { fprintf(stderr, "bad record file\n"); return 2; }
    std::vector<Rec> recs(bytes / sizeof(Rec));
    if (fread(recs.data(), sizeof(Rec), recs.size(), f) != recs.size()) { perror("read"); return 2; }
    fclose(f);
    std::vector<std::string> cats;
    std::ifstream cf(argv[2]);
    for (std::string l; std::getline(cf, l);) cats.push_back(l);

    auto ctx = std::make_unique<VerilatedContext>();
    auto t = std::make_unique<VSegmentedFMAPipe>(ctx.get());
    auto tick = [&] { t->clock = 0; t->eval(); t->clock = 1; t->eval(); };
    uint8_t* sc[8][4] = {
        {&t->io_scale_0_0, &t->io_scale_0_1, &t->io_scale_0_2, &t->io_scale_0_3},
        {&t->io_scale_1_0, &t->io_scale_1_1, &t->io_scale_1_2, &t->io_scale_1_3},
        {&t->io_scale_2_0, &t->io_scale_2_1, &t->io_scale_2_2, &t->io_scale_2_3},
        {&t->io_scale_3_0, &t->io_scale_3_1, &t->io_scale_3_2, &t->io_scale_3_3},
        {&t->io_scale_4_0, &t->io_scale_4_1, &t->io_scale_4_2, &t->io_scale_4_3},
        {&t->io_scale_5_0, &t->io_scale_5_1, &t->io_scale_5_2, &t->io_scale_5_3},
        {&t->io_scale_6_0, &t->io_scale_6_1, &t->io_scale_6_2, &t->io_scale_6_3},
        {&t->io_scale_7_0, &t->io_scale_7_1, &t->io_scale_7_2, &t->io_scale_7_3}};
    const uint8_t* exc[8] = {&t->io_exc_0, &t->io_exc_1, &t->io_exc_2, &t->io_exc_3,
                             &t->io_exc_4, &t->io_exc_5, &t->io_exc_6, &t->io_exc_7};

    t->io_valid = 0;
    t->reset = 1;
    for (int i = 0; i < 5; i++) tick();
    t->reset = 0;

    const int extra = 8;
    std::vector<uint64_t> out(recs.size() + extra), got_exc(recs.size() + extra);
    for (size_t k = 0; k < recs.size() + extra; k++) {
        if (k < recs.size()) {
            const Rec& r = recs[k];
            t->io_valid = 1;
            t->io_frm = r.frm; t->io_op = r.op; t->io_addsub = r.addsub; t->io_mul = r.mul;
            t->io_a_eew = r.a_eew; t->io_b_eew = r.b_eew; t->io_c_eew = r.c_eew;
            t->io_out_eew = r.out_eew; t->io_widen = r.widen; t->io_altfmt = r.alt;
            t->io_a = r.a; t->io_b = r.b; t->io_c = r.c;
            for (int l = 0; l < 8; l++)
                for (int s = 0; s < 4; s++) *sc[l][s] = r.scale[l][s];
        } else {
            t->io_valid = 0;
        }
        t->eval();
        out[k] = t->io_out;
        uint64_t e = 0;
        for (int l = 0; l < 8; l++) e |= uint64_t(*exc[l]) << 8 * l;
        got_exc[k] = e;
        tick();
    }

    int lat = 0, best = -1;
    size_t window = recs.size() < 2000 ? recs.size() : 2000;
    for (int L = 1; L < extra; L++) {
        int score = 0;
        for (size_t k = 0; k < window; k++) score += out[k + L] == recs[k].expect;
        if (score > best) { best = score; lat = L; }
    }
    printf("   latency %d cycles (%d of %zu records line up)\n", lat, best, window);

    size_t bad = 0, bad_exc = 0, nexc = 0, shown = 0;
    std::vector<size_t> n(cats.size()), bad_n(cats.size());
    for (size_t k = 0; k < recs.size(); k++) {
        const Rec& r = recs[k];
        int lanes = r.out_eew ? 4 : 8, w = 64 / lanes;
        uint64_t m = (w == 64 ? ~0ull : (1ull << w) - 1);
        for (int l = 0; l < lanes; l++) {
            uint64_t g = out[k + lat] >> (w * l) & m, x = r.expect >> (w * l) & m;
            bool exc_bad = false;
            if (r.chk >> l & 1) {
                nexc++;
                exc_bad = (got_exc[k + lat] >> 8 * l & 0x1f) != (r.exc >> 8 * l & 0x1f);
                bad_exc += exc_bad;
            }
            n[r.cat]++;
            if (g != x || exc_bad) {
                bad_n[r.cat]++;
                bad++;
                if (shown++ < show) {
                    const uint8_t* s = r.scale[r.out_eew ? 2 * l : l];
                    printf("   MISMATCH %s alt=%d frm=%d op=%d lane %d: a=%llx b=%llx c=%llx s=(%u,%u,%u,%u)"
                           " got %llx/%02llx want %llx/%02llx\n",
                           cats[r.cat].c_str(), r.alt, r.frm, r.op, l,
                           (unsigned long long)(r.a >> (w * l) & m), (unsigned long long)(r.b >> (w * l) & m),
                           (unsigned long long)(r.c >> (w * l) & m), s[0], s[1], s[2], s[3],
                           (unsigned long long)g, (unsigned long long)(got_exc[k + lat] >> 8 * l & 0x1f),
                           (unsigned long long)x, (unsigned long long)(r.exc >> 8 * l & 0x1f));
                }
            }
        }
    }
    for (size_t c = 0; c < cats.size(); c++)
        if (n[c]) printf("   %-28s %zu/%zu lanes agree\n", cats[c].c_str(), n[c] - bad_n[c], n[c]);
    printf("   flags checked on %zu lanes, %zu differ\n", nexc, bad_exc);
    printf("   TOTAL MISMATCHES: %zu  (of %zu lane operations)\n", bad,
           [&] { size_t s = 0; for (auto v : n) s += v; return s; }());
    return bad ? 1 : 0;
}
