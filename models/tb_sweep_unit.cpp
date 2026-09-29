// Streams sweep_vectors.py records into FPConvBlock or SegmentedFMAPipe, taken
// straight from a sweep config's generated Verilog, and checks every output.
//
// Build with -DUNIT_CONV (conv.bin) or -DUNIT_FMA (fma.bin); see
// run_sweep_check.sh. One record goes in per clock. The unit's latency is found
// from the first records rather than assumed, and then every output is compared
// against the record that entered that many cycles earlier.
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <vector>
#include "verilated.h"
#include VTOP_HEADER

#ifdef UNIT_CONV
#ifdef CONV_BLOCK
// A block build: every record also carries the lanes' Binary8p1uf scales.
struct Rec { uint8_t fmt, frm, flags, in_eew; uint32_t pad; uint64_t in, scale, expect, mask; };
static const char* kFile = "conv_block.bin";
#else
struct Rec { uint8_t fmt, frm, flags, in_eew; uint32_t pad; uint64_t in, expect, mask; };
static const char* kFile = "conv.bin";
#endif
static void drive(VTOP* t, const Rec& r, bool valid) {
    t->io_valid = valid;
    t->io_in = r.in;
#ifdef CONV_BLOCK
    t->io_scale = r.scale;
#endif
    t->io_in_eew = r.in_eew;
    t->io_widen = r.flags & 1;
    t->io_narrow = (r.flags >> 1) & 1;
    t->io_rto = (r.flags >> 2) & 1;
    t->io_sat = (r.flags >> 3) & 1;
    t->io_frm = r.frm;
    t->io_signed = 0; t->io_i2f = 0; t->io_f2i = 0; t->io_truncating = 0;
    t->io_in_altfmt = r.fmt & 1;
#ifndef NO_FMT_HI
    t->io_in_fmt_hi = r.fmt >> 1;   // absent when the build has only the pair
#endif
}
static uint64_t maskOf(const Rec& r) { return r.mask; }
static const char* kindOf(const Rec& r) {
    static char b[48];
    snprintf(b, sizeof b, "%s frm=%d%s%s", (r.flags & 1) ? "widen " : "narrow",
             r.frm, (r.flags & 4) ? " rod" : "", (r.flags & 8) ? " sat" : "");
    return b;
}
#else
struct Rec { uint8_t fmt, frm, op, addsub, mul, pad[3]; uint64_t a, b, c, expect; };
static const char* kFile = "fma.bin";
static void drive(VTOP* t, const Rec& r, bool valid) {
    t->io_valid = valid;
    t->io_frm = r.frm;
    t->io_op = r.op;
    t->io_addsub = r.addsub;
    t->io_mul = r.mul;
    t->io_a_eew = 0; t->io_b_eew = 0; t->io_c_eew = 0; t->io_out_eew = 0;
    t->io_widen = 0;
    t->io_altfmt = r.fmt & 1;
#ifndef NO_FMT_HI
    t->io_fmt_hi = r.fmt >> 1;
#endif
    t->io_a = r.a; t->io_b = r.b; t->io_c = r.c;
}
static uint64_t maskOf(const Rec&) { return ~0ull; }
static const char* kindOf(const Rec& r) {
    static char b[48];
    snprintf(b, sizeof b, "%s frm=%d", r.mul ? "mul" : (r.op ? "sub" : "add"), r.frm);
    return b;
}
#endif

int main(int argc, char** argv) {
    const char* path = argc > 1 ? argv[1] : kFile;
    FILE* f = fopen(path, "rb");
    if (!f) { perror(path); return 2; }
    std::vector<Rec> recs;
    Rec r;
    while (fread(&r, sizeof r, 1, f) == 1) recs.push_back(r);
    fclose(f);

    auto ctx = std::make_unique<VerilatedContext>();
    auto t = std::make_unique<VTOP>(ctx.get());
    auto tick = [&] { t->clock = 0; t->eval(); t->clock = 1; t->eval(); };

    Rec idle{};
    t->reset = 1;
    drive(t.get(), idle, false);
    for (int i = 0; i < 5; i++) tick();
    t->reset = 0;

    // out[k] is what the unit shows after record k has been clocked in.
    const int kMaxLat = 16;
    std::vector<uint64_t> out(recs.size() + kMaxLat);
    for (size_t k = 0; k < recs.size() + kMaxLat; k++) {
        drive(t.get(), k < recs.size() ? recs[k] : idle, k < recs.size());
        tick();
        out[k] = t->io_out;
    }

    // Latency: the delay under which the most of records 64..1087 line up.
    // (Scored, not all-or-nothing, so one wrong answer cannot hide it.)
    int lat = 0, best = -1;
    for (int L = 0; L < kMaxLat; L++) {
        int score = 0;
        for (size_t k = 64; k < 1088 && k < recs.size(); k++)
            score += !((out[k + L] ^ recs[k].expect) & maskOf(recs[k]));
        printf("   latency %2d: %4d of 1024 records line up\n", L, score);
        if (score > best) { best = score; lat = L; }
    }

    // Per format code: records and lanes that disagree.
    uint64_t bad[16] = {0}, n[16] = {0}, total = 0;
    int shown = 0;
    for (size_t k = 0; k < recs.size(); k++) {
        const Rec& q = recs[k];
        uint64_t diff = (out[k + lat] ^ q.expect) & maskOf(q);
        n[q.fmt]++;
        if (diff) {
            bad[q.fmt]++; total++;
            if (shown++ < 8)
                printf("   MISMATCH record %zu fmt=%d %s: got %016llx want %016llx\n", k, q.fmt,
                       kindOf(q), (unsigned long long)(out[k + lat] & maskOf(q)),
                       (unsigned long long)(q.expect & maskOf(q)));
        }
    }
    printf("   latency %d cycles\n", lat);
    for (int c = 0; c < 16; c++)
        if (n[c]) printf("   format code %d: %llu/%llu cycles agree\n", c,
                         (unsigned long long)(n[c] - bad[c]), (unsigned long long)n[c]);
    printf("   TOTAL MISMATCHES: %llu  (of %zu cycles)\n", (unsigned long long)total, recs.size());
    return total ? 1 : 0;
}
