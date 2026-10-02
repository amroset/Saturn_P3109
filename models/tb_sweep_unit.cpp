// Drives FPConvBlock (-DUNIT_CONV) or SegmentedFMAPipe (-DUNIT_FMA), taken from
// a precision-sweep config's generated Verilog, with the records
// sweep_vectors.py writes, the format code on the units' pins. One record goes
// in per clock. The latency is found from the first records, then every output
// is compared, codes and exception flags, with the record that entered that
// many cycles earlier. Build with -DNO_FMT_HI for a config with only the pair,
// which has no format-code bits above altfmt.
#include <cstdint>
#include <cstdio>
#include <memory>
#include <vector>
#include "verilated.h"
#include VTOP_HEADER

#ifdef UNIT_CONV
struct Rec {
    uint8_t  fmt, frm, flags, in_eew;   // flags: bit0 widen, bit1 narrow, bit2 rto, bit3 sat
    uint32_t pad;
    uint64_t in, expect, mask, exc;
};
static_assert(sizeof(Rec) == 40, "record layout must match sweep_vectors.py");
static const int kLanes = 4, kExcStride = 2;   // output lane l's flags are io_exc_{2l}
static void drive(VTOP* t, const Rec& r, bool valid) {
    t->io_valid = valid;
    t->io_in = r.in;
    t->io_in_eew = r.in_eew;
    t->io_widen = r.flags & 1;
    t->io_narrow = (r.flags >> 1) & 1;
    t->io_rto = (r.flags >> 2) & 1;
    t->io_sat = (r.flags >> 3) & 1;
    t->io_frm = r.frm;
    t->io_signed = 0;
    t->io_i2f = 0;
    t->io_f2i = 0;
    t->io_truncating = 0;
    t->io_in_altfmt = r.fmt & 1;
#ifndef NO_FMT_HI
    t->io_in_fmt_hi = r.fmt >> 1;
#endif
}
static uint64_t mask_of(const Rec& r) { return r.mask; }
static void describe(const Rec& r, char* b, size_t n) {
    snprintf(b, n, "%s frm=%d%s%s", (r.flags & 1) ? "widen" : "narrow", r.frm,
             (r.flags & 4) ? " rod" : "", (r.flags & 8) ? " sat" : "");
}
#else
struct Rec {
    uint8_t  fmt, frm, op, addsub, mul, pad[3];
    uint64_t a, b, c, expect, exc;
};
static_assert(sizeof(Rec) == 48, "record layout must match sweep_vectors.py");
static const int kLanes = 8, kExcStride = 1;
static void drive(VTOP* t, const Rec& r, bool valid) {
    t->io_valid = valid;
    t->io_frm = r.frm;
    t->io_op = r.op;
    t->io_addsub = r.addsub;
    t->io_mul = r.mul;
    t->io_a_eew = 0;
    t->io_b_eew = 0;
    t->io_c_eew = 0;
    t->io_out_eew = 0;
    t->io_widen = 0;
    t->io_altfmt = r.fmt & 1;
#ifndef NO_FMT_HI
    t->io_fmt_hi = r.fmt >> 1;
#endif
    t->io_a = r.a;
    t->io_b = r.b;
    t->io_c = r.c;
}
static uint64_t mask_of(const Rec&) { return ~0ull; }
static void describe(const Rec& r, char* b, size_t n) {
    snprintf(b, n, "%s frm=%d", r.mul ? "mul" : (r.op ? "sub" : "add"), r.frm);
}
#endif

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <conv.bin|fma.bin>\n", argv[0]); return 2; }
    FILE* f = fopen(argv[1], "rb");
    if (!f) { perror(argv[1]); return 2; }
    fseek(f, 0, SEEK_END);
    long bytes = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (bytes <= 0 || bytes % sizeof(Rec)) { fprintf(stderr, "bad record file %s\n", argv[1]); return 2; }
    std::vector<Rec> recs(bytes / sizeof(Rec));
    if (fread(recs.data(), sizeof(Rec), recs.size(), f) != recs.size()) { perror("read"); return 2; }
    fclose(f);

    auto ctx = std::make_unique<VerilatedContext>();
    auto t = std::make_unique<VTOP>(ctx.get());
    const uint8_t* exc[8] = {&t->io_exc_0, &t->io_exc_1, &t->io_exc_2, &t->io_exc_3,
                             &t->io_exc_4, &t->io_exc_5, &t->io_exc_6, &t->io_exc_7};
    auto tick = [&] { t->clock = 0; t->eval(); t->clock = 1; t->eval(); };

    Rec idle{};
    t->reset = 1;
    drive(t.get(), idle, false);
    for (int i = 0; i < 5; i++) tick();
    t->reset = 0;

    // out[k], got_exc[k]: what the unit shows after record k has been clocked in
    const int kMaxLat = 16;
    std::vector<uint64_t> out(recs.size() + kMaxLat), got_exc(recs.size() + kMaxLat);
    for (size_t k = 0; k < recs.size() + kMaxLat; k++) {
        drive(t.get(), k < recs.size() ? recs[k] : idle, k < recs.size());
        tick();
        out[k] = t->io_out;
        uint64_t e = 0;
        for (int l = 0; l < kLanes; l++) e |= (uint64_t)(*exc[kExcStride * l] & 0x1F) << (8 * l);
        got_exc[k] = e;
    }

    // The latency under which the most of records 64..1087 line up (scored, so
    // one wrong answer cannot hide it). Stop if no latency stands out: records
    // with equal outputs can make two latencies score the same.
    int lat = 0, best = -1, second = -1;
    size_t window = 0;
    for (int L = 0; L < kMaxLat; L++) {
        int score = 0;
        window = 0;
        for (size_t k = 64; k < 1088 && k < recs.size(); k++, window++)
            score += !((out[k + L] ^ recs[k].expect) & mask_of(recs[k]));
        if (score > best) { second = best; best = score; lat = L; }
        else if (score > second) second = score;
    }
    printf("   latency %d cycles (%d of %zu records line up; next best %d)\n", lat, best, window, second);
    if (best == second || best < 0.99 * window) {
        fprintf(stderr, "error: cannot tell the latency from records 64..1087\n");
        return 2;
    }

    uint64_t bad[16] = {0}, n[16] = {0}, total = 0;
    for (size_t k = 0; k < recs.size(); k++) {
        const Rec& q = recs[k];
        uint64_t got = out[k + lat] & mask_of(q), want = q.expect & mask_of(q);
        n[q.fmt & 15]++;
        if (got != want || got_exc[k + lat] != q.exc) {
            if (total++ < 8) {
                char what[48];
                describe(q, what, sizeof what);
                printf("   MISMATCH record %zu fmt=%d %s: got %016llx exc %016llx, want %016llx exc %016llx\n",
                       k, q.fmt, what, (unsigned long long)got, (unsigned long long)got_exc[k + lat],
                       (unsigned long long)want, (unsigned long long)q.exc);
            }
            bad[q.fmt & 15]++;
        }
    }
    for (int c = 0; c < 16; c++)
        if (n[c]) printf("   format code %d: %llu/%llu cycles agree\n", c,
                         (unsigned long long)(n[c] - bad[c]), (unsigned long long)n[c]);
    printf("   TOTAL MISMATCHES: %llu  (of %zu cycles)\n", (unsigned long long)total, recs.size());
    return total ? 1 : 0;
}
