// The element-format code above altfmt: fmt = vtype[10:9] ## vtype.altfmt.
// With VTypeFmtBits = 2 (configs with opuFP4), vsetvli and vsetvl keep bits
// 10:9 and vtype stays legal; without it, any of those bits makes vtype
// illegal (vill), as before. Prints what the core did and checks it against
// what the config should do (pass "expect legal" with -DFMT_BITS=2).
#include <stdio.h>
#include <stdint.h>

#ifndef FMT_BITS
#define FMT_BITS 0
#endif

static uint64_t read_vtype(void) { uint64_t v; asm volatile("csrr %0, vtype" : "=r"(v)); return v; }

int main(void) {
    uint64_t vl, vt, bad = 0;
    // vsetvli, e8 m1, fmt 6 (vtype[10:9] = 11, altfmt 0): zimm = 0x600
    asm volatile(".insn i 0x57, 0x7, %0, %1, 0x600" : "=r"(vl) : "r"(32L));
    vt = read_vtype();
    printf("vsetvli fmt 6: vtype = 0x%016lx, vl = %lu\n", vt, vl);
    if (FMT_BITS >= 2) bad |= !(vt == 0x600 && vl == 32);
    else               bad |= !((vt >> 63) == 1 && vl == 0);
    // vsetvl (register form), e8 m1, fmt 3 (vtype[9] = 1, altfmt 1)
    asm volatile("vsetvl %0, %1, %2" : "=r"(vl) : "r"(32L), "r"(0x300L));
    vt = read_vtype();
    printf("vsetvl  fmt 3: vtype = 0x%016lx, vl = %lu\n", vt, vl);
    if (FMT_BITS >= 2) bad |= !(vt == 0x300 && vl == 32);
    else               bad |= !((vt >> 63) == 1 && vl == 0);
    // a bit above the format field stays reserved in every config
    asm volatile("vsetvl %0, %1, %2" : "=r"(vl) : "r"(32L), "r"(0x800L));
    vt = read_vtype();
    printf("vsetvl  bit 11: vtype = 0x%016lx, vl = %lu\n", vt, vl);
    bad |= !((vt >> 63) == 1 && vl == 0);
    // plain e8 m1 is legal everywhere
    asm volatile("vsetvl %0, %1, %2" : "=r"(vl) : "r"(32L), "r"(0x0L));
    vt = read_vtype();
    printf("vsetvl  fmt 0: vtype = 0x%016lx, vl = %lu\n", vt, vl);
    bad |= !(vt == 0 && vl == 32);
    printf(bad ? "FAILED\n" : "passed\n");
    return bad;
}
