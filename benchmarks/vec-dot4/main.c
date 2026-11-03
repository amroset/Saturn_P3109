// vfdot4 (docs/vfdot4.md) on a build with P3109Formats.dot
// (P3109BlockDotV256D128ShuttleConfig, or its Buf4 variant). Each test runs
// vfdot4.vv or .vx at SEW 32, strip by strip, and checks every FP32 lane bit
// for bit against gen_data.py's reference (data.S, tests.h).
//
// Registers: vs2 v8 (group of LMUL registers), its scales in the group after
// it (v9, v10, v12 at LMUL 1, 2, 4); vs1 v16, scales v17, v18, v20; vd v24.
// .vx's scalar is in t0 (x5), loaded with ld: codes in bits 31:0, scale in 39:32.
// -DSHARED (P3109BlockDotSharedV256D128ShuttleConfig, the explicit-operand
// emulation): the scales are in v30 (vs2) and v31 (vs1) instead.
#pragma GCC optimize("no-tree-vectorize")
#include <stdio.h>
#include <stdint.h>
#include <stddef.h>

#define N 64     // lanes per test: gen_data.py's N
#define LANES 8  // FP32 lanes per register (VLEN 256)

static uint32_t res[N] __attribute__((aligned(64)));
static uint8_t wrong[32] __attribute__((aligned(64))); // zero scales

// vtype: e32, LMUL 2^lm, ta, ma unless mu; altfmt (bit 8); scaled (bit 11),
// log2 E (bits 14:12)
#define VT32(lm, alt, sc, lg, mu) (0x50UL | (lm) | ((unsigned long)!(mu) << 7) | \
	((unsigned long)(alt) << 8) | ((unsigned long)(sc) << 11) | ((unsigned long)(lg) << 12))
#define VT8 0xC0UL
static inline void vsetl(unsigned long avl, unsigned long vt) {
	asm volatile("vsetvl x0, %0, %1" : : "r"(avl), "r"(vt));
}
#define SETFRM(rm) asm volatile("csrw frm, %0" : : "r"((unsigned long)(rm)))
#define INSN_VV   ".insn r 0x57, 0x1, 0x6B, x24, x16, x8"   // vfdot4.vv v24, v8, v16
#define INSN_VV_M ".insn r 0x57, 0x1, 0x6A, x24, x16, x8"   // vfdot4.vv v24, v8, v16, v0.t
#define INSN_VX   ".insn r 0x57, 0x5, 0x6B, x24, x5, x8"    // vfdot4.vx v24, v8, t0
#define INSN_VX_M ".insn r 0x57, 0x5, 0x6A, x24, x5, x8"

// scale registers of vs2 and vs1 for LMUL 2^lm (vle8 at e8 m1: one register)
static void ld_scales2(int lm, const uint8_t *p) {
#ifdef SHARED
	(void)lm; asm volatile("vle8.v v30, (%0)" : : "r"(p) : "memory");
#else
	if (lm == 0) asm volatile("vle8.v v9, (%0)" : : "r"(p) : "memory");
	else if (lm == 1) asm volatile("vle8.v v10, (%0)" : : "r"(p) : "memory");
	else asm volatile("vle8.v v12, (%0)" : : "r"(p) : "memory");
#endif
}
static void ld_scales1(int lm, const uint8_t *p) {
#ifdef SHARED
	(void)lm; asm volatile("vle8.v v31, (%0)" : : "r"(p) : "memory");
#else
	if (lm == 0) asm volatile("vle8.v v17, (%0)" : : "r"(p) : "memory");
	else if (lm == 1) asm volatile("vle8.v v18, (%0)" : : "r"(p) : "memory");
	else asm volatile("vle8.v v20, (%0)" : : "r"(p) : "memory");
#endif
}

static int check(const char *name, const uint32_t *want) {
	int bad = 0;
	for (int i = 0; i < N; i++)
		if (res[i] != want[i] && bad++ < 4)
			printf("MISMATCH %s [%d]: got %08x, want %08x\n", name, i, (unsigned)res[i], (unsigned)want[i]);
	printf("%s: %d of %d differ\n", name, bad, N);
	return bad != 0;
}

// One strip: operands, scales, mask; the instruction under (sc, lg, mu); vd out
#define STRIP(lm, alt, sc, lg, masked, LOADS, OP) \
	for (size_t i = 0; i < N; i += (LANES << (lm))) { \
		size_t q = i / (LANES << (lm)); \
		vsetl(LANES << (lm), VT32(lm, alt, 0, 0, 0)); \
		asm volatile("vle32.v v8, (%0)" : : "r"(&a[i]) : "memory"); \
		asm volatile("vle32.v v24, (%0)" : : "r"(&c[i]) : "memory"); \
		LOADS \
		vsetl(32, VT8); \
		ld_scales2(lm, &sa[32 * q]); \
		if (masked) asm volatile("vlm.v v0, (%0)" : : "r"(&m[4 * q]) : "memory"); \
		vsetl(LANES << (lm), VT32(lm, alt, sc, lg, masked)); \
		OP \
		vsetl(LANES << (lm), VT32(lm, alt, 0, 0, 0)); \
		asm volatile("vse32.v v24, (%0)" : : "r"(&res[i]) : "memory"); \
	}

#define DOT_VV(t, sc, alt, rm, lg, lm, masked) \
	extern uint32_t t##_a[], t##_b[], t##_c[], t##_out[]; extern uint8_t t##_sa[], t##_sb[]; \
	static int run_##t(void) { \
		const uint32_t *a = t##_a, *b = t##_b, *c = t##_c; const uint8_t *sa = t##_sa, *sb = t##_sb; \
		const uint8_t *m = 0; (void)m; \
		DOT_VV_MASKS_##masked(t) \
		SETFRM(rm); \
		STRIP(lm, alt, sc, lg, masked, \
			asm volatile("vle32.v v16, (%0)" : : "r"(&b[i]) : "memory"); \
			vsetl(32, VT8); ld_scales1(lm, &sb[32 * (i / (LANES << (lm)))]);, \
			if (masked) asm volatile(INSN_VV_M); else asm volatile(INSN_VV);) \
		return check(#t, t##_out); \
	}
#define DOT_VX(t, sc, alt, rm, lg, lm, masked) \
	extern uint32_t t##_a[], t##_f[], t##_c[], t##_out[]; extern uint8_t t##_sa[]; \
	static int run_##t(void) { \
		const uint32_t *a = t##_a, *f = t##_f, *c = t##_c; const uint8_t *sa = t##_sa; \
		const uint8_t *m = 0; (void)m; \
		DOT_VV_MASKS_##masked(t) \
		SETFRM(rm); \
		STRIP(lm, alt, sc, lg, masked, , \
			if (masked) asm volatile("ld t0, 0(%0)\n\t" INSN_VX_M : : "r"(&f[2 * q]) : "t0", "memory"); \
			else asm volatile("ld t0, 0(%0)\n\t" INSN_VX : : "r"(&f[2 * q]) : "t0", "memory");) \
		return check(#t, t##_out); \
	}
#define DOT_VV_MASKS_0(t)
#define DOT_VV_MASKS_1(t) extern uint8_t t##_m[]; m = t##_m;
#include "tests.h"

// Scales kept across instructions: fill the buffer with wrong scales (code 0),
// overwrite vs2's scale register with the right ones (vmv.v.v), expect the
// right result; then again with no write in between (a buffer hit).
#define T vv_s_a0_rne_e1_m1
#define CAT_(x, y) x##y
#define CAT(x, y) CAT_(x, y)
#ifdef SHARED
#define SC2 "v30"
#else
#define SC2 "v9"
#endif
extern uint32_t CAT(T, _a)[], CAT(T, _b)[], CAT(T, _c)[], CAT(T, _out)[]; extern uint8_t CAT(T, _sa)[], CAT(T, _sb)[];
static int evict(void) {
	int bad = 0;
	SETFRM(0);
	for (int pass = 0; pass < 2; pass++) {
		for (size_t i = 0; i < N; i += LANES) {
			size_t q = i / LANES;
			vsetl(LANES, VT32(0, 0, 0, 0, 0));
			asm volatile("vle32.v v8, (%0)\n\tvle32.v v16, (%1)\n\tvle32.v v24, (%2)"
			             : : "r"(&CAT(T, _a)[i]), "r"(&CAT(T, _b)[i]), "r"(&CAT(T, _c)[i]) : "memory");
			vsetl(32, VT8);
			asm volatile("vle8.v " SC2 ", (%0)\n\tvle8.v v12, (%1)" : : "r"(wrong), "r"(&CAT(T, _sa)[32 * q]) : "memory");
			ld_scales1(0, &CAT(T, _sb)[32 * q]);
			vsetl(LANES, VT32(0, 0, 1, 0, 0)); asm volatile(INSN_VV);       // wrong scales, now buffered
			vsetl(32, VT8); asm volatile("vmv.v.v " SC2 ", v12");           // the right ones
			vsetl(LANES, VT32(0, 0, 0, 0, 0)); asm volatile("vle32.v v24, (%0)" : : "r"(&CAT(T, _c)[i]) : "memory");
			vsetl(LANES, VT32(0, 0, 1, 0, 0)); asm volatile(INSN_VV);
			if (pass) {                                                     // again: a hit
				vsetl(LANES, VT32(0, 0, 0, 0, 0)); asm volatile("vle32.v v24, (%0)" : : "r"(&CAT(T, _c)[i]) : "memory");
				vsetl(LANES, VT32(0, 0, 1, 0, 0)); asm volatile(INSN_VV);
			}
			vsetl(LANES, VT32(0, 0, 0, 0, 0)); asm volatile("vse32.v v24, (%0)" : : "r"(&res[i]) : "memory");
		}
		bad |= check(pass ? "evict (and a hit)" : "evict", CAT(T, _out));
	}
	return bad;
}

int main(void) {
	int failed = 0;
#undef DOT_VV
#undef DOT_VX
#define DOT_VV(t, ...) failed |= run_##t();
#define DOT_VX(t, ...) failed |= run_##t();
#include "tests.h"
	failed |= evict();
	printf(failed ? "FAILED\n" : "passed\n");
	return failed;
}
