// Scaled mode (docs/scaled-mode.md), on a P3109 build with block scales
// (P3109BlockV256D128ShuttleConfig). Each test sets vtype's scaled bit (bit
// 11) and block size B (bits 14:12, log2), through vsetvl (vsetvli's
// immediate stops at bit 10), and runs one instruction in which every 8-bit
// register operand vN is a block: element i takes Binary8p1uf scale byte
// i >> log2(B) of vN+1. Results are checked bit for bit against gen_data.py's
// P3109 reference, which writes data.S (the arrays) and tests.h (one line per
// test: kind, name, frm, altfmt, log2 B, instruction).
//
// Registers, by kind (see gen_data.py):
//   K8     vs2 v2 (scales v3), vs1 v4 (v5), vd v8 (v9): an 8-bit result
//   KW     vs2 v2 (v3), vs1 v4 (v5), vd v8-v9: a BF16 result (and accumulator)
//   KWW    vs2 v16-v17 (BF16), vs1 v4 (v5), vd v8-v9 (BF16)
//   KWCVT  vs2 v2 (v3), vd v8-v9 (BF16): ConvertFromBlock
//   KNCVT  vs2 v16-v17 (BF16), vd v8 (v9): ConvertToBlock
// The .vf scalar is in f0, NaN-boxed; it is not a block.
#include <stdio.h>
#include <stdint.h>
#include <stddef.h>

#define N 128 // gen_data.py's COUNT
#define VL 32 // elements per strip: VLEN 256, SEW 8, LMUL 1

static uint8_t r8[N] __attribute__((aligned(64)));
static uint16_t r16[N] __attribute__((aligned(64)));

// vtype: e8 m1, ta ma, altfmt (bit 8); scaled (bit 11) with log2 of the block
// size (bits 14:12); and e16 m2 for BF16
#define VT8(alt, sc) (0xC0UL | ((unsigned long)(alt) << 8) | ((unsigned long)(sc) << 11))
#define VT8S(alt, lb) (VT8(alt, 1) | ((unsigned long)(lb) << 12))
#define VT16 0xC9UL
static inline void vset(unsigned long vt) {
	asm volatile("vsetvl x0, %0, %1" : : "r"((unsigned long)VL), "r"(vt));
}
#define LD8(v, p)  asm volatile("vle8.v "  v ", (%0)" : : "r"(p) : "memory")
#define LD16(v, p) asm volatile("vle16.v " v ", (%0)" : : "r"(p) : "memory")
#define ST8(v, p)  asm volatile("vse8.v "  v ", (%0)" : : "r"(p) : "memory")
#define ST16(v, p) asm volatile("vse16.v " v ", (%0)" : : "r"(p) : "memory")
#define SETFRM(rm) asm volatile("csrw frm, %0" : : "r"((unsigned long)(rm)))
// the instruction, with the .vf scalar moved into f0 in the same asm statement
#define RUN_OP(op, f) asm volatile("fmv.w.x f0, %0\n\t" op : : "r"(0xFFFFFF00u | (f)) : "f0", "memory")

static int check(const char *name, const void *got, const void *want, int esize) {
	int bad = 0;
	for (int i = 0; i < N; i++) {
		unsigned g = esize == 1 ? ((const uint8_t *)got)[i] : ((const uint16_t *)got)[i];
		unsigned w = esize == 1 ? ((const uint8_t *)want)[i] : ((const uint16_t *)want)[i];
		if (g != w && bad++ < 4)
			printf("MISMATCH %s [%d]: got %#x, want %#x\n", name, i, g, w);
	}
	printf("%s: %d of %d differ\n", name, bad, N);
	return bad != 0;
}

#define K8(t, rm, alt, lb, op) \
	extern uint8_t t##_a[], t##_sa[], t##_b[], t##_sb[], t##_c[], t##_sr[], t##_f[], t##_out[]; \
	static int run_##t(void) { \
		SETFRM(rm); \
		for (size_t i = 0; i < N; i += VL) { \
			vset(VT8(alt, 0)); \
			LD8("v2", &t##_a[i]); LD8("v3", &t##_sa[i]); LD8("v4", &t##_b[i]); LD8("v5", &t##_sb[i]); \
			LD8("v8", &t##_c[i]); LD8("v9", &t##_sr[i]); \
			vset(VT8S(alt, lb)); RUN_OP(op, t##_f[i / VL]); vset(VT8(alt, 0)); \
			ST8("v8", &r8[i]); \
		} \
		return check(#t, r8, t##_out, 1); \
	}
#define KW(t, rm, alt, lb, op) \
	extern uint8_t t##_a[], t##_sa[], t##_b[], t##_sb[], t##_f[]; extern uint16_t t##_c[], t##_out[]; \
	static int run_##t(void) { \
		SETFRM(rm); \
		for (size_t i = 0; i < N; i += VL) { \
			vset(VT8(alt, 0)); \
			LD8("v2", &t##_a[i]); LD8("v3", &t##_sa[i]); LD8("v4", &t##_b[i]); LD8("v5", &t##_sb[i]); \
			vset(VT16); LD16("v8", &t##_c[i]); \
			vset(VT8S(alt, lb)); RUN_OP(op, t##_f[i / VL]); \
			vset(VT16); ST16("v8", &r16[i]); \
		} \
		return check(#t, r16, t##_out, 2); \
	}
#define KWW(t, rm, alt, lb, op) \
	extern uint8_t t##_b[], t##_sb[]; extern uint16_t t##_a[], t##_out[]; \
	static int run_##t(void) { \
		SETFRM(rm); \
		for (size_t i = 0; i < N; i += VL) { \
			vset(VT16); LD16("v16", &t##_a[i]); \
			vset(VT8(alt, 0)); LD8("v4", &t##_b[i]); LD8("v5", &t##_sb[i]); \
			vset(VT8S(alt, lb)); RUN_OP(op, 0); \
			vset(VT16); ST16("v8", &r16[i]); \
		} \
		return check(#t, r16, t##_out, 2); \
	}
#define KWCVT(t, rm, alt, lb, op) \
	extern uint8_t t##_a[], t##_sa[]; extern uint16_t t##_out[]; \
	static int run_##t(void) { \
		SETFRM(rm); \
		for (size_t i = 0; i < N; i += VL) { \
			vset(VT8(alt, 0)); LD8("v2", &t##_a[i]); LD8("v3", &t##_sa[i]); \
			vset(VT8S(alt, lb)); RUN_OP(op, 0); \
			vset(VT16); ST16("v8", &r16[i]); \
		} \
		return check(#t, r16, t##_out, 2); \
	}
#define KNCVT(t, rm, alt, lb, op) \
	extern uint8_t t##_sr[], t##_out[]; extern uint16_t t##_a[]; \
	static int run_##t(void) { \
		SETFRM(rm); \
		for (size_t i = 0; i < N; i += VL) { \
			vset(VT16); LD16("v16", &t##_a[i]); \
			vset(VT8(alt, 0)); LD8("v9", &t##_sr[i]); \
			vset(VT8S(alt, lb)); RUN_OP(op, 0); vset(VT8(alt, 0)); \
			ST8("v8", &r8[i]); \
		} \
		return check(#t, r8, t##_out, 1); \
	}
#include "tests.h"

// Scales kept across instructions (step 3): a buffered scale group must be
// dropped when its register is written, by any unit. Fill the buffer with
// wrong scales (code 0), overwrite v3 with the right ones by `copy` (vmv.v.v,
// or vfsgnj.vv at SEW 16), and expect the right result; then repeat with no
// write in between (a buffer hit). Uses the vfwmul.vv B = 1 test's arrays.
#define T wmul_vv_a0_rne_b1
#define CAT_(a, b) a##b
#define CAT(a, b) CAT_(a, b)
static uint8_t wrong[VL] __attribute__((aligned(64)));
#define EVICT(fn, setcopy, copy) \
	static int fn(void) { \
		int bad = 0; \
		SETFRM(0); \
		for (int pass = 0; pass < 2; pass++) { \
			for (size_t i = 0; i < N; i += VL) { \
				vset(VT8(0, 0)); \
				LD8("v2", &CAT(T, _a)[i]); LD8("v6", &CAT(T, _sa)[i]); LD8("v3", wrong); \
				LD8("v4", &CAT(T, _b)[i]); LD8("v5", &CAT(T, _sb)[i]); \
				vset(VT8S(0, 0)); asm volatile("vfwmul.vv v8, v2, v4"); \
				setcopy; asm volatile(copy); \
				vset(VT8S(0, 0)); asm volatile("vfwmul.vv v8, v2, v4"); \
				if (pass) asm volatile("vfwmul.vv v8, v2, v4"); /* a hit */ \
				vset(VT16); ST16("v8", &r16[i]); \
			} \
			bad |= check(pass ? #fn " (and a hit)" : #fn, r16, CAT(T, _out), 2); \
		} \
		return bad; \
	}
EVICT(evict_vmv, vset(VT8(0, 0)), "vmv.v.v v3, v6")
EVICT(evict_vfsgnj, asm volatile("vsetvli x0, %0, e16, m1, ta, ma" : : "r"(16UL)), "vfsgnj.vv v3, v6, v6")
#undef T

int main(void) {
	int failed = 0;
	failed |= evict_vmv();
	failed |= evict_vfsgnj();
#undef K8
#undef KW
#undef KWW
#undef KWCVT
#undef KNCVT
#define K8(t, rm, alt, lb, op) failed |= run_##t();
#define KW(t, rm, alt, lb, op) failed |= run_##t();
#define KWW(t, rm, alt, lb, op) failed |= run_##t();
#define KWCVT(t, rm, alt, lb, op) failed |= run_##t();
#define KNCVT(t, rm, alt, lb, op) failed |= run_##t();
#include "tests.h"
	printf(failed ? "FAILED\n" : "passed\n");
	return failed;
}
