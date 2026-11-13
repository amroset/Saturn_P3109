// Scaled mode (docs/scaled-mode.md) beyond vec-block: register groups (LMUL 2,
// 4, 8, scales in the group after the data, E < L reaching the later scale
// registers), masked instructions (mask undisturbed) and tails (vl < VLMAX,
// tail undisturbed), checked bit for bit against gen_data.py's reference,
// inactive and tail elements against vd's old value. Last, a scaled
// instruction with vstart != 0, which Saturn rejects as illegal.
// Each test line in tests.h names its kind, LMUL, vl, whether it is masked,
// its registers (data, then scales; "" if none) and the instruction.
#include <stdio.h>
#include <stdint.h>
#include <stddef.h>

#define N 256 // gen_data.py's COUNT

static uint8_t r8[N] __attribute__((aligned(64)));
static uint16_t r16[N] __attribute__((aligned(64)));

static volatile uintptr_t trap_cause;
uintptr_t handle_trap(uintptr_t cause, uintptr_t epc, uintptr_t regs[32]) {
	trap_cause = cause;
	asm volatile("csrw vstart, x0");
	return epc + 4;
}

static const unsigned long LMC[9] = {0, 0, 1, 0, 2, 0, 0, 0, 3};
// vtype: SEW 8 << sew, LMUL lm; ta ma unless undisturbed (tu mu); altfmt
// (bit 8); scaled (bit 11) with log2 E (bits 14:12)
#define VT(sew, lm, alt, sc, lb, und) (((und) ? 0UL : 0xC0UL) | ((unsigned long)(sew) << 3) | LMC[lm] | \
	((unsigned long)(alt) << 8) | ((unsigned long)(sc) << 11) | ((unsigned long)(lb) << 12))
static inline void vset(unsigned long vl, unsigned long vt) {
	asm volatile("vsetvl x0, %0, %1" : : "r"(vl), "r"(vt));
}
#define LD8(v, p)  asm volatile("vle8.v "  v ", (%0)" : : "r"(p) : "memory")
#define LD16(v, p) asm volatile("vle16.v " v ", (%0)" : : "r"(p) : "memory")
#define ST8(v, p)  asm volatile("vse8.v "  v ", (%0)" : : "r"(p) : "memory")
#define ST16(v, p) asm volatile("vse16.v " v ", (%0)" : : "r"(p) : "memory")
#define SETFRM(rm) asm volatile("csrw frm, %0" : : "r"((unsigned long)(rm)))
#define RUN_OP(op, f) asm volatile("fmv.w.x f0, %0\n\t" op : : "r"(0xFFFFFF00u | (f)) : "f0", "memory")
#define MASK(msk, m, q) do { if (msk) { vset(32, VT(0, 1, 0, 0, 0, 0)); LD8("v0", &m[(q) * 32]); } } while (0)

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

// 8-bit result; vs2, vs1 and vd blocks
#define K8(t, rm, alt, lb, lm, vl, msk, A, AS, B, BS, D, DS, op) \
	extern uint8_t t##_a[], t##_sa[], t##_b[], t##_sb[], t##_c[], t##_sr[], t##_f[], t##_m[], t##_out[]; \
	static int run_##t(void) { \
		SETFRM(rm); \
		for (size_t i = 0; i < N; i += 32 * lm) { \
			vset(32 * lm, VT(0, lm, alt, 0, 0, 0)); \
			LD8(A, &t##_a[i]); LD8(AS, &t##_sa[i]); LD8(B, &t##_b[i]); LD8(BS, &t##_sb[i]); \
			LD8(D, &t##_c[i]); LD8(DS, &t##_sr[i]); \
			MASK(msk, t##_m, i / (32 * lm)); \
			vset(vl, VT(0, lm, alt, 1, lb, msk)); RUN_OP(op, t##_f[i / (32 * lm)]); \
			vset(32 * lm, VT(0, lm, alt, 0, 0, 0)); ST8(D, &r8[i]); \
		} \
		return check(#t, r8, t##_out, 1); \
	}
// .vf: vs2 and vd blocks
#define K8F(t, rm, alt, lb, lm, vl, msk, A, AS, B, BS, D, DS, op) \
	extern uint8_t t##_a[], t##_sa[], t##_c[], t##_sr[], t##_f[], t##_m[], t##_out[]; \
	static int run_##t(void) { \
		SETFRM(rm); \
		for (size_t i = 0; i < N; i += 32 * lm) { \
			vset(32 * lm, VT(0, lm, alt, 0, 0, 0)); \
			LD8(A, &t##_a[i]); LD8(AS, &t##_sa[i]); LD8(D, &t##_c[i]); LD8(DS, &t##_sr[i]); \
			MASK(msk, t##_m, i / (32 * lm)); \
			vset(vl, VT(0, lm, alt, 1, lb, msk)); RUN_OP(op, t##_f[i / (32 * lm)]); \
			vset(32 * lm, VT(0, lm, alt, 0, 0, 0)); ST8(D, &r8[i]); \
		} \
		return check(#t, r8, t##_out, 1); \
	}
// BF16 result (and accumulator) in a group of 2 lm; vs2, vs1 blocks
#define KW(t, rm, alt, lb, lm, vl, msk, A, AS, B, BS, D, DS, op) \
	extern uint8_t t##_a[], t##_sa[], t##_b[], t##_sb[], t##_f[], t##_m[]; extern uint16_t t##_c[], t##_out[]; \
	static int run_##t(void) { \
		SETFRM(rm); \
		for (size_t i = 0; i < N; i += 32 * lm) { \
			vset(32 * lm, VT(0, lm, alt, 0, 0, 0)); \
			LD8(A, &t##_a[i]); LD8(AS, &t##_sa[i]); LD8(B, &t##_b[i]); LD8(BS, &t##_sb[i]); \
			vset(32 * lm, VT(1, 2 * lm, 0, 0, 0, 0)); LD16(D, &t##_c[i]); \
			MASK(msk, t##_m, i / (32 * lm)); \
			vset(vl, VT(0, lm, alt, 1, lb, msk)); RUN_OP(op, t##_f[i / (32 * lm)]); \
			vset(32 * lm, VT(1, 2 * lm, 0, 0, 0, 0)); ST16(D, &r16[i]); \
		} \
		return check(#t, r16, t##_out, 2); \
	}
// ConvertFromBlock: vs2 block to BF16
#define KWCVT(t, rm, alt, lb, lm, vl, msk, A, AS, B, BS, D, DS, op) \
	extern uint8_t t##_a[], t##_sa[], t##_m[]; extern uint16_t t##_c[], t##_out[]; \
	static int run_##t(void) { \
		SETFRM(rm); \
		for (size_t i = 0; i < N; i += 32 * lm) { \
			vset(32 * lm, VT(0, lm, alt, 0, 0, 0)); LD8(A, &t##_a[i]); LD8(AS, &t##_sa[i]); \
			vset(32 * lm, VT(1, 2 * lm, 0, 0, 0, 0)); LD16(D, &t##_c[i]); \
			MASK(msk, t##_m, i / (32 * lm)); \
			vset(vl, VT(0, lm, alt, 1, lb, msk)); RUN_OP(op, 0); \
			vset(32 * lm, VT(1, 2 * lm, 0, 0, 0, 0)); ST16(D, &r16[i]); \
		} \
		return check(#t, r16, t##_out, 2); \
	}
// ConvertToBlock: BF16 in a group of 2 lm to a vd block
#define KNCVT(t, rm, alt, lb, lm, vl, msk, A, AS, B, BS, D, DS, op) \
	extern uint8_t t##_c[], t##_sr[], t##_m[], t##_out[]; extern uint16_t t##_a[]; \
	static int run_##t(void) { \
		SETFRM(rm); \
		for (size_t i = 0; i < N; i += 32 * lm) { \
			vset(32 * lm, VT(1, 2 * lm, 0, 0, 0, 0)); LD16(A, &t##_a[i]); \
			vset(32 * lm, VT(0, lm, alt, 0, 0, 0)); LD8(D, &t##_c[i]); LD8(DS, &t##_sr[i]); \
			MASK(msk, t##_m, i / (32 * lm)); \
			vset(vl, VT(0, lm, alt, 1, lb, msk)); RUN_OP(op, 0); \
			vset(32 * lm, VT(0, lm, alt, 0, 0, 0)); ST8(D, &r8[i]); \
		} \
		return check(#t, r8, t##_out, 1); \
	}
#include "tests.h"

// vstart != 0: Saturn raises an illegal-instruction exception for any vector
// instruction outside the memory pipeline (PipelinedFaultCheck), scaled or
// not; vd must be unchanged. Uses the masked vfadd test's arrays.
#define T add_vv_m1_b1_a0_rne_mask
#define CAT_(a, b) a##b
#define CAT(a, b) CAT_(a, b)
static int vstart_case(int sc) {
	vset(32, VT(0, 1, 0, 0, 0, 0));
	LD8("v2", CAT(T, _a)); LD8("v3", CAT(T, _sa)); LD8("v4", CAT(T, _b)); LD8("v5", CAT(T, _sb));
	LD8("v8", CAT(T, _c)); LD8("v9", CAT(T, _sr));
	vset(32, VT(0, 1, 0, sc, 0, 1));
	trap_cause = 0;
	asm volatile("csrw vstart, %0\n\tvfadd.vv v8, v2, v4" : : "r"(5UL) : "memory");
	unsigned long vs;
	asm volatile("csrr %0, vstart" : "=r"(vs));
	vset(32, VT(0, 1, 0, 0, 0, 0));
	ST8("v8", r8);
	int bad = 0;
	for (int i = 0; i < 32; i++) bad += r8[i] != CAT(T, _c)[i];
	int ok = trap_cause == 2 && bad == 0 && vs == 0;
	printf("%s vstart 5, %s vfadd.vv: cause %lu, %d of 32 elements of vd changed\n",
	       ok ? "ok  " : "FAIL", sc ? "scaled" : "unscaled", (unsigned long)trap_cause, bad);
	return !ok;
}

int main(void) {
	int failed = 0, n = 0;
#undef K8
#undef K8F
#undef KW
#undef KWCVT
#undef KNCVT
#define K8(t, ...) failed |= run_##t(); n++;
#define K8F(t, ...) failed |= run_##t(); n++;
#define KW(t, ...) failed |= run_##t(); n++;
#define KWCVT(t, ...) failed |= run_##t(); n++;
#define KNCVT(t, ...) failed |= run_##t(); n++;
#include "tests.h"
	failed |= vstart_case(1);
	failed |= vstart_case(0);
	printf("%d tests + 2 vstart cases\n", n);
	printf(failed ? "FAILED\n" : "passed\n");
	return failed;
}
