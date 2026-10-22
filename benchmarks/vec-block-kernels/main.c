// Scaled mode (docs/scaled-mode.md) against a software baseline, on
// P3109BlockV256D128ShuttleConfig: four kernels, each written with scaled
// instructions and with the best sequence of unscaled ones, timed with mcycle,
// and their outputs compared element by element.
//
//   dequant   MXFP8 (binary8p4, B = 32 along the vector) -> BF16
//   quant     BF16 -> binary8p4, B = 32 along the vector, scales given
//   quantrow  BF16 -> binary8p4 across rows (the OPU's A layout: column k of a
//             32-row matrix per instruction), one scale per row per 32 k (B = 1)
//   gemv      y[32] += W[:, j] * x[j] (axpy form), W binary8p4 with one scale
//             per row per 32 columns (B = 1), x binary8p4, y BF16
//
// The baseline builds each scale 2^(s-128) as a BF16 value (codes 0 and 255
// give 0 and NaN), switches vtype at most once per four blocks, and for
// dequant applies P3109's +0 (a zero result is +0 in BF16, 4.8.2).
#include <stdio.h>
#include <stdint.h>
#include <stddef.h>
#include "util.h"

#define VL 32
#define NBLK 64            // 1D kernels: 64 blocks of 32
#define NE (NBLK * VL)
#define K 256              // quantrow, gemv: 32 rows x K
#define KB (K / VL)

#define VT_E8      0xC0UL  // e8 m1 ta ma, binary8p4
#define VT_E8S(lb) (0x8C0UL | ((unsigned long)(lb) << 12)) // scaled, B = 2^lb
#define VT_BF16    0x1C9UL // e16 m2 ta ma, altfmt: BF16 arithmetic
#define VT_E16     0xC9UL  // e16 m2 ta ma, integer
static inline void vset(unsigned long vt) { asm volatile("vsetvl x0, %0, %1" : : "r"((unsigned long)VL), "r"(vt)); }

#define WCVT_V8_V2   ".insn r 0x57, 0x1, 0x25, x8, x13, x2"   // vfwcvtbf16.f.f.v v8, v2
#define WCVT(d, s)   ".insn r 0x57, 0x1, 0x25, " d ", x13, " s
#define NCVT(d, s)   ".insn r 0x57, 0x1, 0x25, " d ", x29, " s  // vfncvtbf16.f.f.w

static uint8_t  x8[NE] __attribute__((aligned(64)));   // dequant input
static uint8_t  sc[NBLK + VL] __attribute__((aligned(64)));
static uint16_t y16a[NE] __attribute__((aligned(64))), y16b[NE] __attribute__((aligned(64)));
static uint16_t xq[NE] __attribute__((aligned(64)));   // quant input (BF16)
static uint8_t  q8a[NE] __attribute__((aligned(64))), q8b[NE] __attribute__((aligned(64)));
static uint16_t A[VL][K] __attribute__((aligned(64))); // quantrow input, row-major
static uint8_t  srow[VL][KB] __attribute__((aligned(64)));
static uint8_t  Aqa[K][VL] __attribute__((aligned(64))), Aqb[K][VL] __attribute__((aligned(64)));
static uint8_t  W[K][VL] __attribute__((aligned(64)));  // gemv: column j contiguous
static uint8_t  sW[KB][VL] __attribute__((aligned(64))); // row scales per block
static uint8_t  xg[K] __attribute__((aligned(64)));
static uint16_t xg16[K] __attribute__((aligned(64)));   // baseline: x in BF16
static uint16_t yga[VL] __attribute__((aligned(64))), ygb[VL] __attribute__((aligned(64)));
static uint16_t tmp16[VL] __attribute__((aligned(64)));

static uint32_t seed = 12345;
static uint32_t rnd(void) { seed = seed * 1664525u + 1013904223u; return seed >> 8; }
static uint8_t rnd_fp8(void) { uint8_t c; do c = rnd(); while (c == 0x80 || (c & 0x7F) == 0x7F); return c; }
static uint16_t rnd_bf16(int lo, int hi) {
	unsigned e = 127 + lo + rnd() % (hi - lo + 1);
	return ((rnd() & 1) << 15) | (e << 7) | (rnd() & 0x7F);
}
// 2^(s-128) (sign = +1) or 2^(128-s) (sign = -1) as BF16; P3109 Binary8p1uf
static uint16_t bf16_pow2(unsigned s, int sign) {
	if (s == 255) return 0x7FC0;
	if (s == 0) return 0;
	int e = sign > 0 ? (int)s - 128 : 128 - (int)s;
	if (e >= -126) return (uint16_t)((e + 127) << 7);
	return (uint16_t)(0x40 >> (-127 - e)); // subnormal: 2^-127 .. 2^-133
}
static inline void boxf0_16(uint16_t b) { asm volatile("fmv.w.x f0, %0" : : "r"(0xFFFF0000u | b) : "f0"); }

static unsigned long t0;
static void start(void) { t0 = read_csr(mcycle); }
static unsigned long stop(volatile void *last) {
	asm volatile("fence" : : : "memory");
	(void)*(volatile uint8_t *)last; // waits for the kernel's last vector store
	return read_csr(mcycle) - t0;
}

// ---- dequant -----------------------------------------------------------
static void dequant_scaled(void) {
	vset(VT_E8S(5));
	for (int j = 0; j < NBLK; j++) {
		asm volatile("vle8.v v2, (%0)" : : "r"(&x8[j * VL]) : "memory");
		asm volatile("vle8.v v3, (%0)" : : "r"(&sc[j]) : "memory"); // byte 0: block j's scale
		asm volatile(WCVT_V8_V2);
		asm volatile("vse16.v v8, (%0)" : : "r"(&y16a[j * VL]) : "memory");
	}
}
static void dequant_base(void) {
	for (int j = 0; j < NBLK; j += 4) {
		vset(VT_E8);
		asm volatile("vle8.v v2, (%0)" : : "r"(&x8[(j + 0) * VL]) : "memory");
		asm volatile("vle8.v v3, (%0)" : : "r"(&x8[(j + 1) * VL]) : "memory");
		asm volatile("vle8.v v4, (%0)" : : "r"(&x8[(j + 2) * VL]) : "memory");
		asm volatile("vle8.v v5, (%0)" : : "r"(&x8[(j + 3) * VL]) : "memory");
		asm volatile(WCVT("x8", "x2") "\n\t" WCVT("x10", "x3") "\n\t" WCVT("x12", "x4") "\n\t" WCVT("x14", "x5"));
		vset(VT_BF16);
		asm volatile("fmv.w.x f1, %0" : : "r"(0xFFFF0000u) : "f1"); // +0
		boxf0_16(bf16_pow2(sc[j + 0], 1)); asm volatile("vfmul.vf v8, v8, f0\n\tvfadd.vf v8, v8, f1");
		boxf0_16(bf16_pow2(sc[j + 1], 1)); asm volatile("vfmul.vf v10, v10, f0\n\tvfadd.vf v10, v10, f1");
		boxf0_16(bf16_pow2(sc[j + 2], 1)); asm volatile("vfmul.vf v12, v12, f0\n\tvfadd.vf v12, v12, f1");
		boxf0_16(bf16_pow2(sc[j + 3], 1)); asm volatile("vfmul.vf v14, v14, f0\n\tvfadd.vf v14, v14, f1");
		asm volatile("vse16.v v8, (%0)" : : "r"(&y16b[(j + 0) * VL]) : "memory");
		asm volatile("vse16.v v10, (%0)" : : "r"(&y16b[(j + 1) * VL]) : "memory");
		asm volatile("vse16.v v12, (%0)" : : "r"(&y16b[(j + 2) * VL]) : "memory");
		asm volatile("vse16.v v14, (%0)" : : "r"(&y16b[(j + 3) * VL]) : "memory");
	}
}

// ---- quant -------------------------------------------------------------
static void quant_scaled(void) {
	vset(VT_E8S(5));
	for (int j = 0; j < NBLK; j++) {
		asm volatile("vle16.v v16, (%0)" : : "r"(&xq[j * VL]) : "memory");
		asm volatile("vle8.v v9, (%0)" : : "r"(&sc[j]) : "memory");
		asm volatile(NCVT("x8", "x16"));
		asm volatile("vse8.v v8, (%0)" : : "r"(&q8a[j * VL]) : "memory");
	}
}
static void quant_base(void) {
	for (int j = 0; j < NBLK; j += 4) {
		vset(VT_BF16);
		asm volatile("vle16.v v16, (%0)" : : "r"(&xq[(j + 0) * VL]) : "memory");
		asm volatile("vle16.v v18, (%0)" : : "r"(&xq[(j + 1) * VL]) : "memory");
		asm volatile("vle16.v v20, (%0)" : : "r"(&xq[(j + 2) * VL]) : "memory");
		asm volatile("vle16.v v22, (%0)" : : "r"(&xq[(j + 3) * VL]) : "memory");
		boxf0_16(bf16_pow2(sc[j + 0], -1)); asm volatile("vfmul.vf v16, v16, f0");
		boxf0_16(bf16_pow2(sc[j + 1], -1)); asm volatile("vfmul.vf v18, v18, f0");
		boxf0_16(bf16_pow2(sc[j + 2], -1)); asm volatile("vfmul.vf v20, v20, f0");
		boxf0_16(bf16_pow2(sc[j + 3], -1)); asm volatile("vfmul.vf v22, v22, f0");
		vset(VT_E8);
		asm volatile(NCVT("x8", "x16") "\n\t" NCVT("x9", "x18") "\n\t" NCVT("x10", "x20") "\n\t" NCVT("x11", "x22"));
		asm volatile("vse8.v v8, (%0)" : : "r"(&q8b[(j + 0) * VL]) : "memory");
		asm volatile("vse8.v v9, (%0)" : : "r"(&q8b[(j + 1) * VL]) : "memory");
		asm volatile("vse8.v v10, (%0)" : : "r"(&q8b[(j + 2) * VL]) : "memory");
		asm volatile("vse8.v v11, (%0)" : : "r"(&q8b[(j + 3) * VL]) : "memory");
	}
}

// ---- quantrow ----------------------------------------------------------
static void quantrow_scaled(void) {
	vset(VT_E8S(0));
	for (int b = 0; b < KB; b++) {
		asm volatile("vlse8.v v9, (%0), %1" : : "r"(&srow[0][b]), "r"((long)KB) : "memory"); // row scales
		for (int k = b * VL; k < (b + 1) * VL; k++) {
			asm volatile("vlse16.v v16, (%0), %1" : : "r"(&A[0][k]), "r"((long)(2 * K)) : "memory");
			asm volatile(NCVT("x8", "x16"));
			asm volatile("vse8.v v8, (%0)" : : "r"(&Aqa[k][0]) : "memory");
		}
	}
}
// the row scales 2^(128-s) as a BF16 vector in v24 (codes 0, 255: 0, NaN)
static void row_inv_scales(int b) {
	vset(VT_E8);
	asm volatile("vlse8.v v20, (%0), %1" : : "r"(&srow[0][b]), "r"((long)KB) : "memory");
	vset(VT_E16);
	asm volatile("vzext.vf2 v22, v20\n\t"
	             "vrsub.vx v24, v22, %0\n\t"     // 255 - s: the BF16 exponent of 2^(128-s)
	             "vsll.vi v24, v24, 7\n\t"
	             "vmseq.vi v0, v22, 0\n\t"
	             "vmerge.vxm v24, v24, x0, v0\n\t" // s = 0: 0
	             "vmseq.vx v0, v22, %0\n\t"
	             "vmerge.vxm v24, v24, %1, v0"     // s = 255: NaN
	             : : "r"(255L), "r"(0x7FC0L));
}
static void quantrow_base(void) {
	for (int b = 0; b < KB; b++) {
		row_inv_scales(b);
		for (int k = b * VL; k < (b + 1) * VL; k += 4) {
			vset(VT_BF16);
			asm volatile("vlse16.v v16, (%0), %1" : : "r"(&A[0][k + 0]), "r"((long)(2 * K)) : "memory");
			asm volatile("vlse16.v v18, (%0), %1" : : "r"(&A[0][k + 1]), "r"((long)(2 * K)) : "memory");
			asm volatile("vlse16.v v26, (%0), %1" : : "r"(&A[0][k + 2]), "r"((long)(2 * K)) : "memory");
			asm volatile("vlse16.v v28, (%0), %1" : : "r"(&A[0][k + 3]), "r"((long)(2 * K)) : "memory");
			asm volatile("vfmul.vv v16, v16, v24\n\tvfmul.vv v18, v18, v24\n\t"
			             "vfmul.vv v26, v26, v24\n\tvfmul.vv v28, v28, v24");
			vset(VT_E8);
			asm volatile(NCVT("x8", "x16") "\n\t" NCVT("x9", "x18") "\n\t" NCVT("x10", "x26") "\n\t" NCVT("x11", "x28"));
			asm volatile("vse8.v v8, (%0)" : : "r"(&Aqb[k + 0][0]) : "memory");
			asm volatile("vse8.v v9, (%0)" : : "r"(&Aqb[k + 1][0]) : "memory");
			asm volatile("vse8.v v10, (%0)" : : "r"(&Aqb[k + 2][0]) : "memory");
			asm volatile("vse8.v v11, (%0)" : : "r"(&Aqb[k + 3][0]) : "memory");
		}
	}
}

// ---- gemv --------------------------------------------------------------
static void gemv_scaled(void) {
	vset(VT_BF16);
	asm volatile("vmv.v.i v8, 0");
	vset(VT_E8S(0));
	for (int b = 0; b < KB; b++) {
		asm volatile("vle8.v v3, (%0)" : : "r"(&sW[b][0]) : "memory"); // row scales, B = 1
		for (int j = b * VL; j < (b + 1) * VL; j++) {
			asm volatile("vle8.v v2, (%0)" : : "r"(&W[j][0]) : "memory");
			asm volatile("fmv.w.x f0, %0\n\tvfwmacc.vf v8, f0, v2" : : "r"(0xFFFFFF00u | xg[j]) : "f0");
		}
	}
	vset(VT_BF16);
	asm volatile("vse16.v v8, (%0)" : : "r"(yga) : "memory");
}
// GEMV variants for timing: the same loop unscaled (vt = VT_E8), and two data
// registers alternating so a column loads while the previous one computes;
// under rule A the second (v4) needs its own copy of the scales (v5).
static unsigned long gvt;
static void gemv_one(void) {
	vset(VT_BF16);
	asm volatile("vmv.v.i v8, 0");
	vset(gvt);
	for (int b = 0; b < KB; b++) {
		asm volatile("vle8.v v3, (%0)" : : "r"(&sW[b][0]) : "memory");
		for (int j = b * VL; j < (b + 1) * VL; j++) {
			asm volatile("vle8.v v2, (%0)" : : "r"(&W[j][0]) : "memory");
			asm volatile("fmv.w.x f0, %0\n\tvfwmacc.vf v8, f0, v2" : : "r"(0xFFFFFF00u | xg[j]) : "f0");
		}
	}
	vset(VT_BF16);
	asm volatile("vse16.v v8, (%0)" : : "r"(yga) : "memory");
}
static void gemv_two(void) {
	vset(VT_BF16);
	asm volatile("vmv.v.i v8, 0");
	vset(gvt);
	for (int b = 0; b < KB; b++) {
		asm volatile("vle8.v v3, (%0)" : : "r"(&sW[b][0]) : "memory");
		asm volatile("vmv.v.v v5, v3"); // rule A: v4's scales
		for (int j = b * VL; j < (b + 1) * VL; j += 2) {
			asm volatile("vle8.v v2, (%0)" : : "r"(&W[j][0]) : "memory");
			asm volatile("vle8.v v4, (%0)" : : "r"(&W[j + 1][0]) : "memory");
			asm volatile("fmv.w.x f0, %0\n\tvfwmacc.vf v8, f0, v2" : : "r"(0xFFFFFF00u | xg[j]) : "f0");
			asm volatile("fmv.w.x f0, %0\n\tvfwmacc.vf v8, f0, v4" : : "r"(0xFFFFFF00u | xg[j + 1]) : "f0");
		}
	}
	vset(VT_BF16);
	asm volatile("vse16.v v8, (%0)" : : "r"(yga) : "memory");
}
// Unrolled by 4: four accumulators (v8, v10, v12, v14, summed at the end) break
// the dependency chain, four data registers overlap the loads. Under rule A
// each data register (v2, v4, v6, v16) needs its own copy of the block's
// scales (v3, v5, v7, v17).
static void row_scales(int b);
static uint16_t yg4a[VL] __attribute__((aligned(64))), yg4b[VL] __attribute__((aligned(64)));
#define SUM4(out) \
	vset(VT_BF16); \
	asm volatile("vfadd.vv v8, v8, v10\n\tvfadd.vv v12, v12, v14\n\tvfadd.vv v8, v8, v12"); \
	asm volatile("vse16.v v8, (%0)" : : "r"(out) : "memory");
static void gemv4_scaled(void) {
	vset(VT_BF16);
	asm volatile("vmv.v.i v8, 0\n\tvmv.v.i v10, 0\n\tvmv.v.i v12, 0\n\tvmv.v.i v14, 0");
	vset(gvt);
	for (int b = 0; b < KB; b++) {
		asm volatile("vle8.v v3, (%0)" : : "r"(&sW[b][0]) : "memory");
		asm volatile("vmv.v.v v5, v3\n\tvmv.v.v v7, v3\n\tvmv.v.v v17, v3"); // rule A copies
		for (int j = b * VL; j < (b + 1) * VL; j += 4) {
			asm volatile("vle8.v v2, (%0)" : : "r"(&W[j][0]) : "memory");
			asm volatile("vle8.v v4, (%0)" : : "r"(&W[j + 1][0]) : "memory");
			asm volatile("vle8.v v6, (%0)" : : "r"(&W[j + 2][0]) : "memory");
			asm volatile("vle8.v v16, (%0)" : : "r"(&W[j + 3][0]) : "memory");
			asm volatile("fmv.w.x f0, %0\n\tvfwmacc.vf v8, f0, v2" : : "r"(0xFFFFFF00u | xg[j]) : "f0");
			asm volatile("fmv.w.x f0, %0\n\tvfwmacc.vf v10, f0, v4" : : "r"(0xFFFFFF00u | xg[j + 1]) : "f0");
			asm volatile("fmv.w.x f0, %0\n\tvfwmacc.vf v12, f0, v6" : : "r"(0xFFFFFF00u | xg[j + 2]) : "f0");
			asm volatile("fmv.w.x f0, %0\n\tvfwmacc.vf v14, f0, v16" : : "r"(0xFFFFFF00u | xg[j + 3]) : "f0");
		}
	}
	SUM4(yg4a)
}
static void gemv4_base(void) {
	vset(VT_BF16);
	asm volatile("vmv.v.i v8, 0\n\tvmv.v.i v10, 0\n\tvmv.v.i v12, 0\n\tvmv.v.i v14, 0");
	for (int b = 0; b < KB; b++) {
		row_scales(b);
		for (int j = b * VL; j < (b + 1) * VL; j += 4) {
			vset(VT_E8);
			asm volatile("vle8.v v2, (%0)" : : "r"(&W[j][0]) : "memory");
			asm volatile("vle8.v v3, (%0)" : : "r"(&W[j + 1][0]) : "memory");
			asm volatile("vle8.v v4, (%0)" : : "r"(&W[j + 2][0]) : "memory");
			asm volatile("vle8.v v5, (%0)" : : "r"(&W[j + 3][0]) : "memory");
			asm volatile(WCVT("x16", "x2") "\n\t" WCVT("x18", "x3") "\n\t" WCVT("x20", "x4") "\n\t" WCVT("x22", "x5"));
			vset(VT_BF16);
			asm volatile("vfmul.vv v16, v16, v24\n\tvfmul.vv v18, v18, v24\n\tvfmul.vv v20, v20, v24\n\tvfmul.vv v22, v22, v24");
			asm volatile("fmv.w.x f0, %0\n\tvfmacc.vf v8, f0, v16" : : "r"(0xFFFF0000u | xg16[j]) : "f0");
			asm volatile("fmv.w.x f0, %0\n\tvfmacc.vf v10, f0, v18" : : "r"(0xFFFF0000u | xg16[j + 1]) : "f0");
			asm volatile("fmv.w.x f0, %0\n\tvfmacc.vf v12, f0, v20" : : "r"(0xFFFF0000u | xg16[j + 2]) : "f0");
			asm volatile("fmv.w.x f0, %0\n\tvfmacc.vf v14, f0, v22" : : "r"(0xFFFF0000u | xg16[j + 3]) : "f0");
		}
	}
	SUM4(yg4b)
}

static void time_gemv(const char *name, void (*f)(void), unsigned long vt) {
	gvt = vt;
	f();
	start(); f(); unsigned long c = stop(&yga[VL - 1]);
	printf("gemv %-24s %6lu cycles  %lu.%02lu cycles/element\n", name, c, c / (K * VL), c * 100 / (K * VL) % 100);
}

// the row scales 2^(s-128) as a BF16 vector in v24
static void row_scales(int b) {
	vset(VT_E8);
	asm volatile("vle8.v v20, (%0)" : : "r"(&sW[b][0]) : "memory");
	vset(VT_E16);
	asm volatile("vzext.vf2 v22, v20\n\t"
	             "vadd.vi v24, v22, -1\n\t"        // s - 1: the BF16 exponent of 2^(s-128)
	             "vsll.vi v24, v24, 7\n\t"
	             "vmseq.vi v0, v22, 0\n\t"
	             "vmerge.vxm v24, v24, x0, v0\n\t"
	             "vmseq.vi v0, v22, 1\n\t"
	             "vmerge.vxm v24, v24, %2, v0\n\t" // s = 1: 2^-127, subnormal
	             "vmseq.vx v0, v22, %0\n\t"
	             "vmerge.vxm v24, v24, %1, v0"
	             : : "r"(255L), "r"(0x7FC0L), "r"(0x40L));
}
static void gemv_base(void) {
	vset(VT_BF16);
	asm volatile("vmv.v.i v8, 0");
	for (int b = 0; b < KB; b++) {
		row_scales(b);
		for (int j = b * VL; j < (b + 1) * VL; j += 2) {
			vset(VT_E8);
			asm volatile("vle8.v v2, (%0)" : : "r"(&W[j][0]) : "memory");
			asm volatile("vle8.v v3, (%0)" : : "r"(&W[j + 1][0]) : "memory");
			asm volatile(WCVT("x16", "x2") "\n\t" WCVT("x18", "x3"));
			vset(VT_BF16);
			asm volatile("vfmul.vv v16, v16, v24\n\tvfmul.vv v18, v18, v24");
			asm volatile("fmv.w.x f0, %0\n\tvfmacc.vf v8, f0, v16" : : "r"(0xFFFF0000u | xg16[j]) : "f0");
			asm volatile("fmv.w.x f0, %0\n\tvfmacc.vf v8, f0, v18" : : "r"(0xFFFF0000u | xg16[j + 1]) : "f0");
		}
	}
	asm volatile("vse16.v v8, (%0)" : : "r"(ygb) : "memory");
}

static int differ(const void *a, const void *b, int n, int esize) {
	int d = 0;
	for (int i = 0; i < n; i++)
		d += esize == 1 ? ((const uint8_t *)a)[i] != ((const uint8_t *)b)[i]
		                : ((const uint16_t *)a)[i] != ((const uint16_t *)b)[i];
	return d;
}
static void report(const char *name, void (*fs)(void), void (*fb)(void), volatile void *ls, volatile void *lb,
                   const void *oa, const void *ob, int n, int esize, int elems) {
	fs(); fb(); // warm up
	start(); fs(); unsigned long cs = stop(ls);
	start(); fb(); unsigned long cb = stop(lb);
	printf("%-9s scaled %6lu  baseline %6lu cycles  (%lu.%02lu x)  %lu.%02lu vs %lu.%02lu cycles/element  outputs differ: %d of %d\n",
	       name, cs, cb, cb / cs, cb * 100 / cs % 100,
	       cs / elems, cs * 100 / elems % 100, cb / elems, cb * 100 / elems % 100, differ(oa, ob, n, esize), n);
}

int main(void) {
	for (int i = 0; i < NE; i++) { x8[i] = rnd_fp8(); xq[i] = rnd_bf16(-4, 12); }
	for (int j = 0; j < NBLK + VL; j++) sc[j] = 128 + rnd() % 21 - 10;
	for (int r = 0; r < VL; r++) {
		for (int k = 0; k < K; k++) A[r][k] = rnd_bf16(-4, 12);
		for (int b = 0; b < KB; b++) srow[r][b] = 128 + rnd() % 21 - 10;
	}
	for (int j = 0; j < K; j++) { xg[j] = rnd_fp8(); for (int r = 0; r < VL; r++) W[j][r] = rnd_fp8(); }
	for (int b = 0; b < KB; b++) for (int r = 0; r < VL; r++) sW[b][r] = 128 + rnd() % 9 - 4;
	// x in BF16 for the baseline gemv (exact: every binary8p4 value is a BF16 value)
	for (int j = 0; j < K; j += VL) {
		vset(VT_E8);
		asm volatile("vle8.v v2, (%0)" : : "r"(&xg[j]) : "memory");
		asm volatile(WCVT("x16", "x2"));
		vset(VT_E16);
		asm volatile("vse16.v v16, (%0)" : : "r"(&xg16[j]) : "memory");
	}

	report("dequant", dequant_scaled, dequant_base, &y16a[NE - 1], &y16b[NE - 1], y16a, y16b, NE, 2, NE);
	report("quant", quant_scaled, quant_base, &q8a[NE - 1], &q8b[NE - 1], q8a, q8b, NE, 1, NE);
	report("quantrow", quantrow_scaled, quantrow_base, &Aqa[K - 1][VL - 1], &Aqb[K - 1][VL - 1], Aqa, Aqb, K * VL, 1, K * VL);
	report("gemv", gemv_scaled, gemv_base, &yga[VL - 1], &ygb[VL - 1], yga, ygb, VL, 2, K * VL);
	time_gemv("one register, unscaled", gemv_one, VT_E8);
	time_gemv("one register, scaled", gemv_one, VT_E8S(0));
	time_gemv("two registers, unscaled", gemv_two, VT_E8);
	time_gemv("two registers, scaled", gemv_two, VT_E8S(0));
	gvt = VT_E8S(0);
	report("gemv x4", gemv4_scaled, gemv4_base, &yg4a[VL - 1], &yg4b[VL - 1], yg4a, yg4b, VL, 2, K * VL);
	gvt = VT_E8;
	gemv4_scaled(); start(); gemv4_scaled(); { unsigned long c = stop(&yg4a[VL - 1]);
	printf("gemv x4 unscaled (timing only) %6lu cycles  %lu.%02lu cycles/element\n", c, c / (K * VL), c * 100 / (K * VL) % 100); }
	// the scaled two-register result must still be right
	{ uint16_t keep[VL]; for (int i = 0; i < VL; i++) keep[i] = ygb[i];
	  printf("gemv two registers, scaled: outputs differ from the baseline: %d of %d\n", differ(yga, keep, VL, 2), VL); }
	return 0;
}
