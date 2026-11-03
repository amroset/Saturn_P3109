// Explicit scale operands against rule A (docs/scaled-mode.md, notes:
// vector-scaled-mode.md). Kernels where the encoding matters, each timed in
// the variants a build supports, its outputs summarized by a checksum so runs
// on different builds can be compared:
//
//   rule A    scales in vN+1; an unrolled loop keeps a copy of the scales next
//             to every data register it uses (P3109BlockV256D128ShuttleConfig,
//             and P3109BlockBuf4..., whose buffer holds 4 registers per operand)
//   shared    -DSHARED, on P3109BlockSharedV256D128ShuttleConfig: operands take
//             their scales from v31 (vs1), v30 (vs2), v29 (vd), as kernels
//             with explicit scale operands would, one shared register each
//   unscaled  the same loop without scales: the bound for any scale scheme
//   baseline  the best sequence of unscaled instructions (no scaled mode)
//   scalar    for one or two scales per instruction (E = 32, 16): the unscaled loop plus a
//             scalar load of each scale, an estimate of an explicit scalar
//             scale operand (the scale would arrive with the instruction)
//
// Kernels (VLEN 256, 32 elements per register, unrolled by 4 where marked):
//   quantrow  BF16 -> binary8p4 across rows, x4: 32 rows, one scale per row
//             per 32 k (E = 1), the OPU's A-operand quantizer, unit stride
//   addrow    FP8 block add across rows, x4: y = x + r, three blocks with
//             per-row scales per 32 k (E = 1)
//   gemv      y[32] += W[:, j] x[j], x4, W with per-row scales per 32 columns
//   dequant   binary8p4 -> BF16 along the vector, one scale per register (E = 32)
//   quant     BF16 -> binary8p4 along the vector, one scale per register (E = 32)
//   dequantrow binary8p4 -> BF16 across rows, x4 (E = 1)
//   smulrow   P3109 ScaledMultiply across rows, x4: z = x * w, two blocks,
//             binary8p4 -> BF16 (E = 1)
//   smul      the same along the vector, one scale per register and operand (E = 32)
//   dequant16, quant16  as dequant, quant with two blocks of 16 per register (E = 16)
#include <stdio.h>
#include <stdint.h>
#include <stddef.h>
#include "util.h"

// Layout sweep: -DPAD=n shifts the code after this point by n bytes (with
// -fno-toplevel-reorder); -DREPS=n times n back-to-back calls after the warm-up.
#ifndef PAD
#define PAD 0
#endif
#ifndef REPS
#define REPS 1
#endif
#define PSTR_(x) #x
#define PSTR(x) PSTR_(x)
asm(".text\n.skip " PSTR(PAD) "\n.section .text.startup,\"ax\",@progbits\n.skip " PSTR(PAD) "\n.text");

#define VL 32
#define K 256              // across-rows kernels: 32 rows x K
#define KB (K / VL)
#define NBLK 64            // along-vector kernels: 64 registers
#define NE (NBLK * VL)

#define VT_E8      0xC0UL
#define VT_E8S(lg) (0x8C0UL | ((unsigned long)(lg) << 12))
#define VT_BF16    0x1C9UL
#define VT_E16     0xC9UL
static inline void vset(unsigned long vt) { asm volatile("vsetvl x0, %0, %1" : : "r"((unsigned long)VL), "r"(vt)); }
#define LD8(v, p)  asm volatile("vle8.v "  v ", (%0)" : : "r"(p) : "memory")
#define LD16(v, p) asm volatile("vle16.v " v ", (%0)" : : "r"(p) : "memory")
#define ST8(v, p)  asm volatile("vse8.v "  v ", (%0)" : : "r"(p) : "memory")
#define ST16(v, p) asm volatile("vse16.v " v ", (%0)" : : "r"(p) : "memory")
#define WCVT(d, s) asm volatile(".insn r 0x57, 0x1, 0x25, " d ", x13, " s) // vfwcvtbf16.f.f.v
#define NCVT(d, s) asm volatile(".insn r 0x57, 0x1, 0x25, " d ", x29, " s) // vfncvtbf16.f.f.w

// ---- data -------------------------------------------------------------------
static uint16_t X[K][VL] __attribute__((aligned(64)));   // quantrow input, k-major
static uint8_t  srow[KB][VL] __attribute__((aligned(64)));
static uint8_t  Q[K][VL] __attribute__((aligned(64)));
static uint8_t  xa[K][VL] __attribute__((aligned(64))), ra[K][VL] __attribute__((aligned(64)));
static uint8_t  ya[K][VL] __attribute__((aligned(64)));
static uint8_t  sx[KB][VL] __attribute__((aligned(64))), sr[KB][VL] __attribute__((aligned(64)));
static uint8_t  sy[KB][VL] __attribute__((aligned(64)));
static uint8_t  W[K][VL] __attribute__((aligned(64)));
static uint8_t  sW[KB][VL] __attribute__((aligned(64)));
static uint8_t  xg[K] __attribute__((aligned(64)));
static uint16_t xg16[K] __attribute__((aligned(64)));
static uint16_t yg[VL] __attribute__((aligned(64)));
static uint8_t  x8[NE] __attribute__((aligned(64)));
static uint16_t xq[NE] __attribute__((aligned(64)));
static uint8_t  sc[NBLK + VL] __attribute__((aligned(64)));
static uint16_t y16[NE] __attribute__((aligned(64)));
static uint8_t  q8[NE] __attribute__((aligned(64)));
static uint16_t bf_s[KB][VL] __attribute__((aligned(64)));  // baseline: 2^(s-128) per row, BF16
static uint16_t bf_is[KB][VL] __attribute__((aligned(64))); // baseline: 2^(128-s)
static uint16_t bf_sx[KB][VL] __attribute__((aligned(64))), bf_sr[KB][VL] __attribute__((aligned(64)));
static uint16_t bf_isy[KB][VL] __attribute__((aligned(64)));
static uint16_t bf_sc[NBLK], bf_isc[NBLK];
static uint16_t Yb[K][VL] __attribute__((aligned(64)));    // across-row BF16 outputs
static uint16_t bf_srowd[KB][VL] __attribute__((aligned(64))); // baseline: 2^(srow-128)
static uint8_t  r8[NE] __attribute__((aligned(64)));
static uint8_t  scx[NBLK + VL] __attribute__((aligned(64))), scw[NBLK + VL] __attribute__((aligned(64)));
static uint16_t bf_cs[NBLK];                               // baseline: 2^(kx + kw)
static uint8_t  sc16[2 * NBLK + VL] __attribute__((aligned(64)));
static uint16_t bf_e16[NE] __attribute__((aligned(64))), bf_ie16[NE] __attribute__((aligned(64)));
static volatile uint8_t sink;
static uint8_t ref[16384] __attribute__((aligned(64))); // the scaled version's output
static void keep(const void *out, size_t n) { const uint8_t *b = out; for (size_t i = 0; i < n; i++) ref[i] = b[i]; }
static void differ(const char *kernel, const void *out, size_t n, int esize) {
	int d = 0;
	for (size_t i = 0; i < n / esize; i++)
		d += esize == 1 ? ((const uint8_t *)out)[i] != ref[i] : ((const uint16_t *)out)[i] != ((const uint16_t *)ref)[i];
	printf("DIFF   %-9s baseline differs from scaled in %d of %lu elements\n", kernel, d, (unsigned long)(n / esize));
}

static uint32_t seed = 4242;
static uint32_t rnd(void) { seed = seed * 1664525u + 1013904223u; return seed >> 8; }
static uint8_t rnd_fp8(void) { uint8_t c; do c = rnd(); while (c == 0x80 || (c & 0x7F) == 0x7F); return c; }
static uint16_t rnd_bf16(int lo, int hi) {
	unsigned e = 127 + lo + rnd() % (hi - lo + 1);
	return ((rnd() & 1) << 15) | (e << 7) | (rnd() & 0x7F);
}
static uint16_t bf16_pow2(int e) { // 2^e as BF16, e in -133..127
	if (e >= -126) return (uint16_t)((e + 127) << 7);
	return (uint16_t)(0x40 >> (-127 - e));
}
static uint32_t checksum(const void *p, size_t n) {
	const uint8_t *b = p; uint32_t h = 2166136261u;
	for (size_t i = 0; i < n; i++) h = (h ^ b[i]) * 16777619u;
	return h;
}

static unsigned long t0;
// waits for the previous call's vector work (as stop() does), so none of it is timed
static void start(const volatile void *last) {
	asm volatile("fence" : : : "memory");
	sink = *(const volatile uint8_t *)last;
	t0 = read_csr(mcycle);
}
static unsigned long stop(const volatile void *last) {
	asm volatile("fence" : : : "memory");
	sink = *(const volatile uint8_t *)last;
	return read_csr(mcycle) - t0;
}

// ---- quantrow: BF16 -> FP8 across rows, x4 ---------------------------------
static void quantrow_scaled(int scaled) {
	for (int b = 0; b < KB; b++) {
		vset(VT_E8);
		if (scaled) {
#ifdef SHARED
			LD8("v29", srow[b]);                                      // vd's shared scales
#else
			LD8("v9", srow[b]);                                       // rule A: a copy per vd
			asm volatile("vmv.v.v v11, v9\n\tvmv.v.v v13, v9\n\tvmv.v.v v15, v9");
#endif
		}
		vset(scaled ? VT_E8S(0) : VT_E8);
		for (int k = b * VL; k < (b + 1) * VL; k += 4) {
			LD16("v16", X[k]); LD16("v18", X[k + 1]); LD16("v20", X[k + 2]); LD16("v22", X[k + 3]);
			NCVT("x8", "x16"); NCVT("x10", "x18"); NCVT("x12", "x20"); NCVT("x14", "x22");
			ST8("v8", Q[k]); ST8("v10", Q[k + 1]); ST8("v12", Q[k + 2]); ST8("v14", Q[k + 3]);
		}
	}
}
static void quantrow_base(void) {
	for (int b = 0; b < KB; b++) {
		vset(VT_BF16);
		LD16("v24", bf_is[b]);
		for (int k = b * VL; k < (b + 1) * VL; k += 4) {
			vset(VT_BF16);
			LD16("v16", X[k]); LD16("v18", X[k + 1]); LD16("v20", X[k + 2]); LD16("v22", X[k + 3]);
			asm volatile("vfmul.vv v16, v16, v24\n\tvfmul.vv v18, v18, v24\n\tvfmul.vv v20, v20, v24\n\tvfmul.vv v22, v22, v24");
			vset(VT_E8);
			NCVT("x8", "x16"); NCVT("x10", "x18"); NCVT("x12", "x20"); NCVT("x14", "x22");
			ST8("v8", Q[k]); ST8("v10", Q[k + 1]); ST8("v12", Q[k + 2]); ST8("v14", Q[k + 3]);
		}
	}
}

// ---- addrow: FP8 block add across rows, x4 ---------------------------------
// vfadd.vv vd, vs2, vs1: x in vs2, r in vs1, y in vd
static void addrow_scaled(int scaled) {
	for (int b = 0; b < KB; b++) {
		vset(VT_E8);
		if (scaled) {
#ifdef SHARED
			LD8("v30", sx[b]); LD8("v31", sr[b]); LD8("v29", sy[b]);
#else
			// rule A: lane L uses x in v(6L+2), r in v(6L+4), y in v(6L+6)... with
			// scales in the next register: nine copies per block
			LD8("v3", sx[b]); LD8("v5", sr[b]); LD8("v7", sy[b]);
			asm volatile("vmv.v.v v9, v3\n\tvmv.v.v v11, v5\n\tvmv.v.v v13, v7\n\t"
			             "vmv.v.v v15, v3\n\tvmv.v.v v17, v5\n\tvmv.v.v v19, v7\n\t"
			             "vmv.v.v v21, v3\n\tvmv.v.v v23, v5\n\tvmv.v.v v25, v7");
#endif
		}
		vset(scaled ? VT_E8S(0) : VT_E8);
		for (int k = b * VL; k < (b + 1) * VL; k += 4) {
#ifdef SHARED
			LD8("v2", xa[k]);  LD8("v10", ra[k]);
			LD8("v4", xa[k+1]); LD8("v12", ra[k+1]);
			LD8("v6", xa[k+2]); LD8("v14", ra[k+2]);
			LD8("v8", xa[k+3]); LD8("v16", ra[k+3]);
			asm volatile("vfadd.vv v18, v2, v10\n\tvfadd.vv v20, v4, v12\n\tvfadd.vv v22, v6, v14\n\tvfadd.vv v24, v8, v16");
			ST8("v18", ya[k]); ST8("v20", ya[k+1]); ST8("v22", ya[k+2]); ST8("v24", ya[k+3]);
#else
			LD8("v2", xa[k]);   LD8("v4", ra[k]);
			LD8("v8", xa[k+1]); LD8("v10", ra[k+1]);
			LD8("v14", xa[k+2]); LD8("v16", ra[k+2]);
			LD8("v20", xa[k+3]); LD8("v22", ra[k+3]);
			asm volatile("vfadd.vv v6, v2, v4\n\tvfadd.vv v12, v8, v10\n\tvfadd.vv v18, v14, v16\n\tvfadd.vv v24, v20, v22");
			ST8("v6", ya[k]); ST8("v12", ya[k+1]); ST8("v18", ya[k+2]); ST8("v24", ya[k+3]);
#endif
		}
	}
}
// baseline: widen both to BF16, scale, add, unscale, narrow (x2: registers)
static void addrow_base(void) {
	for (int b = 0; b < KB; b++) {
		vset(VT_BF16);
		LD16("v24", bf_sx[b]); LD16("v26", bf_sr[b]); LD16("v28", bf_isy[b]);
		for (int k = b * VL; k < (b + 1) * VL; k += 2) {
			vset(VT_E8);
			LD8("v2", xa[k]); LD8("v3", ra[k]); LD8("v4", xa[k+1]); LD8("v5", ra[k+1]);
			WCVT("x8", "x2"); WCVT("x10", "x3"); WCVT("x12", "x4"); WCVT("x14", "x5");
			vset(VT_BF16);
			asm volatile("vfmul.vv v8, v8, v24\n\tvfmul.vv v10, v10, v26\n\tvfadd.vv v8, v8, v10\n\tvfmul.vv v8, v8, v28\n\t"
			             "vfmul.vv v12, v12, v24\n\tvfmul.vv v14, v14, v26\n\tvfadd.vv v12, v12, v14\n\tvfmul.vv v12, v12, v28");
			vset(VT_E8);
			NCVT("x16", "x8"); NCVT("x17", "x12");
			ST8("v16", ya[k]); ST8("v17", ya[k+1]);
		}
	}
}

// ---- gemv: y += W[:, j] x[j], x4 -------------------------------------------
static void gemv_scaled(int scaled) {
	vset(VT_BF16);
	asm volatile("vmv.v.i v8, 0\n\tvmv.v.i v10, 0\n\tvmv.v.i v12, 0\n\tvmv.v.i v14, 0");
	for (int b = 0; b < KB; b++) {
		vset(VT_E8);
		if (scaled) {
#ifdef SHARED
			LD8("v30", sW[b]);
#else
			LD8("v3", sW[b]);
			asm volatile("vmv.v.v v5, v3\n\tvmv.v.v v7, v3\n\tvmv.v.v v17, v3");
#endif
		}
		vset(scaled ? VT_E8S(0) : VT_E8);
		for (int j = b * VL; j < (b + 1) * VL; j += 4) {
			LD8("v2", W[j]); LD8("v4", W[j + 1]); LD8("v6", W[j + 2]); LD8("v16", W[j + 3]);
			asm volatile("fmv.w.x f0, %0\n\tvfwmacc.vf v8, f0, v2" : : "r"(0xFFFFFF00u | xg[j]) : "f0");
			asm volatile("fmv.w.x f0, %0\n\tvfwmacc.vf v10, f0, v4" : : "r"(0xFFFFFF00u | xg[j + 1]) : "f0");
			asm volatile("fmv.w.x f0, %0\n\tvfwmacc.vf v12, f0, v6" : : "r"(0xFFFFFF00u | xg[j + 2]) : "f0");
			asm volatile("fmv.w.x f0, %0\n\tvfwmacc.vf v14, f0, v16" : : "r"(0xFFFFFF00u | xg[j + 3]) : "f0");
		}
	}
	vset(VT_BF16);
	asm volatile("vfadd.vv v8, v8, v10\n\tvfadd.vv v12, v12, v14\n\tvfadd.vv v8, v8, v12");
	ST16("v8", yg);
}
static void gemv_base(void) {
	vset(VT_BF16);
	asm volatile("vmv.v.i v8, 0\n\tvmv.v.i v10, 0\n\tvmv.v.i v12, 0\n\tvmv.v.i v14, 0");
	for (int b = 0; b < KB; b++) {
		vset(VT_BF16);
		LD16("v24", bf_s[b]);
		for (int j = b * VL; j < (b + 1) * VL; j += 4) {
			vset(VT_E8);
			LD8("v2", W[j]); LD8("v3", W[j + 1]); LD8("v4", W[j + 2]); LD8("v5", W[j + 3]);
			WCVT("x16", "x2"); WCVT("x18", "x3"); WCVT("x20", "x4"); WCVT("x22", "x5");
			vset(VT_BF16);
			asm volatile("vfmul.vv v16, v16, v24\n\tvfmul.vv v18, v18, v24\n\tvfmul.vv v20, v20, v24\n\tvfmul.vv v22, v22, v24");
			asm volatile("fmv.w.x f0, %0\n\tvfmacc.vf v8, f0, v16" : : "r"(0xFFFF0000u | xg16[j]) : "f0");
			asm volatile("fmv.w.x f0, %0\n\tvfmacc.vf v10, f0, v18" : : "r"(0xFFFF0000u | xg16[j + 1]) : "f0");
			asm volatile("fmv.w.x f0, %0\n\tvfmacc.vf v12, f0, v20" : : "r"(0xFFFF0000u | xg16[j + 2]) : "f0");
			asm volatile("fmv.w.x f0, %0\n\tvfmacc.vf v14, f0, v22" : : "r"(0xFFFF0000u | xg16[j + 3]) : "f0");
		}
	}
	asm volatile("vfadd.vv v8, v8, v10\n\tvfadd.vv v12, v12, v14\n\tvfadd.vv v8, v8, v12");
	ST16("v8", yg);
}

// ---- dequant / quant along the vector, one scale per register (E = 32) -----
static void dequant_scaled(int scaled) {
	vset(scaled ? VT_E8S(5) : VT_E8);
	for (int j = 0; j < NBLK; j++) {
		LD8("v2", &x8[j * VL]);
#ifdef SHARED
		if (scaled) LD8("v30", &sc[j]);
#else
		if (scaled) LD8("v3", &sc[j]);
#endif
		WCVT("x8", "x2");
		ST16("v8", &y16[j * VL]);
	}
}
static void dequant_scalar(void) { // estimate: the scale fetched by the scalar core only
	vset(VT_E8);
	unsigned long acc = 0;
	for (int j = 0; j < NBLK; j++) {
		acc += *(volatile uint8_t *)&sc[j];
		LD8("v2", &x8[j * VL]);
		WCVT("x8", "x2");
		ST16("v8", &y16[j * VL]);
	}
	sink = acc;
}
static void dequant_base(void) {
	for (int j = 0; j < NBLK; j += 4) {
		vset(VT_E8);
		LD8("v2", &x8[j * VL]); LD8("v3", &x8[(j + 1) * VL]); LD8("v4", &x8[(j + 2) * VL]); LD8("v5", &x8[(j + 3) * VL]);
		WCVT("x8", "x2"); WCVT("x10", "x3"); WCVT("x12", "x4"); WCVT("x14", "x5");
		vset(VT_BF16);
		asm volatile("fmv.w.x f1, %0" : : "r"(0xFFFF0000u) : "f1");
		asm volatile("fmv.w.x f0, %0\n\tvfmul.vf v8, v8, f0\n\tvfadd.vf v8, v8, f1" : : "r"(0xFFFF0000u | bf_sc[j]) : "f0");
		asm volatile("fmv.w.x f0, %0\n\tvfmul.vf v10, v10, f0\n\tvfadd.vf v10, v10, f1" : : "r"(0xFFFF0000u | bf_sc[j + 1]) : "f0");
		asm volatile("fmv.w.x f0, %0\n\tvfmul.vf v12, v12, f0\n\tvfadd.vf v12, v12, f1" : : "r"(0xFFFF0000u | bf_sc[j + 2]) : "f0");
		asm volatile("fmv.w.x f0, %0\n\tvfmul.vf v14, v14, f0\n\tvfadd.vf v14, v14, f1" : : "r"(0xFFFF0000u | bf_sc[j + 3]) : "f0");
		ST16("v8", &y16[j * VL]); ST16("v10", &y16[(j + 1) * VL]); ST16("v12", &y16[(j + 2) * VL]); ST16("v14", &y16[(j + 3) * VL]);
	}
}
static void quant_scaled(int scaled) {
	vset(scaled ? VT_E8S(5) : VT_E8);
	for (int j = 0; j < NBLK; j++) {
		LD16("v16", &xq[j * VL]);
#ifdef SHARED
		if (scaled) LD8("v29", &sc[j]);
#else
		if (scaled) LD8("v9", &sc[j]);
#endif
		NCVT("x8", "x16");
		ST8("v8", &q8[j * VL]);
	}
}
static void quant_scalar(void) {
	vset(VT_E8);
	unsigned long acc = 0;
	for (int j = 0; j < NBLK; j++) {
		acc += *(volatile uint8_t *)&sc[j];
		LD16("v16", &xq[j * VL]);
		NCVT("x8", "x16");
		ST8("v8", &q8[j * VL]);
	}
	sink = acc;
}
static void quant_base(void) {
	for (int j = 0; j < NBLK; j += 4) {
		vset(VT_BF16);
		LD16("v16", &xq[j * VL]); LD16("v18", &xq[(j + 1) * VL]); LD16("v20", &xq[(j + 2) * VL]); LD16("v22", &xq[(j + 3) * VL]);
		asm volatile("fmv.w.x f0, %0\n\tvfmul.vf v16, v16, f0" : : "r"(0xFFFF0000u | bf_isc[j]) : "f0");
		asm volatile("fmv.w.x f0, %0\n\tvfmul.vf v18, v18, f0" : : "r"(0xFFFF0000u | bf_isc[j + 1]) : "f0");
		asm volatile("fmv.w.x f0, %0\n\tvfmul.vf v20, v20, f0" : : "r"(0xFFFF0000u | bf_isc[j + 2]) : "f0");
		asm volatile("fmv.w.x f0, %0\n\tvfmul.vf v22, v22, f0" : : "r"(0xFFFF0000u | bf_isc[j + 3]) : "f0");
		vset(VT_E8);
		NCVT("x8", "x16"); NCVT("x9", "x18"); NCVT("x10", "x20"); NCVT("x11", "x22");
		ST8("v8", &q8[j * VL]); ST8("v9", &q8[(j + 1) * VL]); ST8("v10", &q8[(j + 2) * VL]); ST8("v11", &q8[(j + 3) * VL]);
	}
}

// ---- dequantrow: FP8 -> BF16 across rows, x4 -------------------------------
static void dequantrow_scaled(int scaled) {
	for (int b = 0; b < KB; b++) {
		vset(VT_E8);
		if (scaled) {
#ifdef SHARED
			LD8("v30", srow[b]);
#else
			LD8("v3", srow[b]);
			asm volatile("vmv.v.v v5, v3\n\tvmv.v.v v7, v3\n\tvmv.v.v v17, v3");
#endif
		}
		vset(scaled ? VT_E8S(0) : VT_E8);
		for (int k = b * VL; k < (b + 1) * VL; k += 4) {
			LD8("v2", xa[k]); LD8("v4", xa[k + 1]); LD8("v6", xa[k + 2]); LD8("v16", xa[k + 3]);
			WCVT("x8", "x2"); WCVT("x10", "x4"); WCVT("x12", "x6"); WCVT("x14", "x16");
			ST16("v8", Yb[k]); ST16("v10", Yb[k + 1]); ST16("v12", Yb[k + 2]); ST16("v14", Yb[k + 3]);
		}
	}
}
static void dequantrow_base(void) {
	for (int b = 0; b < KB; b++) {
		vset(VT_BF16);
		LD16("v24", bf_srowd[b]);
		asm volatile("fmv.w.x f1, %0" : : "r"(0xFFFF0000u) : "f1");
		for (int k = b * VL; k < (b + 1) * VL; k += 4) {
			vset(VT_E8);
			LD8("v2", xa[k]); LD8("v3", xa[k + 1]); LD8("v4", xa[k + 2]); LD8("v5", xa[k + 3]);
			WCVT("x8", "x2"); WCVT("x10", "x3"); WCVT("x12", "x4"); WCVT("x14", "x5");
			vset(VT_BF16);
			asm volatile("vfmul.vv v8, v8, v24\n\tvfmul.vv v10, v10, v24\n\tvfmul.vv v12, v12, v24\n\tvfmul.vv v14, v14, v24\n\t"
			             "vfadd.vf v8, v8, f1\n\tvfadd.vf v10, v10, f1\n\tvfadd.vf v12, v12, f1\n\tvfadd.vf v14, v14, f1");
			ST16("v8", Yb[k]); ST16("v10", Yb[k + 1]); ST16("v12", Yb[k + 2]); ST16("v14", Yb[k + 3]);
		}
	}
}

// ---- smulrow: ScaledMultiply across rows, x4 (vfwmul.vv vd, vs2 = x, vs1 = w)
// rule A: x in v2 v6 v18 v22, w in v4 v16 v20 v24, each with its scales next
static void smulrow_scaled(int scaled) {
	for (int b = 0; b < KB; b++) {
		vset(VT_E8);
		if (scaled) {
#ifdef SHARED
			LD8("v30", sx[b]); LD8("v31", sr[b]);
#else
			LD8("v3", sx[b]); LD8("v5", sr[b]);
			asm volatile("vmv.v.v v7, v3\n\tvmv.v.v v19, v3\n\tvmv.v.v v23, v3\n\t"
			             "vmv.v.v v17, v5\n\tvmv.v.v v21, v5\n\tvmv.v.v v25, v5");
#endif
		}
		vset(scaled ? VT_E8S(0) : VT_E8);
		for (int k = b * VL; k < (b + 1) * VL; k += 4) {
			LD8("v2", xa[k]); LD8("v4", ra[k]); LD8("v6", xa[k + 1]); LD8("v16", ra[k + 1]);
			LD8("v18", xa[k + 2]); LD8("v20", ra[k + 2]); LD8("v22", xa[k + 3]); LD8("v24", ra[k + 3]);
			asm volatile("vfwmul.vv v8, v2, v4\n\tvfwmul.vv v10, v6, v16\n\tvfwmul.vv v12, v18, v20\n\tvfwmul.vv v14, v22, v24");
			ST16("v8", Yb[k]); ST16("v10", Yb[k + 1]); ST16("v12", Yb[k + 2]); ST16("v14", Yb[k + 3]);
		}
	}
}
static void smulrow_base(void) { // widen both, scale both, multiply (x2: registers)
	for (int b = 0; b < KB; b++) {
		vset(VT_BF16);
		LD16("v24", bf_sx[b]); LD16("v26", bf_sr[b]);
		asm volatile("fmv.w.x f1, %0" : : "r"(0xFFFF0000u) : "f1");
		for (int k = b * VL; k < (b + 1) * VL; k += 2) {
			vset(VT_E8);
			LD8("v2", xa[k]); LD8("v3", ra[k]); LD8("v4", xa[k + 1]); LD8("v5", ra[k + 1]);
			WCVT("x8", "x2"); WCVT("x10", "x3"); WCVT("x12", "x4"); WCVT("x14", "x5");
			vset(VT_BF16);
			asm volatile("vfmul.vv v8, v8, v24\n\tvfmul.vv v10, v10, v26\n\tvfmul.vv v8, v8, v10\n\tvfadd.vf v8, v8, f1\n\t"
			             "vfmul.vv v12, v12, v24\n\tvfmul.vv v14, v14, v26\n\tvfmul.vv v12, v12, v14\n\tvfadd.vf v12, v12, f1");
			ST16("v8", Yb[k]); ST16("v12", Yb[k + 1]);
		}
	}
}

// ---- smul: ScaledMultiply along the vector, one scale per operand per register
static void smul_scaled(int scaled) {
	vset(scaled ? VT_E8S(5) : VT_E8);
	for (int j = 0; j < NBLK; j++) {
		LD8("v2", &x8[j * VL]); LD8("v4", &r8[j * VL]);
#ifdef SHARED
		if (scaled) { LD8("v30", &scx[j]); LD8("v31", &scw[j]); }
#else
		if (scaled) { LD8("v3", &scx[j]); LD8("v5", &scw[j]); }
#endif
		asm volatile("vfwmul.vv v8, v2, v4");
		ST16("v8", &y16[j * VL]);
	}
}
static void smul_scalar(void) {
	vset(VT_E8);
	unsigned long acc = 0;
	for (int j = 0; j < NBLK; j++) {
		acc += *(volatile uint8_t *)&scx[j] + *(volatile uint8_t *)&scw[j];
		LD8("v2", &x8[j * VL]); LD8("v4", &r8[j * VL]);
		asm volatile("vfwmul.vv v8, v2, v4");
		ST16("v8", &y16[j * VL]);
	}
	sink = acc;
}
static void smul_base(void) { // exact unscaled product, then the combined scale
	for (int j = 0; j < NBLK; j += 4) {
		vset(VT_E8);
		LD8("v2", &x8[j * VL]); LD8("v3", &x8[(j + 1) * VL]); LD8("v4", &x8[(j + 2) * VL]); LD8("v5", &x8[(j + 3) * VL]);
		LD8("v6", &r8[j * VL]); LD8("v7", &r8[(j + 1) * VL]); LD8("v16", &r8[(j + 2) * VL]); LD8("v17", &r8[(j + 3) * VL]);
		asm volatile("vfwmul.vv v8, v2, v6\n\tvfwmul.vv v10, v3, v7\n\tvfwmul.vv v12, v4, v16\n\tvfwmul.vv v14, v5, v17");
		vset(VT_BF16);
		asm volatile("fmv.w.x f1, %0" : : "r"(0xFFFF0000u) : "f1");
		asm volatile("fmv.w.x f0, %0\n\tvfmul.vf v8, v8, f0\n\tvfadd.vf v8, v8, f1" : : "r"(0xFFFF0000u | bf_cs[j]) : "f0");
		asm volatile("fmv.w.x f0, %0\n\tvfmul.vf v10, v10, f0\n\tvfadd.vf v10, v10, f1" : : "r"(0xFFFF0000u | bf_cs[j + 1]) : "f0");
		asm volatile("fmv.w.x f0, %0\n\tvfmul.vf v12, v12, f0\n\tvfadd.vf v12, v12, f1" : : "r"(0xFFFF0000u | bf_cs[j + 2]) : "f0");
		asm volatile("fmv.w.x f0, %0\n\tvfmul.vf v14, v14, f0\n\tvfadd.vf v14, v14, f1" : : "r"(0xFFFF0000u | bf_cs[j + 3]) : "f0");
		ST16("v8", &y16[j * VL]); ST16("v10", &y16[(j + 1) * VL]); ST16("v12", &y16[(j + 2) * VL]); ST16("v14", &y16[(j + 3) * VL]);
	}
}

// ---- dequant16 / quant16: two blocks of 16 per register (E = 16) -----------
static void dequant16_scaled(int scaled) {
	vset(scaled ? VT_E8S(4) : VT_E8);
	for (int j = 0; j < NBLK; j++) {
		LD8("v2", &x8[j * VL]);
#ifdef SHARED
		if (scaled) LD8("v30", &sc16[2 * j]);
#else
		if (scaled) LD8("v3", &sc16[2 * j]);
#endif
		WCVT("x8", "x2");
		ST16("v8", &y16[j * VL]);
	}
}
static void dequant16_base(void) { // per-element BF16 scales (expanded in memory)
	asm volatile("fmv.w.x f1, %0" : : "r"(0xFFFF0000u) : "f1");
	for (int j = 0; j < NBLK; j += 4) {
		vset(VT_E8);
		LD8("v2", &x8[j * VL]); LD8("v3", &x8[(j + 1) * VL]); LD8("v4", &x8[(j + 2) * VL]); LD8("v5", &x8[(j + 3) * VL]);
		WCVT("x8", "x2"); WCVT("x10", "x3"); WCVT("x12", "x4"); WCVT("x14", "x5");
		vset(VT_BF16);
		LD16("v16", &bf_e16[j * VL]); LD16("v18", &bf_e16[(j + 1) * VL]); LD16("v20", &bf_e16[(j + 2) * VL]); LD16("v22", &bf_e16[(j + 3) * VL]);
		asm volatile("vfmul.vv v8, v8, v16\n\tvfmul.vv v10, v10, v18\n\tvfmul.vv v12, v12, v20\n\tvfmul.vv v14, v14, v22\n\t"
		             "vfadd.vf v8, v8, f1\n\tvfadd.vf v10, v10, f1\n\tvfadd.vf v12, v12, f1\n\tvfadd.vf v14, v14, f1");
		ST16("v8", &y16[j * VL]); ST16("v10", &y16[(j + 1) * VL]); ST16("v12", &y16[(j + 2) * VL]); ST16("v14", &y16[(j + 3) * VL]);
	}
}
// estimates for E = 16: the register's two scales fetched as one 16-bit scalar
static void dequant16_scalar(void) {
	vset(VT_E8);
	unsigned long acc = 0;
	for (int j = 0; j < NBLK; j++) {
		acc += *(volatile uint16_t *)&sc16[2 * j];
		LD8("v2", &x8[j * VL]);
		WCVT("x8", "x2");
		ST16("v8", &y16[j * VL]);
	}
	sink = acc;
}
static void quant16_scalar(void) {
	vset(VT_E8);
	unsigned long acc = 0;
	for (int j = 0; j < NBLK; j++) {
		acc += *(volatile uint16_t *)&sc16[2 * j];
		LD16("v16", &xq[j * VL]);
		NCVT("x8", "x16");
		ST8("v8", &q8[j * VL]);
	}
	sink = acc;
}
static void quant16_scaled(int scaled) {
	vset(scaled ? VT_E8S(4) : VT_E8);
	for (int j = 0; j < NBLK; j++) {
		LD16("v16", &xq[j * VL]);
#ifdef SHARED
		if (scaled) LD8("v29", &sc16[2 * j]);
#else
		if (scaled) LD8("v9", &sc16[2 * j]);
#endif
		NCVT("x8", "x16");
		ST8("v8", &q8[j * VL]);
	}
}
static void quant16_base(void) {
	for (int j = 0; j < NBLK; j += 4) {
		vset(VT_BF16);
		LD16("v16", &xq[j * VL]); LD16("v18", &xq[(j + 1) * VL]); LD16("v20", &xq[(j + 2) * VL]); LD16("v22", &xq[(j + 3) * VL]);
		LD16("v24", &bf_ie16[j * VL]); LD16("v26", &bf_ie16[(j + 1) * VL]); LD16("v28", &bf_ie16[(j + 2) * VL]); LD16("v30", &bf_ie16[(j + 3) * VL]);
		asm volatile("vfmul.vv v16, v16, v24\n\tvfmul.vv v18, v18, v26\n\tvfmul.vv v20, v20, v28\n\tvfmul.vv v22, v22, v30");
		vset(VT_E8);
		NCVT("x8", "x16"); NCVT("x9", "x18"); NCVT("x10", "x20"); NCVT("x11", "x22");
		ST8("v8", &q8[j * VL]); ST8("v9", &q8[(j + 1) * VL]); ST8("v10", &q8[(j + 2) * VL]); ST8("v11", &q8[(j + 3) * VL]);
	}
}

// ---- driver -----------------------------------------------------------------
#ifdef SHARED
#define SCHEME "shared"
#else
#define SCHEME "ruleA"
#endif
#define TIME(label, call, last, out, n) do { \
		call; start(last); for (int r_ = 0; r_ < REPS; r_++) { call; } unsigned long c = stop(last) / REPS; \
		printf("RESULT %-9s %-9s %7lu cycles  %lu.%03lu cycles/element  checksum %08lx\n", kernel, label, c, \
		       c / (elems), c * 1000 / (elems) % 1000, (unsigned long)checksum(out, n)); \
	} while (0)

int main(void) {
	for (int k = 0; k < K; k++)
		for (int r = 0; r < VL; r++) {
			X[k][r] = rnd_bf16(-4, 12); xa[k][r] = rnd_fp8(); ra[k][r] = rnd_fp8(); W[k][r] = rnd_fp8();
		}
	for (int b = 0; b < KB; b++)
		for (int r = 0; r < VL; r++) {
			srow[b][r] = 128 + rnd() % 21 - 10; sW[b][r] = 128 + rnd() % 9 - 4;
			sx[b][r] = 128 + rnd() % 9 - 4; sr[b][r] = 128 + rnd() % 9 - 4; sy[b][r] = 128 + rnd() % 5 - 2;
			bf_s[b][r] = bf16_pow2(sW[b][r] - 128); bf_is[b][r] = bf16_pow2(128 - srow[b][r]);
			bf_sx[b][r] = bf16_pow2(sx[b][r] - 128); bf_sr[b][r] = bf16_pow2(sr[b][r] - 128);
			bf_isy[b][r] = bf16_pow2(128 - sy[b][r]);
		}
	for (int j = 0; j < K; j++) xg[j] = rnd_fp8();
	for (int i = 0; i < NE; i++) { x8[i] = rnd_fp8(); xq[i] = rnd_bf16(-4, 12); }
	for (int j = 0; j < NBLK + VL; j++) sc[j] = 128 + rnd() % 21 - 10;
	for (int j = 0; j < NBLK; j++) { bf_sc[j] = bf16_pow2(sc[j] - 128); bf_isc[j] = bf16_pow2(128 - sc[j]); }
	for (int b = 0; b < KB; b++) for (int r = 0; r < VL; r++) bf_srowd[b][r] = bf16_pow2(srow[b][r] - 128);
	for (int i = 0; i < NE; i++) r8[i] = rnd_fp8();
	for (int j = 0; j < NBLK + VL; j++) { scx[j] = 128 + rnd() % 9 - 4; scw[j] = 128 + rnd() % 9 - 4; }
	for (int j = 0; j < NBLK; j++) bf_cs[j] = bf16_pow2(scx[j] + scw[j] - 256);
	for (int j = 0; j < 2 * NBLK + VL; j++) sc16[j] = 128 + rnd() % 21 - 10;
	for (int i = 0; i < NE; i++) { bf_e16[i] = bf16_pow2(sc16[i / 16] - 128); bf_ie16[i] = bf16_pow2(128 - sc16[i / 16]); }
	for (int j = 0; j < K; j += VL) { // x in BF16 for the baseline gemv (exact)
		vset(VT_E8); LD8("v2", &xg[j]); WCVT("x16", "x2"); vset(VT_E16); ST16("v16", &xg16[j]);
	}

	const char *kernel; unsigned long elems;
	kernel = "quantrow"; elems = K * VL;
	TIME(SCHEME, quantrow_scaled(1), &Q[K - 1][VL - 1], Q, sizeof Q);
	keep(Q, sizeof Q);
#ifndef SHARED
	TIME("unscaled", quantrow_scaled(0), &Q[K - 1][VL - 1], Q, 0);
	TIME("baseline", quantrow_base(), &Q[K - 1][VL - 1], Q, sizeof Q);
	differ(kernel, Q, sizeof Q, 1);
#endif
	kernel = "addrow"; elems = K * VL;
	TIME(SCHEME, addrow_scaled(1), &ya[K - 1][VL - 1], ya, sizeof ya);
	keep(ya, sizeof ya);
#ifndef SHARED
	TIME("unscaled", addrow_scaled(0), &ya[K - 1][VL - 1], ya, 0);
	TIME("baseline", addrow_base(), &ya[K - 1][VL - 1], ya, sizeof ya);
	differ(kernel, ya, sizeof ya, 1);
#endif
	kernel = "gemv"; elems = K * VL;
	TIME(SCHEME, gemv_scaled(1), &yg[VL - 1], yg, sizeof yg);
	keep(yg, sizeof yg);
#ifndef SHARED
	TIME("unscaled", gemv_scaled(0), &yg[VL - 1], yg, 0);
	TIME("baseline", gemv_base(), &yg[VL - 1], yg, sizeof yg);
	differ(kernel, yg, sizeof yg, 2);
#endif
	kernel = "dequant"; elems = NE;
	TIME(SCHEME, dequant_scaled(1), &y16[NE - 1], y16, sizeof y16);
	keep(y16, sizeof y16);
#ifndef SHARED
	TIME("unscaled", dequant_scaled(0), &y16[NE - 1], y16, 0);
	TIME("scalar", dequant_scalar(), &y16[NE - 1], y16, 0);
	TIME("baseline", dequant_base(), &y16[NE - 1], y16, sizeof y16);
	differ(kernel, y16, sizeof y16, 2);
#endif
	kernel = "quant"; elems = NE;
	TIME(SCHEME, quant_scaled(1), &q8[NE - 1], q8, sizeof q8);
	keep(q8, sizeof q8);
#ifndef SHARED
	TIME("unscaled", quant_scaled(0), &q8[NE - 1], q8, 0);
	TIME("scalar", quant_scalar(), &q8[NE - 1], q8, 0);
	TIME("baseline", quant_base(), &q8[NE - 1], q8, sizeof q8);
	differ(kernel, q8, sizeof q8, 1);
#endif
	kernel = "dequantrow"; elems = K * VL;
	TIME(SCHEME, dequantrow_scaled(1), &Yb[K - 1][VL - 1], Yb, sizeof Yb);
	keep(Yb, sizeof Yb);
#ifndef SHARED
	TIME("unscaled", dequantrow_scaled(0), &Yb[K - 1][VL - 1], Yb, 0);
	TIME("baseline", dequantrow_base(), &Yb[K - 1][VL - 1], Yb, sizeof Yb);
	differ(kernel, Yb, sizeof Yb, 2);
#endif
	kernel = "smulrow"; elems = K * VL;
	TIME(SCHEME, smulrow_scaled(1), &Yb[K - 1][VL - 1], Yb, sizeof Yb);
	keep(Yb, sizeof Yb);
#ifndef SHARED
	TIME("unscaled", smulrow_scaled(0), &Yb[K - 1][VL - 1], Yb, 0);
	TIME("baseline", smulrow_base(), &Yb[K - 1][VL - 1], Yb, sizeof Yb);
	differ(kernel, Yb, sizeof Yb, 2);
#endif
	kernel = "smul"; elems = NE;
	TIME(SCHEME, smul_scaled(1), &y16[NE - 1], y16, sizeof y16);
	keep(y16, sizeof y16);
#ifndef SHARED
	TIME("unscaled", smul_scaled(0), &y16[NE - 1], y16, 0);
	TIME("scalar", smul_scalar(), &y16[NE - 1], y16, 0);
	TIME("baseline", smul_base(), &y16[NE - 1], y16, sizeof y16);
	differ(kernel, y16, sizeof y16, 2);
#endif
	kernel = "dequant16"; elems = NE;
	TIME(SCHEME, dequant16_scaled(1), &y16[NE - 1], y16, sizeof y16);
	keep(y16, sizeof y16);
#ifndef SHARED
	TIME("unscaled", dequant16_scaled(0), &y16[NE - 1], y16, 0);
	TIME("scalar", dequant16_scalar(), &y16[NE - 1], y16, 0);
	TIME("baseline", dequant16_base(), &y16[NE - 1], y16, sizeof y16);
	differ(kernel, y16, sizeof y16, 2);
#endif
	kernel = "quant16"; elems = NE;
	TIME(SCHEME, quant16_scaled(1), &q8[NE - 1], q8, sizeof q8);
	keep(q8, sizeof q8);
#ifndef SHARED
	TIME("unscaled", quant16_scaled(0), &q8[NE - 1], q8, 0);
	TIME("scalar", quant16_scalar(), &q8[NE - 1], q8, 0);
	TIME("baseline", quant16_base(), &q8[NE - 1], q8, sizeof q8);
	differ(kernel, q8, sizeof q8, 1);
#endif
	return 0;
}
