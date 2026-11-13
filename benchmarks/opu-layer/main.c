// The linear part of one Qwen2.5-0.5B decoder layer, prefill of T = 64 tokens,
// on OPUV256D128P3109BlockShuttleConfig, twice on the same chip:
//   hw  block scaling in hardware: activations quantized with vfncvtbf16 in
//       scaled mode (scales in vd+1), GEMMs with OPFMACC.S (rule A scales)
//   sw  the same layer with unscaled instructions: activations multiplied by
//       2^-s then converted; GEMMs with unscaled OPFMACC, each 32-k block's
//       partial sums moved out, scaled by sA * sB in FP32 and accumulated in
//       memory (opu-timing's pattern W)
// Weights binary8p4 with one scale per output feature per 32 inputs (offline);
// activations BF16, quantized at run time to binary8p4 with one scale per token
// per 32 features (MX-style: from the block's largest magnitude).
//
// Every activation is stored k-major, X[feature][token], and each GEMM
// computes C^T = W X^T: weights are the OPU's A operand (W[k][n]), activations
// its B operand, so a tile row is one output feature for 32 tokens and goes
// straight to the next layer's input layout.
//
// Projections (K x N): q, o 896 x 896; k, v 896 x 128 (2 KV heads); gate, up
// 896 x 4864; down 4864 x 896. Attention itself is not computed: q's output
// stands in for the attention output that feeds o. Also: the two residual
// adds and SiLU(gate) * up, identical in both versions.
// Data is generated on the chip (untimed); timing does not depend on values.
#pragma GCC optimize("no-tree-vectorize")
#include <stdio.h>
#include <stdint.h>
#include "util.h"
#include "rvv_mx.h"
#include "bme.h"

#define T  64
#ifndef BS // block size along k: elements per scale (multiple of 4 dividing H and F)
#define BS 32
#endif
#if BS != 32 && (defined(PROBE) || defined(PROBESW) || defined(PROBEK) || defined(PROBE3) || defined(PROBED) || defined(PROBEV))
#error "the probes assume BS = 32"
#endif
#ifdef SMALL // smoke test
#define H  128
#define KV 64
#define F  256
#elif !defined(H) // -DH= -DKV= -DF= for other sizes (multiples of 64)
#define H  896
#define KV 128
#define F  4864
#endif

#define VT_E8      0xC0UL  // e8 m1: binary8p4
#define VT_E8S(lb) (0x8C0UL | ((unsigned long)(lb) << 12)) // scaled mode
#define VT_E16     0xC9UL  // e16 m2, integer
#define VT_BF16    0x1C9UL // e16 m2, altfmt: BF16
#define VT_E32     0xD2UL  // e32 m4
static inline void vsetl(unsigned long avl, unsigned long vt) { asm volatile("vsetvl x0, %0, %1" : : "r"(avl), "r"(vt)); }
static inline void vset(unsigned long vt) { vsetl(32, vt); }
#define NCVT(d, s) ".insn r 0x57, 0x1, 0x25, " d ", x29, " s  // vfncvtbf16.f.f.w (SEW 8)

#define A64 __attribute__((aligned(64)))
// weights W[k][n], scales per 32 k and output feature: codes and 2^(s-128) in FP32
static uint8_t Wq[H * H] A64, Wk[H * KV] A64, Wv[H * KV] A64, Wo[H * H] A64;
static uint8_t Wg[H * F] A64, Wu[H * F] A64, Wd[F * H] A64;
static uint8_t sWq[H / BS * H] A64, sWk[H / BS * KV] A64, sWv[H / BS * KV] A64, sWo[H / BS * H] A64;
static uint8_t sWg[H / BS * F] A64, sWu[H / BS * F] A64, sWd[F / BS * H] A64;
static uint32_t fWq[H / BS * H] A64, fWk[H / BS * KV] A64, fWv[H / BS * KV] A64, fWo[H / BS * H] A64;
static uint32_t fWg[H / BS * F] A64, fWu[H / BS * F] A64, fWd[F / BS * H] A64;
// activations, BF16 [feature][token]
static uint16_t X[H * T] A64, Qo[H * T] A64, Ko[KV * T] A64, Vo[KV * T] A64, Oo[H * T] A64;
static uint16_t Hh[H * T] A64, G[F * T] A64, U[F * T] A64, Ac[F * T] A64, Do[H * T] A64;
static uint16_t Out_hw[H * T] A64, Out_sw[H * T] A64;
// quantized activations [feature][token] and their scales [block][token]
static uint8_t Xq[F * T] A64, sX[F / BS * T] A64;
static uint32_t fX[F / BS * T] A64;
static uint32_t Cacc[64 * T] A64; // sw: one 64-feature block of C^T in FP32

// ---- data (untimed) ------------------------------------------------------
// LCG state in v8 (e32 m8, 64 lanes); each step leaves 256 random bytes in v16
static void rng_init(uint32_t seed) {
	vsetl(64, 0xD3UL);
	asm volatile("vid.v v8\n\tvmul.vx v8, v8, %0\n\tvadd.vx v8, v8, %1" : : "r"(2654435761UL), "r"((unsigned long)seed));
}
static void rng_step(void) {
	vsetl(64, 0xD3UL);
	asm volatile("vmul.vx v8, v8, %0\n\tvadd.vx v8, v8, %1\n\tvsrl.vi v16, v8, 13\n\tvxor.vv v16, v16, v8"
	             : : "r"(1664525UL), "r"(1013904223UL));
}
// binary8p4 codes with exponent field 4..11: no zero, NaN or infinity
static void gen_fp8(uint8_t *p, unsigned long n, uint32_t seed) {
	rng_init(seed);
	for (unsigned long i = 0; i < n; i += 256) {
		rng_step();
		vsetl(256, 0xC3UL);
		asm volatile("vand.vx v24, v16, %0\n\tvsrl.vi v16, v16, 3\n\tvand.vi v16, v16, 7\n\t"
		             "vadd.vi v16, v16, 4\n\tvsll.vi v16, v16, 3\n\tvor.vv v24, v24, v16\n\tvse8.v v24, (%1)"
		             : : "r"(0x87UL), "r"(p + i) : "memory");
	}
}
// scale codes 118..121 and their FP32 values 2^(s-128)
static void gen_scales(uint8_t *s, uint32_t *f, unsigned long n, uint32_t seed) {
	rng_init(seed);
	for (unsigned long i = 0; i < n; i += 64) {
		rng_step();
		vsetl(64, 0xC1UL); // e8 m2
		asm volatile("vand.vi v24, v16, 3\n\tvadd.vx v24, v24, %0\n\tvse8.v v24, (%1)" : : "r"(118UL), "r"(s + i) : "memory");
		vsetl(64, 0xD3UL); // e32 m8
		asm volatile("vzext.vf4 v0, v24\n\tvadd.vi v0, v0, -1\n\tvsll.vi v0, v0, 23\n\tvse32.v v0, (%0)" : : "r"(f + i) : "memory");
	}
}
// BF16 in +-[0.5, 2)
static void gen_bf16(uint16_t *p, unsigned long n, uint32_t seed) {
	rng_init(seed);
	for (unsigned long i = 0; i < n; i += 128) {
		rng_step();
		vsetl(128, 0xCBUL);
		asm volatile("vand.vx v24, v16, %0\n\tvsrl.vi v16, v16, 8\n\tvand.vi v16, v16, 1\n\t"
		             "vadd.vx v16, v16, %1\n\tvsll.vi v16, v16, 7\n\tvor.vv v24, v24, v16\n\tvse16.v v24, (%2)"
		             : : "r"(0x807FUL), "r"(126UL), "r"(p + i) : "memory");
	}
}

// ---- timing ----------------------------------------------------------------
static unsigned long tick(void) { return read_csr(mcycle); }
static unsigned long tock(unsigned long t0, const volatile void *last) {
	asm volatile("fence" : : : "memory");
	(void)*(const volatile uint8_t *)last; // waits for the last vector store
	return read_csr(mcycle) - t0;
}

// ---- quantization of Y[K][T] (BF16) to Yq, one scale per token per BS k ----
// Scale code from the block's largest magnitude: its BF16 exponent - 5, so
// the largest scaled value is below 2^7 (binary8p4's largest finite is 224).
static void amax_code(const uint16_t *y) {           // codes in v20 (e16 m2)
	vset(VT_E16);
	asm volatile("vmv.v.i v16, 0");
	for (int f = 0; f < BS; f++)
		asm volatile("vle16.v v18, (%0)\n\tvand.vx v18, v18, %1\n\tvmaxu.vv v16, v16, v18"
		             : : "r"(y + f * T), "r"(0x7FFFUL) : "memory");
	asm volatile("vsrl.vi v20, v16, 7\n\tvadd.vi v20, v20, -5\n\tvmax.vx v20, v20, %0\n\tvmin.vx v20, v20, %1"
	             : : "r"(1UL), "r"(254UL));
}
static void quant_hw(const uint16_t *Y, int K, uint8_t *Yq) {
	for (int b = 0; b < K / BS; b++)
		for (int h = 0; h < T; h += 32) {
			const uint16_t *y = Y + b * BS * T + h;
			amax_code(y);
			vset(VT_E8);
			asm volatile("vnsrl.wi v9, v20, 0\n\tvse8.v v9, (%0)" : : "r"(sX + b * T + h) : "memory");
			vset(VT_E8S(0));
			for (int f = 0; f < BS; f++) {
				asm volatile("vle16.v v16, (%0)" : : "r"(y + f * T) : "memory");
				asm volatile(NCVT("x8", "x16"));
				asm volatile("vse8.v v8, (%0)" : : "r"(Yq + (b * BS + f) * T + h) : "memory");
			}
		}
}
static void quant_sw(const uint16_t *Y, int K, uint8_t *Yq) {
	for (int b = 0; b < K / BS; b++)
		for (int h = 0; h < T; h += 32) {
			const uint16_t *y = Y + b * BS * T + h;
			amax_code(y);
			asm volatile("vrsub.vx v24, v20, %0\n\tvsll.vi v24, v24, 7" : : "r"(255UL)); // BF16 2^(128-s)
			vset(VT_E32);
			asm volatile("vzext.vf2 v28, v20\n\tvadd.vi v28, v28, -1\n\tvsll.vi v28, v28, 23\n\tvse32.v v28, (%0)"
			             : : "r"(fX + b * T + h) : "memory"); // FP32 2^(s-128), for the GEMM
			for (int f = 0; f < BS; f += 4) {
				vset(VT_BF16);
				asm volatile("vle16.v v16, (%0)\n\tvle16.v v18, (%1)\n\tvle16.v v26, (%2)\n\tvle16.v v28, (%3)"
				             : : "r"(y + f * T), "r"(y + (f + 1) * T), "r"(y + (f + 2) * T), "r"(y + (f + 3) * T) : "memory");
				asm volatile("vfmul.vv v16, v16, v24\n\tvfmul.vv v18, v18, v24\n\tvfmul.vv v26, v26, v24\n\tvfmul.vv v28, v28, v24");
				vset(VT_E8);
				asm volatile(NCVT("x8", "x16") "\n\t" NCVT("x9", "x18") "\n\t" NCVT("x10", "x26") "\n\t" NCVT("x11", "x28"));
				uint8_t *q = Yq + (b * BS + f) * T + h;
				asm volatile("vse8.v v8, (%0)\n\tvse8.v v9, (%1)\n\tvse8.v v10, (%2)\n\tvse8.v v11, (%3)"
				             : : "r"(q), "r"(q + T), "r"(q + 2 * T), "r"(q + 3 * T) : "memory");
			}
		}
}

// ---- GEMM: C^T[N][T] = W^T-blocks x Xq, 2x2 tiles of 32 x 32 --------------
// m0 = (features n0..+31, tokens 0..31), m1 = (n0.., tokens 32..), m2 = (n0+32..,
// tokens 0..), m3 = (n0+32.., tokens 32..). A0, A1 (weights) in v16, v18,
// their scales in v17, v19; B0, B1 (tokens) in v20, v22, scales in v21, v23.
// Tile rows leave through v0..v15, which hold no scale register.
#define LD(v, p) asm volatile("vle8.v " v ", (%0)" : : "r"(p) : "memory")
#define KSTEP(MACC, w, x) LD("v16", w); LD("v18", (w) + 32); LD("v20", x); LD("v22", (x) + 32); \
	MACC(m0, v20, v16); MACC(m1, v22, v16); MACC(m2, v20, v18); MACC(m3, v22, v18);
#define KSTEP2(MACC, w, x) LD("v24", w); LD("v26", (w) + 32); LD("v28", x); LD("v30", (x) + 32); \
	MACC(m0, v28, v24); MACC(m1, v30, v24); MACC(m2, v28, v26); MACC(m3, v30, v26);
#ifdef SETTLE // diagnostic: wait for the OPU to finish everything queued, then a delay, before reading tiles
static void settle(void) {
	unsigned long x;
	vset(VT_E32);
	VMV_VR(v12, 0, m3);
	asm volatile("vmv.x.s %0, v12" : "=r"(x));
	for (volatile int i = 0; i < 200; i++) ;
	(void)x;
}
#define SETTLE_POINT settle();
#else
#define SETTLE_POINT
#endif
static void clear_tiles(void) {
	vset(VT_E32);
	asm volatile("vmv.v.i v12, 0");
	OPMVINBCAST(m0, v12); OPMVINBCAST(m1, v12); OPMVINBCAST(m2, v12); OPMVINBCAST(m3, v12);
}
// tile rows out, FP32 -> BF16, to out[row][token]; 4 rows per vtype switch
#define TILE_OUT(mt, out, coff) \
	for (unsigned long r = 0; r < 32; r += 4) { \
		vset(VT_E32); \
		VMV_VR(v0, r, mt); VMV_VR(v4, r + 1, mt); VMV_VR(v8, r + 2, mt); VMV_VR(v12, r + 3, mt); \
		vset(VT_BF16); \
		asm volatile("vfncvt.f.f.w v16, v0\n\tvfncvt.f.f.w v18, v4\n\tvfncvt.f.f.w v20, v8\n\tvfncvt.f.f.w v22, v12"); \
		uint16_t *o_ = (out) + r * T + (coff); \
		asm volatile("vse16.v v16, (%0)\n\tvse16.v v18, (%1)\n\tvse16.v v20, (%2)\n\tvse16.v v22, (%3)" \
		             : : "r"(o_), "r"(o_ + T), "r"(o_ + 2 * T), "r"(o_ + 3 * T) : "memory"); \
	}
// -DGEMMONLY adds the range of 64-feature tile columns computed (all of them in the layer)
#ifdef GEMMONLY
#define COLS_ARGS , int n_lo, int n_hi
#define ALLCOLS(N) , 0, (N)
#else
#define COLS_ARGS
#define ALLCOLS(N)
#define n_lo 0
#define n_hi N
#endif
static void gemm_hw(const uint8_t *W, const uint8_t *sW, int N, int K, uint16_t *out COLS_ARGS) {
	for (int n0 = n_lo; n0 < n_hi; n0 += 64) {
#ifdef DRAIN // diagnostic: let every older vector store finish first
		asm volatile("fence" : : : "memory");
		(void)*(volatile uint8_t *)(n0 ? (void *)&out[n0 * T - 1] : (void *)&Xq[K * T - 1]);
#endif
		clear_tiles();
		vset(VT_E8);
		for (int b = 0; b < K / BS; b++) {
			LD("v17", sW + b * N + n0); LD("v19", sW + b * N + n0 + 32);
			LD("v21", sX + b * T); LD("v23", sX + b * T + 32);
			for (int k = b * BS; k < b * BS + BS; k++) {
				KSTEP(OPFMACCS, W + k * N + n0, Xq + k * T)
			}
		}
		SETTLE_POINT
		uint16_t *o = out + n0 * T;
		TILE_OUT(m0, o, 0) TILE_OUT(m1, o, 32) TILE_OUT(m2, o + 32 * T, 0) TILE_OUT(m3, o + 32 * T, 32)
	}
}

// sw: after each k block, C += sW[n] * (tile row * sX[tokens]) in FP32
#define TILE_ACC(mt, roff, coff, sx) \
	for (unsigned long r = 0; r < 32; r++) { \
		uint32_t *c_ = Cacc + ((roff) + r) * T + (coff); \
		const uint32_t *w_ = fW + b * N + n0 + (roff) + r; \
		VMV_VR(v12, r, mt); \
		if (b == 0) asm volatile("flw f0, 0(%0)\n\tvfmul.vv v12, v12, " sx "\n\tvfmul.vf v16, v12, f0\n\tvse32.v v16, (%1)" \
		                         : : "r"(w_), "r"(c_) : "f0", "memory"); \
		else asm volatile("flw f0, 0(%0)\n\tvfmul.vv v12, v12, " sx "\n\tvle32.v v16, (%1)\n\tvfmacc.vf v16, f0, v12\n\tvse32.v v16, (%1)" \
		                  : : "r"(w_), "r"(c_) : "f0", "memory"); \
	}
static void gemm_sw(const uint8_t *W, const uint32_t *fW, int N, int K, uint16_t *out COLS_ARGS) {
	for (int n0 = n_lo; n0 < n_hi; n0 += 64) {
		for (int b = 0; b < K / BS; b++) {
			clear_tiles();
			vset(VT_E8);
#ifdef DBUF // diagnostic: even k in v16..v22, odd k in v24..v30
			for (int k = b * BS; k < b * BS + BS; k += 2) {
				KSTEP(OPFMACC, W + k * N + n0, Xq + k * T)
				KSTEP2(OPFMACC, W + (k + 1) * N + n0, Xq + (k + 1) * T)
			}
#else
			for (int k = b * BS; k < b * BS + BS; k++) {
				KSTEP(OPFMACC, W + k * N + n0, Xq + k * T)
			}
#endif
			SETTLE_POINT
			vset(VT_E32);
			asm volatile("vle32.v v24, (%0)\n\tvle32.v v28, (%1)" : : "r"(fX + b * T), "r"(fX + b * T + 32) : "memory");
			TILE_ACC(m0, 0, 0, "v24") TILE_ACC(m1, 0, 32, "v28") TILE_ACC(m2, 32, 0, "v24") TILE_ACC(m3, 32, 32, "v28")
		}
		for (int r = 0; r < 64; r += 2) {
			vset(VT_E32);
			asm volatile("vle32.v v0, (%0)\n\tvle32.v v4, (%1)\n\tvle32.v v8, (%2)\n\tvle32.v v12, (%3)"
			             : : "r"(Cacc + r * T), "r"(Cacc + r * T + 32), "r"(Cacc + (r + 1) * T), "r"(Cacc + (r + 1) * T + 32) : "memory");
			vset(VT_BF16);
			asm volatile("vfncvt.f.f.w v16, v0\n\tvfncvt.f.f.w v18, v4\n\tvfncvt.f.f.w v20, v8\n\tvfncvt.f.f.w v22, v12");
			uint16_t *o = out + (n0 + r) * T;
			asm volatile("vse16.v v16, (%0)\n\tvse16.v v18, (%1)\n\tvse16.v v20, (%2)\n\tvse16.v v22, (%3)"
			             : : "r"(o), "r"(o + 32), "r"(o + T), "r"(o + T + 32) : "memory");
		}
	}
}
#ifndef GEMMONLY
#undef n_lo
#undef n_hi
#endif

// ---- element-wise, identical in both versions ------------------------------
static void add_bf16(const uint16_t *a, const uint16_t *b, uint16_t *c, int n) {
	vset(VT_BF16);
	for (int i = 0; i < n; i += 64)
		asm volatile("vle16.v v16, (%0)\n\tvle16.v v18, (%1)\n\tvle16.v v20, (%2)\n\tvle16.v v22, (%3)\n\t"
		             "vfadd.vv v16, v16, v20\n\tvfadd.vv v18, v18, v22\n\tvse16.v v16, (%4)\n\tvse16.v v18, (%5)"
		             : : "r"(a + i), "r"(a + i + 32), "r"(b + i), "r"(b + i + 32), "r"(c + i), "r"(c + i + 32) : "memory");
}
// SiLU(g) * u in FP32: e^-x = 2^n * p(f), n = round(-x log2 e), p cubic on [-0.5, 0.5]
static void silu_mul(const uint16_t *g, const uint16_t *u, uint16_t *a, int n) {
	for (int i = 0; i < n; i += 32) {
		vset(VT_BF16);
		asm volatile("vle16.v v4, (%0)\n\tvle16.v v6, (%1)\n\tvfwcvt.f.f.v v8, v4\n\tvfwcvt.f.f.v v12, v6"
		             : : "r"(g + i), "r"(u + i) : "memory");
		vset(VT_E32);
		asm volatile("vfmul.vf v16, v8, %[nl2e]\n\tvfmin.vf v16, v16, %[hi]\n\tvfmax.vf v16, v16, %[lo]\n\t"
		             "vfcvt.x.f.v v20, v16\n\tvfcvt.f.x.v v24, v20\n\tvfsub.vv v16, v16, v24\n\t"
		             "vfmul.vf v24, v16, %[c3]\n\tvfadd.vf v24, v24, %[c2]\n\tvfmul.vv v24, v24, v16\n\t"
		             "vfadd.vf v24, v24, %[c1]\n\tvfmul.vv v24, v24, v16\n\tvfadd.vf v24, v24, %[one]\n\t"
		             "vadd.vx v20, v20, %[b]\n\tvsll.vi v20, v20, 23\n\tvfmul.vv v24, v24, v20\n\t"
		             "vfadd.vf v24, v24, %[one]\n\t"
		             // 1 / d without the divider (one element at a time): r = magic - bits(d),
		             // then three Newton steps r = r (2 - d r)
		             "vrsub.vx v16, v24, %[magic]\n\t"
		             "vfmv.v.f v20, %[two]\n\tvfnmsac.vv v20, v24, v16\n\tvfmul.vv v16, v16, v20\n\t"
		             "vfmv.v.f v20, %[two]\n\tvfnmsac.vv v20, v24, v16\n\tvfmul.vv v16, v16, v20\n\t"
		             "vfmv.v.f v20, %[two]\n\tvfnmsac.vv v20, v24, v16\n\tvfmul.vv v16, v16, v20\n\t"
		             "vfmul.vv v24, v16, v8\n\tvfmul.vv v24, v24, v12"
		             : : [nl2e] "f"(-1.4426950f), [hi] "f"(126.0f), [lo] "f"(-126.0f), [c3] "f"(0.0555041f),
		                 [c2] "f"(0.2402265f), [c1] "f"(0.6931472f), [one] "f"(1.0f), [two] "f"(2.0f),
		                 [b] "r"(127UL), [magic] "r"(0x7EF311C7UL));
		vset(VT_BF16);
		asm volatile("vfncvt.f.f.w v28, v24\n\tvse16.v v28, (%0)" : : "r"(a + i) : "memory");
	}
}

// ---- the layer -------------------------------------------------------------
enum { S_QUANT, S_GEMM_QKV, S_GEMM_O, S_GEMM_GU, S_GEMM_D, S_ELEM, NS };
static const char *sname[NS] = {"quantize", "gemm qkv", "gemm o", "gemm gate+up", "gemm down", "element-wise"};
static unsigned long st[NS];
#define STAGE(s, last, body) do { unsigned long t_ = tick(); body; st[s] += tock(t_, last); } while (0)

static void layer(int hw, uint16_t *out) {
	for (int i = 0; i < NS; i++) st[i] = 0;
#define QUANT(Y, K) STAGE(S_QUANT, &Xq[(K) * T - 1], hw ? quant_hw(Y, K, Xq) : quant_sw(Y, K, Xq))
#define GEMM(s, W, sW, fW, N, K, o) STAGE(s, &(o)[(N) * T - 1], hw ? gemm_hw(W, sW, N, K, o ALLCOLS(N)) : gemm_sw(W, fW, N, K, o ALLCOLS(N)))
	QUANT(X, H);
	GEMM(S_GEMM_QKV, Wq, sWq, fWq, H, H, Qo);
	GEMM(S_GEMM_QKV, Wk, sWk, fWk, KV, H, Ko);
	GEMM(S_GEMM_QKV, Wv, sWv, fWv, KV, H, Vo);
	QUANT(Qo, H);                                   // stands in for the attention output
	GEMM(S_GEMM_O, Wo, sWo, fWo, H, H, Oo);
	STAGE(S_ELEM, &Hh[H * T - 1], add_bf16(X, Oo, Hh, H * T));
	QUANT(Hh, H);
	GEMM(S_GEMM_GU, Wg, sWg, fWg, F, H, G);
	GEMM(S_GEMM_GU, Wu, sWu, fWu, F, H, U);
	STAGE(S_ELEM, &Ac[F * T - 1], silu_mul(G, U, Ac, F * T));
	QUANT(Ac, F);
	GEMM(S_GEMM_D, Wd, sWd, fWd, H, F, Do);
	STAGE(S_ELEM, &out[H * T - 1], add_bf16(Hh, Do, out, H * T));
}

static unsigned long report(const char *name) {
	unsigned long tot = 0;
	for (int i = 0; i < NS; i++) tot += st[i];
	printf("RESULT %s total %lu", name, tot);
	for (int i = 0; i < NS; i++) printf(" | %s %lu", sname[i], st[i]);
	printf("\n");
	return tot;
}

// BF16 as float, for the comparison
static float bf(uint16_t v) { union { uint32_t u; float f; } x = {.u = (uint32_t)v << 16}; return x.f; }

#if defined(PROBE) || defined(PROBESW) || defined(PROBEK) || defined(PROBE3) || defined(PROBED) || defined(PROBEV)
// -DPROBE (with -DSMALL): quantization and the q GEMM only, both versions, against a scalar reference
static uint8_t pXq[H * T], psX[H / 32 * T];
static uint16_t pQo[H * T];
static double fp8(uint8_t c) {           // binary8p4: bias 8, 3 fraction bits
	if (c == 0x80) return 0.0 / 0.0;
	int s = c >> 7, e = (c >> 3) & 15, m = c & 7;
	double v = e ? (1.0 + m / 8.0) * (double)(1L << e) / 256.0 : m / 1024.0;
	if ((c & 0x7F) == 0x7F) v = 1.0 / 0.0;
	return s ? -v : v;
}
static double p2(int code) { double v = 1; for (int i = 128; i < code; i++) v *= 2; for (int i = code; i < 128; i++) v /= 2; return v; }
static long fx(double v) { return (long)(v * 1e6); } // micro-units, for printf without %f
static void probe(void) {
	quant_hw(X, H, Xq);
	for (int i = 0; i < H * T; i++) pXq[i] = Xq[i];
	for (int i = 0; i < H / 32 * T; i++) psX[i] = sX[i];
	gemm_hw(Wq, sWq, H, H, Qo);
	for (int i = 0; i < H * T; i++) pQo[i] = Qo[i];
	quant_sw(X, H, Xq);
	int dq = 0, bad = 0;
	for (int i = 0; i < H * T; i++) { dq += pXq[i] != Xq[i]; bad += pXq[i] == 0x80 || (pXq[i] & 0x7F) == 0x7F; }
	printf("PROBE Xq: hw vs sw differ in %d of %d, hw NaN/Inf codes %d\n", dq, H * T, bad);
	for (int i = 0; i < H / 32 * T; i += 7)
		if (fx(p2(psX[i])) != fx(*(float *)&fX[i]) && i < 64) printf("  scale[%d] code %d fX %08x\n", i, psX[i], fX[i]);
	gemm_sw(Wq, fWq, H, H, Qo);
	const int ns[] = {0, 1, 31, 32, 63, 64, 127}, ts[] = {0, 5, 19, 20, 21, 31, 32, 63};
	for (int a = 0; a < 7; a++) for (int c = 0; c < 8; c++) {
		int n = ns[a], t = ts[c];
		double r = 0;
		for (int k = 0; k < H; k++)
			r += fp8(Wq[k * H + n]) * p2(sWq[(k / 32) * H + n]) * fp8(pXq[k * T + t]) * p2(psX[(k / 32) * T + t]);
		printf("PROBE q[%d][%d] ref %ld hw %ld sw %ld (e-6)\n", n, t, fx(r), fx(bf(pQo[n * T + t])), fx(bf(Qo[n * T + t])));
	}
}
#endif
#ifdef PROBESW
// -DPROBESW (with -DSMALL): unscaled GEMM only, whole rows against a scalar reference
static float fw[H * H], fxq[H * T];
static void probe_sw(void) {
	quant_sw(X, H, Xq);
	asm volatile("fence" : : : "memory"); (void)*(volatile uint8_t *)&Xq[H * T - 1];
	gemm_sw(Wq, fWq, H, H, Qo);
	asm volatile("fence" : : : "memory"); (void)*(volatile uint16_t *)&Qo[H * T - 1];
	for (int i = 0; i < H * H; i++) { union { uint32_t u; float f; } s = {.u = fWq[(i / H / 32) * H + i % H]}; fw[i] = (float)fp8(Wq[i]) * s.f; }
	for (int i = 0; i < H * T; i++) { union { uint32_t u; float f; } s = {.u = fX[(i / T / 32) * T + i % T]}; fxq[i] = (float)fp8(Xq[i]) * s.f; }
	const int ns[] = {0, 1, 31, 32, 63, 64, 127};
	int bad = 0;
	for (int a = 0; a < 7; a++) {
		int n = ns[a];
		printf("ROW %3d bad tokens:", n);
		for (int t = 0; t < T; t++) {
			float r = 0;
			for (int k = 0; k < H; k++) r += fw[k * H + n] * fxq[k * T + t];
			float g = bf(Qo[n * T + t]), d = g > r ? g - r : r - g;
			if (!(d <= 0.02f * (r > 0 ? r : -r) + 0.002f)) { printf(" %d", t); bad++; }
		}
		printf("\n");
	}
	printf("PROBESW %d bad of %d\n", bad, 7 * T);
}
#endif
#ifdef PROBEK
// -DPROBEK (with -DSMALL): one OPFMACC per k, products of tokens 5, 20, 40 for all 64 rows of
// the first tile column, against the exact product of the two FP8 values
static uint32_t rowbuf[T] __attribute__((aligned(64)));
static void probe_k(void) {
	quant_sw(X, H, Xq);
	asm volatile("fence" : : : "memory"); (void)*(volatile uint8_t *)&Xq[H * T - 1];
	const int toks[] = {5, 20, 40};
	int bad = 0, shown = 0;
	for (int k = 0; k < H; k++) {
		clear_tiles();
		vset(VT_E8);
		KSTEP(OPFMACC, Wq + k * H, Xq + k * T)
		for (int r = 0; r < 64; r++) {
			vset(VT_E32);
			if (r < 32) { VMV_VR(v12, (unsigned long)r, m0); asm volatile("vse32.v v12, (%0)" : : "r"(rowbuf) : "memory");
			              VMV_VR(v12, (unsigned long)r, m1); asm volatile("vse32.v v12, (%0)" : : "r"(rowbuf + 32) : "memory"); }
			else        { VMV_VR(v12, (unsigned long)(r - 32), m2); asm volatile("vse32.v v12, (%0)" : : "r"(rowbuf) : "memory");
			              VMV_VR(v12, (unsigned long)(r - 32), m3); asm volatile("vse32.v v12, (%0)" : : "r"(rowbuf + 32) : "memory"); }
			asm volatile("fence" : : : "memory"); (void)*(volatile uint32_t *)&rowbuf[T - 1];
			for (int q = 0; q < 3; q++) {
				int t = toks[q];
				uint8_t w = Wq[k * H + r], x = Xq[k * T + t];
				double ref = fp8(w) * fp8(x);
				union { uint32_t u; float f; } g = {.u = rowbuf[t]};
				if ((double)g.f != ref) {
					bad++;
					if (shown++ < 24) printf("PROBEK k=%d row=%d tok=%d w=%02x x=%02x ref %ld got %ld (e-6) bits %08x\n",
					                         k, r, t, w, x, (long)(ref * 1e6), (long)((double)g.f * 1e6), rowbuf[t]);
				}
			}
		}
	}
	printf("PROBEK %d wrong of %d products\n", bad, H * 64 * 3);
}
#endif
#ifdef PROBE3
// -DPROBE3 (with -DSMALL): (A) single scaled OPFMACC.S steps vs the exact scaled product,
// (B) 32 accumulated unscaled steps vs the exact sum, (C) vector store then load, same address
static uint32_t rb[T] __attribute__((aligned(64)));
static uint32_t stl[64 * T] __attribute__((aligned(64))), stl2[64 * T] __attribute__((aligned(64)));
static void rows_out(int r) {
	vset(VT_E32);
	if (r < 32) { VMV_VR(v12, (unsigned long)r, m0); asm volatile("vse32.v v12, (%0)" : : "r"(rb) : "memory");
	              VMV_VR(v12, (unsigned long)r, m1); asm volatile("vse32.v v12, (%0)" : : "r"(rb + 32) : "memory"); }
	else        { VMV_VR(v12, (unsigned long)(r - 32), m2); asm volatile("vse32.v v12, (%0)" : : "r"(rb) : "memory");
	              VMV_VR(v12, (unsigned long)(r - 32), m3); asm volatile("vse32.v v12, (%0)" : : "r"(rb + 32) : "memory"); }
	asm volatile("fence" : : : "memory"); (void)*(volatile uint32_t *)&rb[T - 1];
}
static void probe3(void) {
	quant_hw(X, H, Xq);
	asm volatile("fence" : : : "memory"); (void)*(volatile uint8_t *)&Xq[H * T - 1];
	const int toks[] = {5, 19, 20, 21, 40};
	int bad = 0, shown = 0;
	for (int k = 0; k < H; k++) {                          // (A)
		int b = k / 32;
		clear_tiles();
		vset(VT_E8);
		LD("v17", sWq + b * H); LD("v19", sWq + b * H + 32); LD("v21", sX + b * T); LD("v23", sX + b * T + 32);
		KSTEP(OPFMACCS, Wq + k * H, Xq + k * T)
		for (int r = 0; r < 64; r++) {
			rows_out(r);
			for (int q = 0; q < 5; q++) {
				int t = toks[q];
				double ref = fp8(Wq[k * H + r]) * p2(sWq[b * H + r]) * fp8(Xq[k * T + t]) * p2(sX[b * T + t]);
				union { uint32_t u; float f; } g = {.u = rb[t]};
				if ((double)g.f != ref) { bad++; if (shown++ < 12)
					printf("A k=%d row=%d tok=%d w=%02x x=%02x sw=%d sx=%d ref %ld got %ld (e-9)\n", k, r, t, Wq[k * H + r], Xq[k * T + t],
					       sWq[b * H + r], sX[b * T + t], (long)(ref * 1e9), (long)((double)g.f * 1e9)); }
			}
		}
	}
	printf("PROBE3 A: %d wrong of %d scaled products\n", bad, H * 64 * 5);
	bad = 0; shown = 0;                                     // (B) block 0, unscaled, accumulated
	clear_tiles();
	vset(VT_E8);
	for (int k = 0; k < 32; k++) { KSTEP(OPFMACC, Wq + k * H, Xq + k * T) }
	for (int r = 0; r < 64; r++) {
		rows_out(r);
		for (int t = 0; t < T; t++) {
			double ref = 0;
			for (int k = 0; k < 32; k++) ref += fp8(Wq[k * H + r]) * fp8(Xq[k * T + t]);
			union { uint32_t u; float f; } g = {.u = rb[t]};
			double d = (double)g.f - ref; if (d < 0) d = -d;
			if (d > 1e-5 * (ref > 0 ? ref : -ref) + 1e-6) { bad++; if (shown++ < 12)
				printf("B row=%d tok=%d ref %ld got %ld (e-6)\n", r, t, (long)(ref * 1e6), (long)((double)g.f * 1e6)); }
		}
	}
	printf("PROBE3 B: %d wrong of %d accumulated sums\n", bad, 64 * T);
	bad = 0;                                                // (C) store then load the same row
	vset(VT_E32);
	for (int i = 0; i < 64 * T; i += 32) {
		asm volatile("vid.v v16\n\tvadd.vx v16, v16, %0\n\tvse32.v v16, (%1)\n\tvle32.v v20, (%1)\n\tvse32.v v20, (%2)"
		             : : "r"((long)i), "r"(stl + i), "r"(stl2 + i) : "memory");
	}
	asm volatile("fence" : : : "memory"); (void)*(volatile uint32_t *)&stl2[64 * T - 1];
	for (int i = 0; i < 64 * T; i++) bad += stl2[i] != (uint32_t)i;
	printf("PROBE3 C: %d of %d words wrong after store-then-load\n", bad, 64 * T);
}
#endif
#ifdef PROBED
// -DPROBED=1|2|3 (with -DSMALL): a GEMM of the q projection over PKD k (default 64) for the 64
// features from PN0 (default 0), all 64 x 64 outputs against the exact result. 1: hw (OPFMACC.S);
// 2: hw, OPU drained before each block's scale loads; 3: sw (unscaled, drained, scaled and
// accumulated in memory per block). -DDUMP (with 3): v24 and v28 stored before and after the
// block's TILE_ACCs, compared with fX
#ifndef PKD
#define PKD 64
#endif
#ifndef PN0
#define PN0 0
#endif
static uint32_t rbd[T] __attribute__((aligned(64)));
static double wd[PKD * 64], xd[PKD * T];
#ifdef DUMP
static uint32_t dmp[4][PKD / 32 * T] __attribute__((aligned(64)));
#endif
static double p2f(int code) { union { uint64_t u; double d; } x = {.u = (uint64_t)(code - 128 + 1023) << 52}; return x.d; }
static void opu_drain(void) {
	unsigned long x;
	vset(VT_E32);
	VMV_VR(v12, 0, m3);
	asm volatile("vmv.x.s %0, v12" : "=r"(x));
	(void)x;
}
static void probe_d(void) {
	const int KD = PKD, N = H, n0 = PN0;
#if PROBED == 3
	quant_sw(X, H, Xq);
#else
	quant_hw(X, H, Xq);
#endif
	asm volatile("fence" : : : "memory"); (void)*(volatile uint8_t *)&Xq[H * T - 1];
#if PROBED == 3
	const uint32_t *fW = fWq;
	for (int b = 0; b < KD / 32; b++) {
		clear_tiles();
		vset(VT_E8);
		for (int k = b * 32; k < b * 32 + 32; k++) { KSTEP(OPFMACC, Wq + k * N + n0, Xq + k * T) }
		vset(VT_E32);
		asm volatile("vle32.v v24, (%0)\n\tvle32.v v28, (%1)" : : "r"(fX + b * T), "r"(fX + b * T + 32) : "memory");
#ifdef DUMP
		asm volatile("vse32.v v24, (%0)\n\tvse32.v v28, (%1)" : : "r"(dmp[0] + b * T), "r"(dmp[0] + b * T + 32) : "memory");
#endif
		TILE_ACC(m0, 0, 0, "v24") TILE_ACC(m1, 0, 32, "v28") TILE_ACC(m2, 32, 0, "v24") TILE_ACC(m3, 32, 32, "v28")
#ifdef DUMP
		vset(VT_E32);
		asm volatile("vse32.v v24, (%0)\n\tvse32.v v28, (%1)" : : "r"(dmp[1] + b * T), "r"(dmp[1] + b * T + 32) : "memory");
#endif
	}
	asm volatile("fence" : : : "memory"); (void)*(volatile uint32_t *)&Cacc[64 * T - 1];
#ifdef DUMP
	for (int j = 0; j < 2; j++) {
		int bad = 0, shown = 0;
		for (int i = 0; i < KD / 32 * T; i++)
			if (dmp[j][i] != fX[i]) { bad++; if (shown++ < 8) printf("DUMP %s b=%d tok=%d fX %08x reg %08x\n", j ? "after" : "before", i / T, i % T, fX[i], dmp[j][i]); }
		printf("DUMP %s: %d of %d scale words differ\n", j ? "after" : "before", bad, KD / 32 * T);
	}
#endif
#else
	clear_tiles();
	vset(VT_E8);
	for (int b = 0; b < KD / 32; b++) {
#if PROBED == 2
		opu_drain(); vset(VT_E8);
#endif
		LD("v17", sWq + b * N + n0); LD("v19", sWq + b * N + n0 + 32);
		LD("v21", sX + b * T); LD("v23", sX + b * T + 32);
		for (int k = b * 32; k < b * 32 + 32; k++) { KSTEP(OPFMACCS, Wq + k * N + n0, Xq + k * T) }
	}
#endif
	for (int k = 0; k < KD; k++) {
		int b = k / 32;
		for (int r = 0; r < 64; r++) {
#if PROBED == 3
			union { uint32_t u; float f; } s_ = {.u = fW[b * N + n0 + r]}; wd[k * 64 + r] = fp8(Wq[k * N + n0 + r]) * s_.f;
#else
			wd[k * 64 + r] = fp8(Wq[k * N + n0 + r]) * p2f(sWq[b * N + n0 + r]);
#endif
		}
		for (int t = 0; t < T; t++) {
#if PROBED == 3
			union { uint32_t u; float f; } s_ = {.u = fX[b * T + t]}; xd[k * T + t] = fp8(Xq[k * T + t]) * s_.f;
#else
			xd[k * T + t] = fp8(Xq[k * T + t]) * p2f(sX[b * T + t]);
#endif
		}
	}
	int bad = 0, shown = 0, tb[4] = {0}, tmin[4] = {T, T, T, T}, tmax[4] = {-1, -1, -1, -1};
	for (int r = 0; r < 64; r++) {
#if PROBED == 3
		for (int t = 0; t < T; t++) rbd[t] = Cacc[r * T + t];
#else
		vset(VT_E32);
		if (r < 32) { VMV_VR(v12, (unsigned long)r, m0); asm volatile("vse32.v v12, (%0)" : : "r"(rbd) : "memory");
		              VMV_VR(v12, (unsigned long)r, m1); asm volatile("vse32.v v12, (%0)" : : "r"(rbd + 32) : "memory"); }
		else        { VMV_VR(v12, (unsigned long)(r - 32), m2); asm volatile("vse32.v v12, (%0)" : : "r"(rbd) : "memory");
		              VMV_VR(v12, (unsigned long)(r - 32), m3); asm volatile("vse32.v v12, (%0)" : : "r"(rbd + 32) : "memory"); }
		asm volatile("fence" : : : "memory"); (void)*(volatile uint32_t *)&rbd[T - 1];
#endif
		for (int t = 0; t < T; t++) {
			double ref = 0, mag = 0;
			for (int k = 0; k < KD; k++) { double term = wd[k * 64 + r] * xd[k * T + t]; ref += term; mag += term > 0 ? term : -term; }
			union { uint32_t u; float f; } g = {.u = rbd[t]};
			double d = (double)g.f - ref; if (d < 0) d = -d;
			if (!(d <= 1e-5 * mag + 1e-9)) { int q = (r >= 32) * 2 + (t >= 32); tb[q]++; if (t < tmin[q]) tmin[q] = t; if (t > tmax[q]) tmax[q] = t; bad++; if (shown++ < 10)
				printf("D row=%d tok=%d ref %ld got %ld (e-9)\n", r, t, (long)(ref * 1e9), (long)((double)g.f * 1e9)); }
		}
	}
	for (int q = 0; q < 4; q++) printf("tile m%d: %d wrong, tokens %d..%d\n", q, tb[q], tmin[q], tmax[q]);
	printf("PROBED%d: %d wrong of %d outputs (K = %d, features %d..%d)\n", PROBED, bad, 64 * T, KD, n0, n0 + 63);
}
#endif
#ifdef PROBEV
// -DPROBEV (with -DSMALL): which instruction of TILE_ACC changes the scale register v28.
// v24, v28 loaded from fX, then: 1 moves out only; 2 moves out and vfmul by v28; 3 TILE_ACC;
// 4 TILE_ACC with the OPU idle (no OPFMACC before); 5 TILE_ACC with v28 stored after each row
static uint32_t vd_[2][T] __attribute__((aligned(64))), rowd[32][32] __attribute__((aligned(64)));
static void probe_v(void) {
	const int N = H, n0 = 0, b = 0;
	const uint32_t *fW = fWq;
	quant_sw(X, H, Xq);
	asm volatile("fence" : : : "memory"); (void)*(volatile uint8_t *)&Xq[H * T - 1];
	for (int v = 1; v <= 10; v++) {
		clear_tiles();
		vset(VT_E8);
		if (v != 4)
			for (int k = 0; k < 32; k++) { KSTEP(OPFMACC, Wq + k * N + n0, Xq + k * T) }
		vset(VT_E32);
		asm volatile("vle32.v v24, (%0)\n\tvle32.v v28, (%1)" : : "r"(fX), "r"(fX + 32) : "memory");
		if (v == 1) for (unsigned long r = 0; r < 32; r++) { VMV_VR(v12, r, m1); }
		if (v == 2) for (unsigned long r = 0; r < 32; r++) { VMV_VR(v12, r, m1); asm volatile("vfmul.vv v12, v12, v28"); }
		if (v == 3 || v == 4) { TILE_ACC(m1, 0, 32, "v28") }
		if (v == 5)
			for (unsigned long r = 0; r < 32; r++) {
				uint32_t *c_ = Cacc + r * T + 32;
				VMV_VR(v12, r, m1);
				asm volatile("flw f0, 0(%0)\n\tvfmul.vv v12, v12, v28\n\tvfmul.vf v16, v12, f0\n\tvse32.v v16, (%1)\n\tvse32.v v28, (%2)"
				             : : "r"(fW + r), "r"(c_), "r"(rowd[r]) : "f0", "memory");
			}
		// 6: no store of v16; 7: no vfmul.vf (v12 stored); 8: vfmul.vf into v20; 9: no flw (f0 preset); 10: = 3
		for (unsigned long r = 0; v >= 6 && r < 32; r++) {
			uint32_t *c_ = Cacc + r * T + 32;
			VMV_VR(v12, r, m1);
			if (v == 6) asm volatile("flw f0, 0(%0)\n\tvfmul.vv v12, v12, v28\n\tvfmul.vf v16, v12, f0" : : "r"(fW + r) : "f0");
			if (v == 7) asm volatile("flw f0, 0(%0)\n\tvfmul.vv v12, v12, v28\n\tvse32.v v12, (%1)" : : "r"(fW + r), "r"(c_) : "f0", "memory");
			if (v == 8) asm volatile("flw f0, 0(%0)\n\tvfmul.vv v12, v12, v28\n\tvfmul.vf v20, v12, f0\n\tvse32.v v20, (%1)" : : "r"(fW + r), "r"(c_) : "f0", "memory");
			if (v == 9) asm volatile("vfmul.vv v12, v12, v28\n\tvfmul.vf v16, v12, f0\n\tvse32.v v16, (%0)" : : "r"(c_) : "memory");
			if (v == 10) asm volatile("flw f0, 0(%0)\n\tvfmul.vv v12, v12, v28\n\tvfmul.vf v16, v12, f0\n\tvse32.v v16, (%1)" : : "r"(fW + r), "r"(c_) : "f0", "memory");
		}
		vset(VT_E32);
		asm volatile("vse32.v v24, (%0)\n\tvse32.v v28, (%1)" : : "r"(vd_[0]), "r"(vd_[1]) : "memory");
		asm volatile("fence" : : : "memory"); (void)*(volatile uint32_t *)&vd_[1][31];
		int bad24 = 0, bad28 = 0; uint64_t mask = 0;
		for (int i = 0; i < 32; i++) {
			bad24 += vd_[0][i] != fX[i];
			if (vd_[1][i] != fX[32 + i]) { bad28++; mask |= 1ULL << i; }
		}
		printf("PROBEV%d: v24 %d of 32 differ, v28 %d of 32 differ, v28 elements %08lx\n", v, bad24, bad28, (unsigned long)mask);
		if (mask) { // what was written: compare with this block's tile values, raw and times fX
			asm volatile("fence" : : : "memory");
			static uint32_t tv[32][32] __attribute__((aligned(64)));
			for (unsigned long r = 0; r < 32; r++) { VMV_VR(v12, r, m1); asm volatile("vse32.v v12, (%0)" : : "r"(tv[r]) : "memory"); }
			asm volatile("fence" : : : "memory"); (void)*(volatile uint32_t *)&tv[31][31];
			for (int i = 0, n = 0; i < 32 && n < 4; i++) if (mask >> i & 1) {
				union { uint32_t u; float f; } g = {.u = vd_[1][i]};
				int hit = -1, kind = 0;
				for (int r = 0; r < 32 && hit < 0; r++) for (int j = 0; j < 32 && hit < 0; j++) {
					union { uint32_t u; float f; } t = {.u = tv[r][j]}, x = {.u = fX[32 + j]}, w = {.u = fW[r]};
					if (t.u == g.u) { hit = r * 32 + j; kind = 1; }
					else if (t.f * x.f == g.f) { hit = r * 32 + j; kind = 2; }
					else if (t.f * x.f * w.f == g.f) { hit = r * 32 + j; kind = 3; }
				}
				printf("  v28[%d] = %08x: %s row %d element %d\n", i, g.u, kind == 1 ? "raw tile value" : kind == 2 ? "tile * fX" : kind == 3 ? "tile * fX * fW" : "no match", hit / 32, hit % 32);
				n++;
			}
		}
		if (v == 5)
			for (int r = 0; r < 32; r++) {
				int bad = 0;
				for (int i = 0; i < 32; i++) bad += rowd[r][i] != fX[32 + i];
				if (bad) { printf("PROBEV5: first changed after row %d (%d words)\n", r, bad); break; }
			}
	}
}
#endif
#ifdef CHECK
// -DCHECK (with -DSMALL): every intermediate of the hw pass against the sw pass
static uint16_t cQo[H * T], cKo[KV * T], cVo[KV * T], cOo[H * T], cHh[H * T], cG[F * T], cU[F * T], cAc[F * T], cDo[H * T];
static void snap(uint16_t *d, const uint16_t *s, int n) { for (int i = 0; i < n; i++) d[i] = s[i]; }
static int fin(uint16_t v) { return (v & 0x7F80) != 0x7F80; }
static void cmp(const char *name, const uint16_t *h, const uint16_t *w, int n) {
	int same = 0, nfh = 0, nfw = 0, big = 0, shown = 0;
	for (int i = 0; i < n; i++) {
		same += h[i] == w[i]; nfh += !fin(h[i]); nfw += !fin(w[i]);
		float a = bf(h[i]), b = bf(w[i]), d = a > b ? a - b : b - a, m = (a > 0 ? a : -a) + (b > 0 ? b : -b);
		if (fin(h[i]) && fin(w[i]) && d > 0.02f * m + 1e-6f) big++;
		if ((!fin(h[i]) || !fin(w[i]) || d > 0.02f * m + 1e-6f) && shown++ < 3)
			printf("  %s[%d][%d] hw %04x sw %04x\n", name, i / T, i % T, h[i], w[i]);
	}
	printf("CHECK %-4s %d of %d identical, %d differ by > 2%%, non-finite hw %d sw %d\n", name, same, n, big, nfh, nfw);
}
#endif

#ifdef GEMMONLY
// -DGEMMONLY: block-size sensitivity without the whole layer. For the two GEMM
// shapes of the layer, K = H (q, k, v, o, gate, up; measured with gate's
// N = F stride) and K = F (down, N = H), NT tile columns of 64 output features
// at the full K and the real stride, hw and sw, and both quantizations of the
// GEMM's input. As in the layer, the weights and their scales come from memory
// no earlier run touched (cold), the quantized activations are warm. The timed
// tile columns are never written, so their weights are zeros; the OPU's and
// the vector unit's timing does not depend on values. Correctness: two tile
// columns of generated weights, hw against sw.
#ifndef NT
#define NT 2
#endif
// generated region: 256 bytes of each weight row (tile columns 0..3), 256 scales per block
static void gen_cols(uint8_t *W, uint8_t *sW, uint32_t *fW, int N, int K, uint32_t seed) {
	rng_init(seed);
	for (int k = 0; k < K; k++) {
		rng_step();
		vsetl(256, 0xC3UL);
		asm volatile("vand.vx v24, v16, %0\n\tvsrl.vi v16, v16, 3\n\tvand.vi v16, v16, 7\n\t"
		             "vadd.vi v16, v16, 4\n\tvsll.vi v16, v16, 3\n\tvor.vv v24, v24, v16\n\tvse8.v v24, (%1)"
		             : : "r"(0x87UL), "r"(W + (unsigned long)k * N) : "memory");
	}
	for (int b = 0; b < K / BS; b++)
		for (int i = 0; i < 256; i += 64) {
			rng_step();
			vsetl(64, 0xC1UL);
			asm volatile("vand.vi v24, v16, 3\n\tvadd.vx v24, v24, %0\n\tvse8.v v24, (%1)" : : "r"(118UL), "r"(sW + b * N + i) : "memory");
			vsetl(64, 0xD3UL);
			asm volatile("vzext.vf4 v0, v24\n\tvadd.vi v0, v0, -1\n\tvsll.vi v0, v0, 23\n\tvse32.v v0, (%0)" : : "r"(fW + b * N + i) : "memory");
		}
}
static uint16_t chk_hw[128 * T] A64;
static int same_bf16(const uint16_t *a, const uint16_t *b, int n, int *big) {
	int same = 0; *big = 0;
	for (int i = 0; i < n; i++) {
		float x = bf(a[i]), y = bf(b[i]), d = x > y ? x - y : y - x, m = (x > 0 ? x : -x) + (y > 0 ? y : -y);
		same += a[i] == b[i]; *big += !(d <= 0.02f * m + 1e-6f);
	}
	return same;
}
static void gemm_only(const char *name, const uint16_t *Y, const uint8_t *W, const uint8_t *sW, const uint32_t *fW,
                      int N, int K, uint16_t *out) {
	unsigned long t;
	quant_hw(Y, K, Xq);
	gemm_hw(W, sW, N, K, out, 0, 128);
	asm volatile("fence" : : : "memory"); (void)*(volatile uint16_t *)&out[128 * T - 1];
	for (int i = 0; i < 128 * T; i++) chk_hw[i] = out[i];
	quant_sw(Y, K, Xq);
	gemm_sw(W, fW, N, K, out, 0, 128);
	asm volatile("fence" : : : "memory"); (void)*(volatile uint16_t *)&out[128 * T - 1];
	int big, same = same_bf16(chk_hw, out, 128 * T, &big);
	printf("CHECK %s BS %d: %d of %d outputs identical hw vs sw, %d differ by > 2%%\n", name, BS, same, 128 * T, big);
	t = tick(); quant_hw(Y, K, Xq); unsigned long qhw = tock(t, &Xq[K * T - 1]);
	t = tick(); gemm_hw(W, sW, N, K, out, 256, 256 + 64 * NT); unsigned long ghw = tock(t, &out[(256 + 64 * NT) * T - 1]);
	t = tick(); quant_sw(Y, K, Xq); unsigned long qsw = tock(t, &Xq[K * T - 1]);
	t = tick(); gemm_sw(W, fW, N, K, out, 256 + 64 * NT, 256 + 128 * NT); unsigned long gsw = tock(t, &out[(256 + 128 * NT) * T - 1]);
	printf("RESULT %s BS %d K %d N %d | quant hw %lu sw %lu | gemm per tile hw %lu sw %lu\n",
	       name, BS, K, N, qhw, qsw, ghw / NT, gsw / NT);
}
#endif

int main(void) {
#ifdef GEMMONLY
	gen_cols(Wg, sWg, fWg, F, H, 5); gen_cols(Wd, sWd, fWd, H, F, 7);
	gen_bf16(X, H * T, 21); gen_bf16(Ac, F * T, 22);
	printf("data ready\n");
	gemm_only("gate", X, Wg, sWg, fWg, F, H, G);
	gemm_only("down", Ac, Wd, sWd, fWd, H, F, Do);
	return 0;
#endif
	gen_fp8(Wq, H * H, 1); gen_fp8(Wk, H * KV, 2); gen_fp8(Wv, H * KV, 3); gen_fp8(Wo, H * H, 4);
	gen_fp8(Wg, H * F, 5); gen_fp8(Wu, H * F, 6); gen_fp8(Wd, F * H, 7);
	gen_scales(sWq, fWq, H / BS * H, 11); gen_scales(sWk, fWk, H / BS * KV, 12); gen_scales(sWv, fWv, H / BS * KV, 13);
	gen_scales(sWo, fWo, H / BS * H, 14); gen_scales(sWg, fWg, H / BS * F, 15); gen_scales(sWu, fWu, H / BS * F, 16);
	gen_scales(sWd, fWd, F / BS * H, 17);
	gen_bf16(X, H * T, 21);
	printf("data ready\n");
#ifdef PROBE
	probe();
	return 0;
#endif
#ifdef PROBESW
	probe_sw();
	return 0;
#endif
#ifdef PROBEK
	probe_k();
	return 0;
#endif
#ifdef PROBE3
	probe3();
	return 0;
#endif
#ifdef PROBED
	probe_d();
	return 0;
#endif
#ifdef PROBEV
	probe_v();
	return 0;
#endif

	layer(1, Out_hw); unsigned long thw = report("hw");
#ifdef CHECK
	snap(cQo, Qo, H * T); snap(cKo, Ko, KV * T); snap(cVo, Vo, KV * T); snap(cOo, Oo, H * T); snap(cHh, Hh, H * T);
	snap(cG, G, F * T); snap(cU, U, F * T); snap(cAc, Ac, F * T); snap(cDo, Do, H * T);
#endif
	layer(0, Out_sw); unsigned long tsw = report("sw");
#ifdef CHECK
	cmp("q", cQo, Qo, H * T); cmp("k", cKo, Ko, KV * T); cmp("v", cVo, Vo, KV * T); cmp("o", cOo, Oo, H * T);
	cmp("h", cHh, Hh, H * T); cmp("gate", cG, G, F * T); cmp("up", cU, U, F * T); cmp("act", cAc, Ac, F * T);
	cmp("down", cDo, Do, H * T); cmp("out", Out_hw, Out_sw, H * T);
	return 0;
#endif
	layer(1, Out_hw); report("hw-again");

	// 2 * (q + k + v + o + gate + up + down) multiply-adds per token
	unsigned long macs = (unsigned long)T * (2UL * H * H + 2UL * H * KV + 3UL * H * F);
	printf("macs %lu  hw %lu.%02lu MAC/cycle  sw %lu.%02lu MAC/cycle  speedup %lu.%02lu\n", macs,
	       macs / thw, macs * 100 / thw % 100, macs / tsw, macs * 100 / tsw % 100, tsw / thw, tsw * 100 / thw % 100);

	int same = 0; float maxd = 0, maxa = 0;
	for (int i = 0; i < H * T; i++) {
		float a = bf(Out_hw[i]), b = bf(Out_sw[i]), d = a > b ? a - b : b - a;
		same += Out_hw[i] == Out_sw[i];
		if (d > maxd) maxd = d;
		if ((a > 0 ? a : -a) > maxa) maxa = a > 0 ? a : -a;
	}
	printf("outputs: %d of %d identical, max |hw - sw| = %d e-6 of max |out| = %d e-3\n",
	       same, H * T, (int)(maxd * 1e6f), (int)(maxa * 1e3f));
	return 0;
}
