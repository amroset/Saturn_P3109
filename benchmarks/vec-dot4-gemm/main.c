// vfdot4 (docs/vfdot4.md) in an MX-style GEMV and GEMM: does pairing cost
// anything against explicit scale operands in a dot-product kernel?
//
//   y[t][r] += sum_k W[r][k] x[t][k]     R = 64 rows, K = 256, T = 1 (GEMV) or 4 (GEMM)
//
// W and x are binary8p4 with one scale per row (per token) per 32 k. W is
// packed chunk-major: lane l of Wp[g][c] holds W[16g + l][4c .. 4c+3], so one
// vle32 at e32 m2 loads one chunk of 16 rows. x's four codes of a chunk and
// its block scale go into one integer register (vfdot4.vx). LMUL 2: 16 rows per
// instruction; the tokens' accumulators are v0, v2, v4, v6.
//
// Variants:
//   single    one W register (v8, scales v10 by pairing), reloaded per chunk
//   unroll    four W registers (v8, v12, v16, v20), a block's chunks in turn;
//             pairing: the block's scales loaded into v10 and copied to v14,
//             v18, v22; -DSHARED (P3109BlockDotShared, the explicit-operand
//             emulation): loaded once into v30, no copies
//   unscaled  unroll without scales (scaled mode off): the bound
// Scaled variants must give the same checksum (same rounding order per lane).
// Build at -DPAD=0..60 with -fno-toplevel-reorder; -DREPS=4.
#pragma GCC optimize("no-tree-vectorize")
#include <stdio.h>
#include <stdint.h>
#include <stddef.h>
#include "util.h"

#ifndef PAD
#define PAD 0
#endif
#ifndef REPS
#define REPS 1
#endif
#define PSTR_(x) #x
#define PSTR(x) PSTR_(x)
asm(".text\n.skip " PSTR(PAD) "\n.section .text.startup,\"ax\",@progbits\n.skip " PSTR(PAD) "\n.text");

#define R 64
#define K 256
#define T 4
#define LANES 16          // e32 m2
#define G (R / LANES)     // row groups
#define CH (K / 4)        // chunks
#define KB (K / 32)       // blocks

static uint32_t Wp[G][CH][LANES] __attribute__((aligned(64)));
static uint8_t  sW[G][KB][32] __attribute__((aligned(64)));  // bytes 0..15: the group's rows
static uint32_t xw[T][CH] __attribute__((aligned(64)));
static uint8_t  sx[T][KB];
static uint32_t y[T][R] __attribute__((aligned(64)));
static volatile uint8_t sink;

static uint32_t seed = 1;
static uint32_t rnd(void) { seed = seed * 1664525u + 1013904223u; return seed >> 8; }
static uint8_t rnd_fp8(void) { uint8_t c; do c = rnd(); while (c == 0x80 || (c & 0x7F) == 0x7F); return c; }
static uint32_t checksum(const void *p, size_t n) {
	const uint8_t *b = p; uint32_t h = 2166136261u;
	for (size_t i = 0; i < n; i++) h = (h ^ b[i]) * 16777619u;
	return h;
}

// e32 m2, ta ma, binary8p4, scaled (bit 11), E = 1; e8 m1 for the scale register
#define VT_DOT(sc) (0xD1UL | ((unsigned long)(sc) << 11))
#define VT8 0xC0UL
static inline void vsetl(unsigned long avl, unsigned long vt) { asm volatile("vsetvl x0, %0, %1" : : "r"(avl), "r"(vt)); }
// vfdot4.vx v<vd>, v<vs2>, x
#define DOT(vd, vs2, x) asm volatile(".insn r 0x57, 0x5, 0x6B, x" #vd ", %0, x" #vs2 : : "r"(x))
#define LDW(v, p) asm volatile("vle32.v " v ", (%0)" : : "r"(p) : "memory")
#define ZERO(v) asm volatile("vmv.v.i " v ", 0")
#define STY(v, p) asm volatile("vse32.v " v ", (%0)" : : "r"(p) : "memory")
#ifdef SHARED
#define SC0 "v30"
#else
#define SC0 "v10"
#endif

// the tokens: X(t, vd) for each accumulator
#define TOK1(X) X(0, 0)
#define TOK4(X) X(0, 0) X(1, 2) X(2, 4) X(3, 6)
#define ZACC(t, vd) ZERO("v" #vd);
#define SACC(t, vd) STY("v" #vd, &y[t][LANES * g]);
#define SCALE(t, vd) uint64_t s##t = (uint64_t)sx[t][b] << 32;

// one block's scales into the W registers' scale registers
static inline void scales(int g, int b, int copies) {
	vsetl(32, VT8);
	asm volatile("vle8.v " SC0 ", (%0)" : : "r"(sW[g][b]) : "memory");
	if (copies) asm volatile("vmv.v.v v14, v10\n\tvmv.v.v v18, v10\n\tvmv.v.v v22, v10");
}

#define KERNELS(NT, TOKS) \
static void single_##NT(void) { \
	for (int g = 0; g < G; g++) { \
		vsetl(LANES, VT_DOT(1)); TOKS(ZACC) \
		for (int b = 0; b < KB; b++) { \
			scales(g, b, 0); vsetl(LANES, VT_DOT(1)); \
			TOKS(SCALE) \
			for (int c = 8 * b; c < 8 * b + 8; c++) { \
				LDW("v8", Wp[g][c]); \
				TOKS(DOT8) \
			} \
		} \
		TOKS(SACC) \
	} \
} \
static void unroll_##NT(int sc) { \
	for (int g = 0; g < G; g++) { \
		vsetl(LANES, VT_DOT(sc)); TOKS(ZACC) \
		for (int b = 0; b < KB; b++) { \
			uint64_t s0 = 0, s1 = 0, s2 = 0, s3 = 0; (void)s1; (void)s2; (void)s3; \
			if (sc) { scales(g, b, COPIES); vsetl(LANES, VT_DOT(1)); TOKS(SCALE_SET) } \
			for (int c = 8 * b; c < 8 * b + 8; c += 4) { \
				LDW("v8", Wp[g][c]); LDW("v12", Wp[g][c + 1]); LDW("v16", Wp[g][c + 2]); LDW("v20", Wp[g][c + 3]); \
				TOKS(DOTU8) TOKS(DOTU12) TOKS(DOTU16) TOKS(DOTU20) \
			} \
		} \
		TOKS(SACC) \
	} \
}
#define DOT8(t, vd)   DOT(vd, 8, xw[t][c] | s##t);
#define DOTU8(t, vd)  DOT(vd, 8, xw[t][c] | s##t);
#define DOTU12(t, vd) DOT(vd, 12, xw[t][c + 1] | s##t);
#define DOTU16(t, vd) DOT(vd, 16, xw[t][c + 2] | s##t);
#define DOTU20(t, vd) DOT(vd, 20, xw[t][c + 3] | s##t);
#define SCALE_SET(t, vd) s##t = (uint64_t)sx[t][b] << 32;
#ifdef SHARED
#define COPIES 0
#else
#define COPIES 1
#endif
KERNELS(1, TOK1)
KERNELS(4, TOK4)

// ---- 8 tokens (follow-up): more instructions per W chunk, closer to compute-bound ----
// x is packed once, outside the timed region (codes | scale << 32, xp8), so each vfdot4.vx
// needs one ld.
//   gemm8m1  LMUL 1, 8 rows per instruction: accumulators v0..v7; W v8, v10, v12, v14,
//            pairing scales v9 (copied to v11, v13, v15); -DSHARED v30
//   gemm8m2  LMUL 2: accumulators v0..v14 (even); W v16, v20, v24, v28, pairing scales
//            v18 (copied to v22, v26, v30): all 32 registers; -DSHARED v30
#define T8 8
#define H (R / 8)          // 8-row groups (LMUL 1)
static uint32_t xw8[T8][CH] __attribute__((aligned(64)));
static uint8_t  sx8[T8][KB];
static uint64_t xp8[T8][CH] __attribute__((aligned(64)));
static uint8_t  sWh[H][KB][32] __attribute__((aligned(64)));  // bytes 0..7: the 8-row group's rows
static uint32_t y8[T8][R] __attribute__((aligned(64)));
#define VT_DOT1(sc) (0xD0UL | ((unsigned long)(sc) << 11))     // e32 m1
#define TOK8M1(X) X(0, 0) X(1, 1) X(2, 2) X(3, 3) X(4, 4) X(5, 5) X(6, 6) X(7, 7)
#define TOK8M2(X) X(0, 0) X(1, 2) X(2, 4) X(3, 6) X(4, 8) X(5, 10) X(6, 12) X(7, 14)
#define ZACC8(t, vd) ZERO("v" #vd);
static void pack8(int sc) {
	for (int t = 0; t < T8; t++)
		for (int c = 0; c < CH; c++)
			xp8[t][c] = xw8[t][c] | (sc ? (uint64_t)sx8[t][c / 8] << 32 : 0);
}
#define DOTP(t, vd, vs2, cc) DOT(vd, vs2, xp8[t][cc]);
#define M1_W8(t, vd)  DOTP(t, vd, 8, c)
#define M1_W10(t, vd) DOTP(t, vd, 10, c + 1)
#define M1_W12(t, vd) DOTP(t, vd, 12, c + 2)
#define M1_W14(t, vd) DOTP(t, vd, 14, c + 3)
#define M2_W16(t, vd) DOTP(t, vd, 16, c)
#define M2_W20(t, vd) DOTP(t, vd, 20, c + 1)
#define M2_W24(t, vd) DOTP(t, vd, 24, c + 2)
#define M2_W28(t, vd) DOTP(t, vd, 28, c + 3)
#define ST8M1(t, vd) STY("v" #vd, &y8[t][8 * h]);
#define ST8M2(t, vd) STY("v" #vd, &y8[t][LANES * g]);

static void gemm8m1(int sc) {
	for (int h = 0; h < H; h++) {
		const uint32_t (*w)[LANES] = Wp[h / 2];
		int off = 8 * (h % 2);
		vsetl(8, VT_DOT1(sc)); TOK8M1(ZACC8)
		for (int b = 0; b < KB; b++) {
			if (sc) {
				vsetl(32, VT8);
#ifdef SHARED
				asm volatile("vle8.v v30, (%0)" : : "r"(sWh[h][b]) : "memory");
#else
				asm volatile("vle8.v v9, (%0)\n\tvmv.v.v v11, v9\n\tvmv.v.v v13, v9\n\tvmv.v.v v15, v9" : : "r"(sWh[h][b]) : "memory");
#endif
				vsetl(8, VT_DOT1(1));
			}
			for (int c = 8 * b; c < 8 * b + 8; c += 4) {
				LDW("v8", &w[c][off]); LDW("v10", &w[c + 1][off]); LDW("v12", &w[c + 2][off]); LDW("v14", &w[c + 3][off]);
				TOK8M1(M1_W8) TOK8M1(M1_W10) TOK8M1(M1_W12) TOK8M1(M1_W14)
			}
		}
		TOK8M1(ST8M1)
	}
}

static void gemm8m2(int sc) {
	for (int g = 0; g < G; g++) {
		vsetl(LANES, VT_DOT(sc)); TOK8M2(ZACC8)
		for (int b = 0; b < KB; b++) {
			if (sc) {
				vsetl(32, VT8);
#ifdef SHARED
				asm volatile("vle8.v v30, (%0)" : : "r"(sW[g][b]) : "memory");
#else
				asm volatile("vle8.v v18, (%0)\n\tvmv.v.v v22, v18\n\tvmv.v.v v26, v18\n\tvmv.v.v v30, v18" : : "r"(sW[g][b]) : "memory");
#endif
				vsetl(LANES, VT_DOT(1));
			}
			for (int c = 8 * b; c < 8 * b + 8; c += 4) {
				LDW("v16", Wp[g][c]); LDW("v20", Wp[g][c + 1]); LDW("v24", Wp[g][c + 2]); LDW("v28", Wp[g][c + 3]);
				TOK8M2(M2_W16) TOK8M2(M2_W20) TOK8M2(M2_W24) TOK8M2(M2_W28)
			}
		}
		TOK8M2(ST8M2)
	}
}

static unsigned long t0;
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

#ifdef SHARED
#define SCHEME "shared"
#else
#define SCHEME "ruleA"
#endif
#define TIME(kernel, label, nt, call) do { \
		call; start(&y[(nt) - 1][R - 1]); for (int r_ = 0; r_ < REPS; r_++) { call; } \
		unsigned long c_ = stop(&y[(nt) - 1][R - 1]) / REPS; \
		unsigned long macs = (unsigned long)R * K * (nt); \
		printf("RESULT %-7s %-6s %-9s %7lu cycles  %lu.%03lu MAC/cycle  checksum %08lx\n", kernel, SCHEME, label, c_, \
		       macs / c_, macs * 1000 / c_ % 1000, (unsigned long)checksum(y, sizeof(uint32_t) * R * (nt))); \
	} while (0)
#define TIME8(kernel, label, call) do { \
		call; start(&y8[T8 - 1][R - 1]); for (int r_ = 0; r_ < REPS; r_++) { call; } \
		unsigned long c_ = stop(&y8[T8 - 1][R - 1]) / REPS; \
		unsigned long macs = (unsigned long)R * K * T8; \
		printf("RESULT %-7s %-6s %-9s %7lu cycles  %lu.%03lu MAC/cycle  checksum %08lx\n", kernel, SCHEME, label, c_, \
		       macs / c_, macs * 1000 / c_ % 1000, (unsigned long)checksum(y8, sizeof y8)); \
	} while (0)

int main(void) {
	for (int g = 0; g < G; g++)
		for (int c = 0; c < CH; c++)
			for (int l = 0; l < LANES; l++)
				Wp[g][c][l] = rnd_fp8() | rnd_fp8() << 8 | rnd_fp8() << 16 | (uint32_t)rnd_fp8() << 24;
	for (int g = 0; g < G; g++)
		for (int b = 0; b < KB; b++)
			for (int i = 0; i < 32; i++) sW[g][b][i] = 128 + rnd() % 9 - 4;
	for (int t = 0; t < T; t++) {
		for (int c = 0; c < CH; c++) xw[t][c] = rnd_fp8() | rnd_fp8() << 8 | rnd_fp8() << 16 | (uint32_t)rnd_fp8() << 24;
		for (int b = 0; b < KB; b++) sx[t][b] = 128 + rnd() % 9 - 4;
	}
	for (int t = 0; t < T8; t++) {
		for (int c = 0; c < CH; c++) xw8[t][c] = rnd_fp8() | rnd_fp8() << 8 | rnd_fp8() << 16 | (uint32_t)rnd_fp8() << 24;
		for (int b = 0; b < KB; b++) sx8[t][b] = 128 + rnd() % 9 - 4;
	}
	// the 8-row groups' scales are the 16-row groups' (gemm8m1 and gemm8m2 compute the same)
	for (int h = 0; h < H; h++)
		for (int b = 0; b < KB; b++)
			for (int i = 0; i < 32; i++) sWh[h][b][i] = i < 8 ? sW[h / 2][b][8 * (h % 2) + i] : 128 + rnd() % 9 - 4;
	asm volatile("csrw frm, zero");
	TIME("gemv", "single", 1, single_1());
	TIME("gemv", "unroll", 1, unroll_1(1));
	TIME("gemv", "unscaled", 1, unroll_1(0));
	TIME("gemm", "single", 4, single_4());
	TIME("gemm", "unroll", 4, unroll_4(1));
	TIME("gemm", "unscaled", 4, unroll_4(0));
	pack8(1);
	TIME8("gemm8m1", "unroll", gemm8m1(1));
	TIME8("gemm8m2", "unroll", gemm8m2(1));
	pack8(0);
	TIME8("gemm8m1", "unscaled", gemm8m1(0));
	TIME8("gemm8m2", "unscaled", gemm8m2(0));
	return 0;
}
