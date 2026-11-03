// Scaled mode at LMUL > 1 (docs/scaled-mode.md), on P3109BlockV256D128ShuttleConfig:
// a block operand's data group vN..vN+L-1 has its scales in the group after it,
// from vN+L, element i taking byte i >> log2(E). Along the blocks, one
// instruction then covers L blocks of 32 (or 2L of 16), and one scale load
// serves them all. Each kernel runs at LMUL 1, 2, 4, scaled and unscaled; the
// scaled outputs must be identical at every LMUL (LMUL 1 is verified by
// vec-block against the gfloat reference), so their checksums must match.
//
//   dequant   binary8p4 -> BF16, E = 32 (MX)       quant    BF16 -> binary8p4, E = 32
//   dequant16 binary8p4 -> BF16, E = 16 (NVFP4)    quant16  BF16 -> binary8p4, E = 16
//   smul      ScaledMultiply binary8p4 x binary8p4 -> BF16, E = 32, two blocks
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

#define NE 2048
#define VL1 32
static uint8_t  x8[NE] __attribute__((aligned(64))), w8[NE] __attribute__((aligned(64)));
static uint16_t xq[NE] __attribute__((aligned(64)));
static uint8_t  sc[NE / 32 + 64] __attribute__((aligned(64))), sw[NE / 32 + 64] __attribute__((aligned(64)));
static uint8_t  sc16[NE / 16 + 64] __attribute__((aligned(64)));
static uint16_t y16[NE] __attribute__((aligned(64)));
static uint8_t  q8[NE] __attribute__((aligned(64)));
static volatile uint8_t sink;

static uint32_t seed = 777;
static uint32_t rnd(void) { seed = seed * 1664525u + 1013904223u; return seed >> 8; }
static uint8_t rnd_fp8(void) { uint8_t c; do c = rnd(); while (c == 0x80 || (c & 0x7F) == 0x7F); return c; }
static uint16_t rnd_bf16(void) { unsigned e = 127 - 4 + rnd() % 17; return ((rnd() & 1) << 15) | (e << 7) | (rnd() & 0x7F); }
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
static inline void vset(unsigned long avl, unsigned long vt) { asm volatile("vsetvl x0, %0, %1" : : "r"(avl), "r"(vt)); }
// e8 ta ma, LMUL 2^lm, optionally scaled with 2^lg elements per scale byte
#define VT(lm, sc, lg) (0xC0UL | (lm) | ((unsigned long)(sc) << 11) | ((unsigned long)(lg) << 12))
#define WCVT(d, s) ".insn r 0x57, 0x1, 0x25, " d ", x13, " s "\n\t"
#define NCVT(d, s) ".insn r 0x57, 0x1, 0x25, " d ", x29, " s "\n\t"

// -DSHARED (P3109BlockSharedV256D128ShuttleConfig): scales from v31 (vs1),
// v30 (vs2), v29 (vd) instead, as with explicit scale operands
#ifdef SHARED
#define SC1(r) "v31"
#define SC2(r) "v30"
#define SCD(r) "v29"
#else
#define SC1(r) r
#define SC2(r) r
#define SCD(r) r
#endif

// Scalar-operand estimate (mode 2): the unscaled loop plus, per instruction and
// block operand, one scalar load of that instruction's scale bytes (L of them at
// E = 32, 2L at E = 16), as a packed scalar scale operand would need.
// The sum is kept in a local (a register), so the estimate adds no stores.
static volatile unsigned long sacc;
static inline unsigned long sload(const uint8_t *p, int nbytes) {
	switch (nbytes) {
	case 1: return *(volatile uint8_t *)p;
	case 2: return *(volatile uint16_t *)p;
	case 4: return *(volatile uint32_t *)p;
	default: return *(volatile uint64_t *)p;
	}
}

// One kernel at one LMUL. L = 2^LM; registers chosen by rule A for that LMUL:
// data D, its scales S = D + L; for smul a second operand D2, S2 = D2 + L.
#define DEQUANT(name, LM, LG, D, S, Y, DX, YX, SCARR, SCSTEP) \
	static void name(int scaled) { \
		const int L = 1 << LM, n = VL1 * L; \
		vset(n, VT(LM, scaled == 1, LG)); \
		unsigned long acc_ = 0; \
		for (int j = 0; j < NE; j += n) { \
			asm volatile("vle8.v " D ", (%0)" : : "r"(&x8[j]) : "memory"); \
			if (scaled == 1) asm volatile("vl1re8.v " SC2(S) ", (%0)" : : "r"(&SCARR[j / SCSTEP]) : "memory"); \
			if (scaled == 2) acc_ += sload(&SCARR[j / SCSTEP], n / SCSTEP); \
			asm volatile(".insn r 0x57, 0x1, 0x25, " YX ", x13, " DX); \
			asm volatile("vse16.v " Y ", (%0)" : : "r"(&y16[j]) : "memory"); \
		} \
		sacc = acc_; \
	}
#define QUANT(name, LM, LG, X, Q, S, XX, QX, SCARR, SCSTEP) \
	static void name(int scaled) { \
		const int L = 1 << LM, n = VL1 * L; \
		vset(n, VT(LM, scaled == 1, LG)); \
		unsigned long acc_ = 0; \
		for (int j = 0; j < NE; j += n) { \
			asm volatile("vle16.v " X ", (%0)" : : "r"(&xq[j]) : "memory"); \
			if (scaled == 1) asm volatile("vl1re8.v " SCD(S) ", (%0)" : : "r"(&SCARR[j / SCSTEP]) : "memory"); \
			if (scaled == 2) acc_ += sload(&SCARR[j / SCSTEP], n / SCSTEP); \
			asm volatile(".insn r 0x57, 0x1, 0x25, " QX ", x29, " XX); \
			asm volatile("vse8.v " Q ", (%0)" : : "r"(&q8[j]) : "memory"); \
		} \
		sacc = acc_; \
	}
#define SMUL(name, LM, D, S, D2, S2, Y) \
	static void name(int scaled) { \
		const int L = 1 << LM, n = VL1 * L; \
		vset(n, VT(LM, scaled == 1, 5)); \
		unsigned long acc_ = 0; \
		for (int j = 0; j < NE; j += n) { \
			asm volatile("vle8.v " D ", (%0)" : : "r"(&x8[j]) : "memory"); \
			asm volatile("vle8.v " D2 ", (%0)" : : "r"(&w8[j]) : "memory"); \
			if (scaled == 2) acc_ += sload(&sc[j / 32], L) + sload(&sw[j / 32], L); \
			if (scaled == 1) { \
				asm volatile("vl1re8.v " SC2(S) ", (%0)" : : "r"(&sc[j / 32]) : "memory"); \
				asm volatile("vl1re8.v " SC1(S2) ", (%0)" : : "r"(&sw[j / 32]) : "memory"); \
			} \
			asm volatile("vfwmul.vv " Y ", " D ", " D2); \
			asm volatile("vse16.v " Y ", (%0)" : : "r"(&y16[j]) : "memory"); \
		} \
		sacc = acc_; \
	}

// LMUL 1: data v2 (scales v3); LMUL 2: v2-v3 (v4); LMUL 4: v4-v7 (v8)
DEQUANT(dequant_1, 0, 5, "v2", "v3", "v8", "x2", "x8", sc, 32)
DEQUANT(dequant_2, 1, 5, "v2", "v4", "v8", "x2", "x8", sc, 32)
DEQUANT(dequant_4, 2, 5, "v4", "v8", "v16", "x4", "x16", sc, 32)
DEQUANT(dequant16_1, 0, 4, "v2", "v3", "v8", "x2", "x8", sc16, 16)
DEQUANT(dequant16_2, 1, 4, "v2", "v4", "v8", "x2", "x8", sc16, 16)
DEQUANT(dequant16_4, 2, 4, "v4", "v8", "v16", "x4", "x16", sc16, 16)
// quant: source BF16 at EMUL 2L, result vd at LMUL L, its scales after vd
QUANT(quant_1, 0, 5, "v16", "v8", "v9", "x16", "x8", sc, 32)
QUANT(quant_2, 1, 5, "v16", "v8", "v10", "x16", "x8", sc, 32)
QUANT(quant_4, 2, 5, "v16", "v4", "v8", "x16", "x4", sc, 32)
QUANT(quant16_1, 0, 4, "v16", "v8", "v9", "x16", "x8", sc16, 16)
QUANT(quant16_2, 1, 4, "v16", "v8", "v10", "x16", "x8", sc16, 16)
QUANT(quant16_4, 2, 4, "v16", "v4", "v8", "x16", "x4", sc16, 16)
SMUL(smul_1, 0, "v2", "v3", "v4", "v5", "v8")
SMUL(smul_2, 1, "v2", "v4", "v6", "v8", "v12")
#ifdef SHARED // the result group v24-v31 would hold the shared scales
SMUL(smul_4, 2, "v4", "v8", "v12", "v16", "v16")
#else
SMUL(smul_4, 2, "v4", "v8", "v12", "v16", "v24")
#endif

static void run(const char *kernel, int lmul, void (*f)(int), const volatile void *last, const void *out, size_t n) {
	unsigned long c[3] = {0, 0, 0}, ck = 0;
#ifdef SHARED
	const int modes = 2;
#else
	const int modes = 3;
#endif
	for (int s = 0; s < modes; s++) {
		f(s); start(last); for (int r = 0; r < REPS; r++) f(s); c[s] = stop(last) / REPS;
		if (s == 1) ck = checksum(out, n);
	}
	printf("RESULT %-9s LMUL %d  scaled %6lu  unscaled %6lu  scalar %6lu cycles  ratio %lu.%02lu  checksum %08lx\n", kernel, lmul,
	       c[1], c[0], c[2], c[1] / c[0], c[1] * 100 / c[0] % 100, ck);
}

int main(void) {
	for (int i = 0; i < NE; i++) { x8[i] = rnd_fp8(); w8[i] = rnd_fp8(); xq[i] = rnd_bf16(); }
	for (int j = 0; j < NE / 32 + 64; j++) { sc[j] = 128 + rnd() % 21 - 10; sw[j] = 128 + rnd() % 9 - 4; }
	for (int j = 0; j < NE / 16 + 64; j++) sc16[j] = 128 + rnd() % 21 - 10;
	run("dequant", 1, dequant_1, &y16[NE - 1], y16, sizeof y16);
	run("dequant", 2, dequant_2, &y16[NE - 1], y16, sizeof y16);
	run("dequant", 4, dequant_4, &y16[NE - 1], y16, sizeof y16);
	run("quant", 1, quant_1, &q8[NE - 1], q8, sizeof q8);
	run("quant", 2, quant_2, &q8[NE - 1], q8, sizeof q8);
	run("quant", 4, quant_4, &q8[NE - 1], q8, sizeof q8);
	run("dequant16", 1, dequant16_1, &y16[NE - 1], y16, sizeof y16);
	run("dequant16", 2, dequant16_2, &y16[NE - 1], y16, sizeof y16);
	run("dequant16", 4, dequant16_4, &y16[NE - 1], y16, sizeof y16);
	run("quant16", 1, quant16_1, &q8[NE - 1], q8, sizeof q8);
	run("quant16", 2, quant16_2, &q8[NE - 1], q8, sizeof q8);
	run("quant16", 4, quant16_4, &q8[NE - 1], q8, sizeof q8);
	run("smul", 1, smul_1, &y16[NE - 1], y16, sizeof y16);
	run("smul", 2, smul_2, &y16[NE - 1], y16, sizeof y16);
	run("smul", 4, smul_4, &y16[NE - 1], y16, sizeof y16);
	return 0;
}
