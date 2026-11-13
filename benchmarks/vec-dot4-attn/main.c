// vfdot4.vv (docs/vfdot4.md) in multi-head attention scores while decoding one token: does pairing
// cost anything against explicit scale operands when both operands are blocks?
//
//   S[j][h] = sum_d q[h][d] k[j][h][d]     NH = 32 heads, HD = 96, NK = 64 cached keys (Phi-3-mini)
//
// q and k are binary8p4 with one scale per head (and key) per 32 dimensions. Heads across the
// elements (e32 m1, 8 heads per instruction): lane l of Qp[g][c] holds q[8g + l][4c .. 4c+3], and
// of Kp[j][g][c] the same dimensions of key j. One key at a time, so the query is reloaded per key
// and both operands' scales change once per block (eight instructions).
//
// Variants:
//   single    one q register (v8, scales v9 by pairing) and one k register (v16, scales v17)
//   unroll    four of each, a block's chunks in turn: q v8, v10, v12, v14, k v16, v18, v20, v22,
//             into four accumulators v0..v3 (chunk c into v(c mod 4)), summed (v0+v1)+(v2+v3) at the
//             end; single's one accumulator makes every vfdot4 wait for the previous one; pairing: each block's scales loaded into v9 and v17 and copied to v11, v13, v15 and
//             v19, v21, v23; -DSHARED (P3109BlockDotShared, the explicit-operand emulation): loaded
//             once into v31 (vs1) and v30 (vs2), no copies
//   unscaled  unroll without scales (scaled mode off): the bound
// -DNOCOPY: pairing without its scale copies, timing only (stale scales, wrong checksum).
// single and unroll differ in rounding order, hence in checksum (ref.py gives both).
// Build at -DPAD=0..60 -DDOFF=n with -fno-toplevel-reorder -falign-labels=16; -DREPS=4.
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

#define NH 32             // heads
#define HD 96             // head dimension
#define NK 64             // cached keys
#define LANES 8           // e32 m1: heads per instruction
#define G (NH / LANES)    // head groups
#define CH (HD / 4)       // chunks of four dimensions
#define KB (HD / 32)      // blocks

// Every array is kept, in this order, in .bss.bench at a fixed address (see vec-dot4-gemm)
#ifndef DOFF
#define DOFF 0
#endif
#define BENCH __attribute__((used, section(".bss.bench"), aligned(64)))
static uint8_t dpad[DOFF + 1] BENCH;
static uint32_t Qp[G][CH][LANES] BENCH;
static uint8_t  sQ[G][KB][LANES] BENCH;
static uint32_t Kp[NK][G][CH][LANES] BENCH;
static uint8_t  sK[NK][G][KB][LANES] BENCH;
static uint32_t S[NK][NH] BENCH;
static volatile uint8_t sink;

static uint32_t seed = 1;
static uint32_t rnd(void) { seed = seed * 1664525u + 1013904223u; return seed >> 8; }
static uint8_t rnd_fp8(void) { uint8_t c; do c = rnd(); while (c == 0x80 || (c & 0x7F) == 0x7F); return c; }
// four codes, element j in bits 8j+7:8j, drawn in that order (models the data in Python: ref.py)
static uint32_t rnd4(void) { uint32_t w = 0; for (int j = 0; j < 4; j++) w |= (uint32_t)rnd_fp8() << 8 * j; return w; }
static uint32_t checksum(const void *p, size_t n) {
	const uint8_t *b = p; uint32_t h = 2166136261u;
	for (size_t i = 0; i < n; i++) h = (h ^ b[i]) * 16777619u;
	return h;
}

// e32 m1, ta ma, binary8p4, scaled (bit 11), E = 1; e8 m1 for the scale registers
#define VT_DOT1(sc) (0xD0UL | ((unsigned long)(sc) << 11))
#define VT8 0xC0UL
static inline void vsetl(unsigned long avl, unsigned long vt) { asm volatile("vsetvl x0, %0, %1" : : "r"(avl), "r"(vt)); }
// vfdot4.vv v<vd>, v<vs2>, v<vs1>
#define DOTVV(vd, vs2, vs1) asm volatile(".insn r 0x57, 0x1, 0x6B, x" #vd ", x" #vs1 ", x" #vs2)
#ifdef NOCOPY
#define COPY(s)
#else
#define COPY(s) asm volatile(s)
#endif
#define LDW(v, p) asm volatile("vle32.v " v ", (%0)" : : "r"(p) : "memory")
#define STS(v, p) asm volatile("vse32.v " v ", (%0)" : : "r"(p) : "memory")

// one block's scales of the query and of key j into the operands' scale registers
static inline void scales(int g, int j, int b, int copies) {
	vsetl(LANES, VT8);
#ifdef SHARED
	asm volatile("vle8.v v31, (%0)\n\tvle8.v v30, (%1)" : : "r"(sQ[g][b]), "r"(sK[j][g][b]) : "memory");
	(void)copies;
#else
	asm volatile("vle8.v v9, (%0)\n\tvle8.v v17, (%1)" : : "r"(sQ[g][b]), "r"(sK[j][g][b]) : "memory");
	if (copies) COPY("vmv.v.v v11, v9\n\tvmv.v.v v13, v9\n\tvmv.v.v v15, v9\n\t"
	                         "vmv.v.v v19, v17\n\tvmv.v.v v21, v17\n\tvmv.v.v v23, v17");
#endif
	vsetl(LANES, VT_DOT1(1));
}

static void single(void) {
	for (int g = 0; g < G; g++)
		for (int j = 0; j < NK; j++) {
			vsetl(LANES, VT_DOT1(1)); asm volatile("vmv.v.i v0, 0");
			for (int b = 0; b < KB; b++) {
				scales(g, j, b, 0);
				for (int c = 8 * b; c < 8 * b + 8; c++) {
					LDW("v8", Qp[g][c]); LDW("v16", Kp[j][g][c]);
					DOTVV(0, 16, 8);
				}
			}
			STS("v0", &S[j][LANES * g]);
		}
}

static void unroll(int sc) {
	for (int g = 0; g < G; g++)
		for (int j = 0; j < NK; j++) {
			vsetl(LANES, VT_DOT1(sc)); asm volatile("vmv.v.i v0, 0\n\tvmv.v.i v1, 0\n\tvmv.v.i v2, 0\n\tvmv.v.i v3, 0");
			for (int b = 0; b < KB; b++) {
				if (sc) scales(g, j, b, 1);
				for (int c = 8 * b; c < 8 * b + 8; c += 4) {
					LDW("v8", Qp[g][c]); LDW("v10", Qp[g][c + 1]); LDW("v12", Qp[g][c + 2]); LDW("v14", Qp[g][c + 3]);
					LDW("v16", Kp[j][g][c]); LDW("v18", Kp[j][g][c + 1]); LDW("v20", Kp[j][g][c + 2]); LDW("v22", Kp[j][g][c + 3]);
					DOTVV(0, 16, 8); DOTVV(1, 18, 10); DOTVV(2, 20, 12); DOTVV(3, 22, 14);
				}
			}
			vsetl(LANES, VT_DOT1(0));
			asm volatile("vfadd.vv v0, v0, v1\n\tvfadd.vv v2, v2, v3\n\tvfadd.vv v0, v0, v2");
			STS("v0", &S[j][LANES * g]);
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
#define TIME(label, call) do { \
		call; start(&S[NK - 1][NH - 1]); for (int r_ = 0; r_ < REPS; r_++) { call; } \
		unsigned long c_ = stop(&S[NK - 1][NH - 1]) / REPS; \
		unsigned long macs = (unsigned long)NK * NH * HD; \
		printf("RESULT attn    %-6s %-9s %7lu cycles  %lu.%03lu MAC/cycle  checksum %08lx\n", SCHEME, label, c_, \
		       macs / c_, macs * 1000 / c_ % 1000, (unsigned long)checksum(S, sizeof S)); \
	} while (0)

int main(void) {
	for (int g = 0; g < G; g++)
		for (int c = 0; c < CH; c++)
			for (int l = 0; l < LANES; l++)
				Qp[g][c][l] = rnd4();
	for (int g = 0; g < G; g++)
		for (int b = 0; b < KB; b++)
			for (int l = 0; l < LANES; l++) sQ[g][b][l] = 128 + rnd() % 9 - 4;
	for (int j = 0; j < NK; j++) {
		for (int g = 0; g < G; g++)
			for (int c = 0; c < CH; c++)
				for (int l = 0; l < LANES; l++)
					Kp[j][g][c][l] = rnd4();
		for (int g = 0; g < G; g++)
			for (int b = 0; b < KB; b++)
				for (int l = 0; l < LANES; l++) sK[j][g][b][l] = 128 + rnd() % 9 - 4;
	}
	asm volatile("csrw frm, zero");
	TIME("single", single());
	TIME("unroll", unroll(1));
	TIME("unscaled", unroll(0));
	return 0;
}
