// Cycles per instruction of scaled mode (docs/scaled-mode.md), with B = 1 and
// B = 32, against the same instruction unscaled, on P3109BlockV256D128ShuttleConfig. Each pattern
// is a run of independent LMUL-1 instructions (destinations rotate over four
// register pairs); cycles(256) - cycles(128), over 128, best of three, is the
// steady-state cost per instruction, free of start-up and drain. The data values do not affect
// the timing.
#include <stdio.h>
#include <stdint.h>
#include "util.h"

#define VT8(sc) (0xC0UL | ((unsigned long)(sc) << 11)) // e8 m1 ta ma, scaled bit 11
static inline void vset(unsigned long vt) {
	asm volatile("vsetvl x0, %0, %1" : : "r"(32UL), "r"(vt));
}
// Waits for every timed instruction: each store waits for the last writer of
// its register, and the scalar load of the stored bytes waits for the stores
static volatile uint8_t sink[8 * 32] __attribute__((aligned(64)));
static unsigned long drain(void) {
	asm volatile("vsetvli x0, %0, e8, m1, ta, ma" : : "r"(32UL));
	asm volatile("vse8.v v8, (%0)" : : "r"(&sink[0]) : "memory");
	asm volatile("vse8.v v10, (%0)" : : "r"(&sink[32]) : "memory");
	asm volatile("vse8.v v12, (%0)" : : "r"(&sink[64]) : "memory");
	asm volatile("vse8.v v14, (%0)" : : "r"(&sink[96]) : "memory");
	asm volatile("vse8.v v16, (%0)" : : "r"(&sink[128]) : "memory");
	asm volatile("vse8.v v18, (%0)" : : "r"(&sink[160]) : "memory");
	asm volatile("vse8.v v20, (%0)" : : "r"(&sink[192]) : "memory");
	asm volatile("vse8.v v22, (%0)" : : "r"(&sink[224]) : "memory");
	asm volatile("fence" : : : "memory");
	unsigned long x = 0;
	for (int i = 31; i < 256; i += 32) x += sink[i];
	return x;
}

#define X4(op8, op10, op12, op14) op8 "\n\t" op10 "\n\t" op12 "\n\t" op14 "\n\t"
#define X2(a) a a
#define X8(a) X2(X2(X2(a)))
#define X16(a) X2(X8(a))
#define X32(a) X2(X16(a))
#define X64(a) X2(X32(a))

#define PATTERN(name, a, b, c, d) \
	static unsigned long name(int n64, unsigned long vt) { \
		vset(vt); drain(); vset(vt); \
		unsigned long t = read_csr(mcycle); \
		if (n64) asm volatile(X64(X4(a, b, c, d)) ::: "memory"); /* 256 */ \
		else     asm volatile(X32(X4(a, b, c, d)) ::: "memory"); /* 128 */ \
		drain(); \
		return read_csr(mcycle) - t; \
	}
// 8-bit result (scales of vd in vd+1)
PATTERN(p_mul,  "vfmul.vv v8, v2, v4", "vfmul.vv v10, v2, v4", "vfmul.vv v12, v2, v4", "vfmul.vv v14, v2, v4")
// one destination: its scales (v9) are reused, at the cost of a WAW chain
PATTERN(p_mul1, "vfmul.vv v8, v2, v4", "vfmul.vv v8, v2, v4", "vfmul.vv v8, v2, v4", "vfmul.vv v8, v2, v4")
PATTERN(p_macc, "vfmacc.vv v8, v4, v2", "vfmacc.vv v10, v4, v2", "vfmacc.vv v12, v4, v2", "vfmacc.vv v14, v4, v2")
// BF16 result (widening)
PATTERN(p_wmul, "vfwmul.vv v16, v2, v4", "vfwmul.vv v18, v2, v4", "vfwmul.vv v20, v2, v4", "vfwmul.vv v22, v2, v4")
// conversions
PATTERN(p_wcvt, ".insn r 0x57, 0x1, 0x25, x16, x13, x2", ".insn r 0x57, 0x1, 0x25, x18, x13, x2",
                ".insn r 0x57, 0x1, 0x25, x20, x13, x2", ".insn r 0x57, 0x1, 0x25, x22, x13, x2")
PATTERN(p_ncvt, ".insn r 0x57, 0x1, 0x25, x8, x29, x24", ".insn r 0x57, 0x1, 0x25, x10, x29, x24",
                ".insn r 0x57, 0x1, 0x25, x12, x29, x24", ".insn r 0x57, 0x1, 0x25, x14, x29, x24")

static void report(const char *name, unsigned long (*f)(int, unsigned long)) {
	// unscaled, scaled with B = 1, scaled with B = 32 (vtype bits 14:12 = 5)
	const unsigned long vt[3] = {VT8(0), VT8(1), VT8(1) | (5UL << 12)};
	unsigned long c[3];
	for (int m = 0; m < 3; m++) {
		f(1, vt[m]); f(0, vt[m]); // warm up both: they are separate code
		c[m] = ~0UL;
		for (int r = 0; r < 3; r++) { // best of three
			unsigned long d = f(1, vt[m]) - f(0, vt[m]);
			if (d < c[m]) c[m] = d;
		}
	}
	// cycles per instruction over the 128 extra instructions, two decimals
	printf("%-6s unscaled %lu.%02lu  B=1 %lu.%02lu  B=32 %lu.%02lu cycles/instruction\n", name,
	       c[0] * 100 / 128 / 100, c[0] * 100 / 128 % 100, c[1] * 100 / 128 / 100, c[1] * 100 / 128 % 100,
	       c[2] * 100 / 128 / 100, c[2] * 100 / 128 % 100);
}

int main(void) {
	asm volatile("vsetvli x0, %0, e8, m8, ta, ma\n\tvmv.v.i v0, 0\n\tvmv.v.i v8, 0\n\tvmv.v.i v16, 0\n\tvmv.v.i v24, 0"
	             : : "r"(256UL));
	report("mul", p_mul);
	report("mul1", p_mul1);
	report("macc", p_macc);
	report("wmul", p_wmul);
	report("wcvt", p_wcvt);
	report("ncvt", p_ncvt);
	return 0;
}
