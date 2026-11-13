// Scaled mode (docs/scaled-mode.md): register uses that the pairing rule
// leaves undefined raise an illegal-instruction exception, and their legal
// neighbours do not. Each case runs one instruction under its vtype;
// handle_trap records the cause and skips the instruction. Register contents
// are irrelevant: only whether the instruction traps is checked.
//   default: vector cases, on a scaled-mode build (P3109Block*)
//   -DDOT:   also vfdot4, on P3109BlockDot*
//   -DOPU:   OPFMACC.S and scaled OPFMACC, on OPUV256D128P3109Block
#include <stdio.h>
#include <stdint.h>

static volatile uintptr_t trap_cause;

uintptr_t handle_trap(uintptr_t cause, uintptr_t epc, uintptr_t regs[32]) {
	trap_cause = cause;
	return epc + 4;
}

// vtype: SEW 8 << sew, LMUL 2^lm, ta ma; scaled (bit 11) with E = 1
#define VT(sew, lm, sc) (0xC0UL | ((unsigned long)(sew) << 3) | (lm) | ((unsigned long)(sc) << 11))

static int failed = 0;
#define CASE(vt, illegal, insn) do { \
	asm volatile("vsetvl x0, %0, %1" : : "r"(8UL), "r"(vt)); \
	trap_cause = 0; \
	asm volatile(insn ::: "memory"); \
	int ok = (illegal) ? trap_cause == 2 : trap_cause == 0; \
	printf("%s %-42s %-7s cause %lu\n", ok ? "ok  " : "FAIL", insn, (illegal) ? "illegal" : "legal", (unsigned long)trap_cause); \
	failed |= !ok; \
} while (0)

int main(void) {
#ifndef OPU
	// LMUL 1
	CASE(VT(0, 0, 1), 1, "vfadd.vv v9, v8, v10");          // vd is vs2's scale register
	CASE(VT(0, 0, 1), 1, "vfadd.vv v11, v8, v10");         // vd is vs1's scale register
	CASE(VT(0, 0, 0), 0, "vfadd.vv v9, v8, v10");          // unscaled: no scale registers
	CASE(VT(0, 0, 1), 0, "vfadd.vv v2, v4, v6");
	CASE(VT(0, 0, 1), 0, "vfadd.vv v2, v4, v6, v0.t");
	CASE(VT(0, 0, 1), 0, "vfadd.vv v8, v2, v3");           // vs1 is vs2's scale register: read only
	CASE(VT(0, 0, 1), 1, "vfwadd.vv v2, v1, v8");          // wide vd v2-v3 holds vs2's scales
	CASE(VT(0, 0, 1), 0, "vfwadd.vv v4, v1, v8");
	CASE(VT(0, 0, 1), 1, "vfadd.vv v2, v31, v4");          // vs2's scales past v31
	CASE(VT(0, 0, 1), 1, ".insn r 0x57, 0x1, 0x25, x2, x13, x1"); // vfwcvtbf16.f.f.v v2, v1
	CASE(VT(0, 0, 1), 0, ".insn r 0x57, 0x1, 0x25, x4, x13, x2"); // vfwcvtbf16.f.f.v v4, v2
	// LMUL 2
	CASE(VT(0, 1, 1), 1, "vfadd.vv v4, v2, v8");           // v2's scales are v4-v5
	CASE(VT(0, 1, 1), 1, "vfadd.vv v8, v3, v12");          // v3 not aligned to the group
	CASE(VT(0, 1, 1), 0, "vfadd.vv v8, v4, v12");
	CASE(VT(0, 1, 1), 1, "vfadd.vv v8, v30, v12");         // v30's scales past v31
	// LMUL 4
	CASE(VT(0, 2, 1), 1, "vfwadd.vv v8, v4, v16");         // wide vd v8-v15 holds v4's scales
	CASE(VT(0, 2, 1), 0, "vfmul.vv v8, v16, v24");
	CASE(VT(0, 2, 1), 1, "vfmul.vv v8, v28, v16");         // v28's scales past v31
#ifdef DOT
	// vfdot4 (SEW 32): vs2 and .vv's vs1 are blocks, vd is not
	CASE(VT(2, 0, 1), 1, ".insn r 0x57, 0x1, 0x6B, x9, x16, x8");  // vfdot4.vv v9, v8, v16
	CASE(VT(2, 0, 1), 0, ".insn r 0x57, 0x1, 0x6B, x24, x16, x8"); // vfdot4.vv v24, v8, v16
	CASE(VT(2, 0, 1), 1, ".insn r 0x57, 0x5, 0x6B, x9, x5, x8");   // vfdot4.vx v9, v8, t0
	CASE(VT(2, 0, 1), 0, ".insn r 0x57, 0x5, 0x6B, x24, x5, x8");  // vfdot4.vx v24, v8, t0
#endif
#else
	// OPFMACC.S m0, vs2, vs1 and OPFMACC: scales in vN+1, N even
	CASE(VT(0, 0, 0), 0, ".insn r 0x57, 0x1, 0x4d, x0, x4, x2");   // opfmacc.s m0, v2, v4
	CASE(VT(0, 0, 0), 1, ".insn r 0x57, 0x1, 0x4d, x0, x3, x2");   // vs1 v3
	CASE(VT(0, 0, 0), 1, ".insn r 0x57, 0x1, 0x4d, x0, x4, x5");   // vs2 v5
	CASE(VT(0, 0, 1), 1, ".insn r 0x57, 0x1, 0x4b, x0, x3, x2");   // scaled opfmacc, vs1 v3
	CASE(VT(0, 0, 0), 0, ".insn r 0x57, 0x1, 0x4b, x0, x3, x2");   // unscaled opfmacc, vs1 v3
#endif
	printf(failed ? "FAILED\n" : "PASSED\n");
	return failed;
}
