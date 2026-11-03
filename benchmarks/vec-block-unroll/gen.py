#!/usr/bin/env python3
"""Generates main.c: scaled kernels across the blocks (E = 1) at N blocks in flight.

N = the number of live data registers that need the same scales (the unroll
factor). Rule A needs a copy of the scales next to each of them: N - 1 copies
per block and scale operand, and 2 registers per block operand in flight, so N
is capped by the register file. With -DSHARED (P3109BlockSharedV256D128Shuttle-
Config) every instruction takes its scales from v31 (vs1), v30 (vs2), v29 (vd),
as with explicit scale operands: no copies, one register per operand.

  quantrow  BF16 -> FP8, scales on vd           rule A N <= 8, shared N <= 8
  addrow    FP8 + FP8 -> FP8, scales on all 3   rule A N <= 5, shared N <= 8
  gemv      y += W[:, j] x[j], scales on W      rule A N <= 8, shared N <= 8

Run: python3 gen.py > main.c
"""

NS = {"quantrow": [1, 2, 4, 8], "addrow": [1, 2, 4, 5, 8], "gemv": [1, 2, 4, 8]}
MAXN = {("quantrow", "A"): 8, ("addrow", "A"): 5, ("gemv", "A"): 8,
        ("quantrow", "S"): 8, ("addrow", "S"): 8, ("gemv", "S"): 8}
VL = 32


def v(n): return f"v{n}"
def x(n): return f"x{n}"


def alloc(kernel, scheme, i):
    """Registers of lane i. scheme: 'A' rule A (also used unscaled), 'S' shared."""
    if kernel == "quantrow":
        if scheme == "A": return {"dst": 2 * i, "sc": 2 * i + 1, "src": 16 + 2 * i}
        return {"dst": i, "src": 8 + 2 * i}
    if kernel == "addrow":
        if scheme == "A":
            b = 6 * i
            return {"x": b, "sx": b + 1, "r": b + 2, "sr": b + 3, "y": b + 4, "sy": b + 5}
        return {"x": 3 * i, "r": 3 * i + 1, "y": 3 * i + 2}
    if kernel == "gemv":
        if scheme == "A": return {"w": 2 * i, "sw": 2 * i + 1, "acc": 16 + 2 * i}
        return {"w": i, "acc": 8 + 2 * i}


def body(kernel, scheme, lanes, kexpr):
    """One pass over `lanes` k steps starting at kexpr."""
    L = [alloc(kernel, scheme, i) for i in range(lanes)]
    o = []
    if kernel == "quantrow":
        o += [f'LD16("{v(a["src"])}", X[{kexpr} + {i}]);' for i, a in enumerate(L)]
        o += [f'NCVT("{x(a["dst"])}", "{x(a["src"])}");' for a in L]
        o += [f'ST8("{v(a["dst"])}", Q[{kexpr} + {i}]);' for i, a in enumerate(L)]
    elif kernel == "addrow":
        for i, a in enumerate(L):
            o += [f'LD8("{v(a["x"])}", xa[{kexpr} + {i}]);', f'LD8("{v(a["r"])}", ra[{kexpr} + {i}]);']
        o += [f'asm volatile("vfadd.vv {v(a["y"])}, {v(a["x"])}, {v(a["r"])}");' for a in L]
        o += [f'ST8("{v(a["y"])}", ya[{kexpr} + {i}]);' for i, a in enumerate(L)]
    elif kernel == "gemv":
        o += [f'LD8("{v(a["w"])}", W[{kexpr} + {i}]);' for i, a in enumerate(L)]
        o += [f'asm volatile("fmv.w.x f0, %0\\n\\tvfwmacc.vf {v(a["acc"])}, f0, {v(a["w"])}" : : "r"(0xFFFFFF00u | xg[{kexpr} + {i}]) : "f0");'
              for i, a in enumerate(L)]
    return o


def scale_setup(kernel, scheme, n):
    """Per block: load the block's scales (and rule A's copies)."""
    L = [alloc(kernel, scheme, i) for i in range(n)]
    o = []
    if scheme == "S":
        if kernel == "quantrow": o.append('LD8("v29", srow[b]);')
        if kernel == "addrow": o += ['LD8("v30", sx[b]);', 'LD8("v31", sr[b]);', 'LD8("v29", sy[b]);']
        if kernel == "gemv": o.append('LD8("v30", sW[b]);')
        return o
    keys = {"quantrow": [("sc", "srow")], "addrow": [("sx", "sx"), ("sr", "sr"), ("sy", "sy")], "gemv": [("sw", "sW")]}[kernel]
    for key, arr in keys:
        o.append(f'LD8("{v(L[0][key])}", {arr}[b]);')
        for a in L[1:]:
            o.append(f'asm volatile("vmv.v.v {v(a[key])}, {v(L[0][key])}");')
    return o


def func(kernel, scheme, n, scaled):
    name = f"{kernel}_{'s' if scaled else 'u'}{scheme}{n}"
    o = [f"static void {name}(void) {{"]
    if kernel == "gemv":
        o.append("\tvset(VT_BF16);")
        o += [f'\tasm volatile("vmv.v.i {v(alloc(kernel, scheme, i)["acc"])}, 0");' for i in range(n)]
    o.append("\tfor (int b = 0; b < KB; b++) {")
    if scaled:
        o.append("\t\tvset(VT_E8);")
        o += ["\t\t" + s for s in scale_setup(kernel, scheme, n)]
    o.append(f"\t\tvset({'VT_E8S(0)' if scaled else 'VT_E8'});")
    full = (VL // n) * n
    o.append(f"\t\tfor (int k = b * VL; k < b * VL + {full}; k += {n}) {{")
    o += ["\t\t\t" + s for s in body(kernel, scheme, n, "k")]
    o.append("\t\t}")
    if full < VL:
        o.append(f"\t\t{{ const int k = b * VL + {full};")
        o += ["\t\t\t" + s for s in body(kernel, scheme, VL - full, "k")]
        o.append("\t\t}")
    o.append("\t}")
    if kernel == "gemv":
        o.append("\tvset(VT_BF16);")
        a0 = alloc(kernel, scheme, 0)["acc"]
        o += [f'\tasm volatile("vfadd.vv {v(a0)}, {v(a0)}, {v(alloc(kernel, scheme, i)["acc"])}");' for i in range(1, n)]
        o.append(f'\tST16("{v(a0)}", yg);')
    o.append("}")
    return name, "\n".join(o)


OUT = {"quantrow": ("&Q[K - 1][VL - 1]", "Q", "sizeof Q"), "addrow": ("&ya[K - 1][VL - 1]", "ya", "sizeof ya"),
       "gemv": ("&yg[VL - 1]", "yg", "sizeof yg")}

head = r'''// GENERATED by gen.py: do not edit. Blocks in flight (N) against rule A's
// register cost, across the blocks (E = 1), on P3109BlockV256D128ShuttleConfig
// (rule A and unscaled) and, with -DSHARED, P3109BlockSharedV256D128Shuttle-
// Config (one shared scale register per operand, as explicit operands allow).
// Rule A stops where its registers run out (addrow N = 5); the shared build and
// the unscaled loop go on to N = 8. Checksums: equal across N and builds for
// quantrow and addrow; gemv's sum order depends on N, so per N.
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

#define VL 32
#define K 256
#define KB (K / VL)
#define VT_E8      0xC0UL
#define VT_E8S(lg) (0x8C0UL | ((unsigned long)(lg) << 12))
#define VT_BF16    0x1C9UL
static inline void vset(unsigned long vt) { asm volatile("vsetvl x0, %0, %1" : : "r"((unsigned long)VL), "r"(vt)); }
#define LD8(r, p)  asm volatile("vle8.v "  r ", (%0)" : : "r"(p) : "memory")
#define LD16(r, p) asm volatile("vle16.v " r ", (%0)" : : "r"(p) : "memory")
#define ST8(r, p)  asm volatile("vse8.v "  r ", (%0)" : : "r"(p) : "memory")
#define ST16(r, p) asm volatile("vse16.v " r ", (%0)" : : "r"(p) : "memory")
#define NCVT(d, s) asm volatile(".insn r 0x57, 0x1, 0x25, " d ", x29, " s) // vfncvtbf16.f.f.w

static uint16_t X[K][VL] __attribute__((aligned(64)));
static uint8_t  Q[K][VL] __attribute__((aligned(64))), srow[KB][VL] __attribute__((aligned(64)));
static uint8_t  xa[K][VL] __attribute__((aligned(64))), ra[K][VL] __attribute__((aligned(64))), ya[K][VL] __attribute__((aligned(64)));
static uint8_t  sx[KB][VL] __attribute__((aligned(64))), sr[KB][VL] __attribute__((aligned(64))), sy[KB][VL] __attribute__((aligned(64)));
static uint8_t  W[K][VL] __attribute__((aligned(64))), sW[KB][VL] __attribute__((aligned(64))), xg[K] __attribute__((aligned(64)));
static uint16_t yg[VL] __attribute__((aligned(64)));
static volatile uint8_t sink;

static uint32_t seed = 4242;
static uint32_t rnd(void) { seed = seed * 1664525u + 1013904223u; return seed >> 8; }
static uint8_t rnd_fp8(void) { uint8_t c; do c = rnd(); while (c == 0x80 || (c & 0x7F) == 0x7F); return c; }
static uint16_t rnd_bf16(int lo, int hi) {
	unsigned e = 127 + lo + rnd() % (hi - lo + 1);
	return ((rnd() & 1) << 15) | (e << 7) | (rnd() & 0x7F);
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
#define TIME(kernel, n, label, call, last, out, sz) do { \
		call; start(last); for (int r_ = 0; r_ < REPS; r_++) { call; } unsigned long c = stop(last) / REPS; \
		printf("RESULT %-9s N %d  %-9s %7lu cycles  checksum %08lx\n", kernel, n, label, c, (unsigned long)checksum(out, sz)); \
	} while (0)
'''

funcs, calls = [], []
for kernel, ns in NS.items():
    last, out, sz = OUT[kernel]
    for n in ns:
        # unscaled (rule A's allocation, which fits every N here) in both binaries
        nm, f = func(kernel, "A" if n <= 8 and kernel != "addrow" or n <= 5 else "S", n, False)
        funcs.append(f)
        calls.append(f'\tTIME("{kernel}", {n}, "unscaled", {nm}(), {last}, {out}, 0);')
        if n <= MAXN[(kernel, "A")]:
            nm, f = func(kernel, "A", n, True)
            funcs.append("#ifndef SHARED\n" + f + "\n#endif")
            calls.append(f'#ifndef SHARED\n\tTIME("{kernel}", {n}, "ruleA", {nm}(), {last}, {out}, {sz});\n#endif')
        if n <= MAXN[(kernel, "S")]:
            nm, f = func(kernel, "S", n, True)
            funcs.append("#ifdef SHARED\n" + f + "\n#endif")
            calls.append(f'#ifdef SHARED\n\tTIME("{kernel}", {n}, "shared", {nm}(), {last}, {out}, {sz});\n#endif')

main = r'''
int main(void) {
	for (int k = 0; k < K; k++)
		for (int r = 0; r < VL; r++) {
			X[k][r] = rnd_bf16(-4, 12); xa[k][r] = rnd_fp8(); ra[k][r] = rnd_fp8(); W[k][r] = rnd_fp8();
		}
	for (int b = 0; b < KB; b++)
		for (int r = 0; r < VL; r++) {
			srow[b][r] = 128 + rnd() % 21 - 10; sW[b][r] = 128 + rnd() % 9 - 4;
			sx[b][r] = 128 + rnd() % 9 - 4; sr[b][r] = 128 + rnd() % 9 - 4; sy[b][r] = 128 + rnd() % 5 - 2;
		}
	for (int j = 0; j < K; j++) xg[j] = rnd_fp8();
''' + "\n".join(calls) + "\n\treturn 0;\n}\n"

print(head + "\n" + "\n\n".join(funcs) + "\n" + main)
