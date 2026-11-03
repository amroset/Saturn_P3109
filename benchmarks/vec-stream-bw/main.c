// Vector memory bandwidth on the benchmarks' working-set sizes, for the roofline
// of the block-scaling study: load-only, store-only and copy loops at LMUL 1
// and 8 over 16 KB, warmed once, then 4 timed passes. Bytes per cycle.
#include <stdio.h>
#include <stdint.h>
#include "util.h"

#define NB 16384
static uint8_t src[NB] __attribute__((aligned(64))), dst[NB] __attribute__((aligned(64)));
static volatile uint8_t sink;
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
#define LOOP(name, LMUL, BODY) \
	static void name(void) { \
		unsigned long n = 32 * LMUL; \
		asm volatile("vsetvli x0, %0, e8, m" #LMUL ", ta, ma" : : "r"(n)); \
		for (unsigned long j = 0; j < NB; j += n) { BODY } \
	}
LOOP(ld1, 1, asm volatile("vle8.v v8, (%0)" : : "r"(&src[j]) : "memory");)
LOOP(st1, 1, asm volatile("vse8.v v8, (%0)" : : "r"(&dst[j]) : "memory");)
LOOP(cp1, 1, asm volatile("vle8.v v8, (%0)\n\tvse8.v v8, (%1)" : : "r"(&src[j]), "r"(&dst[j]) : "memory");)
LOOP(ld8, 8, asm volatile("vle8.v v8, (%0)" : : "r"(&src[j]) : "memory");)
LOOP(st8, 8, asm volatile("vse8.v v8, (%0)" : : "r"(&dst[j]) : "memory");)
LOOP(cp8, 8, asm volatile("vle8.v v8, (%0)\n\tvse8.v v8, (%1)" : : "r"(&src[j]), "r"(&dst[j]) : "memory");)
static void run(const char *name, void (*f)(void), unsigned long bytes) {
	f(); start(&dst[NB - 1]); for (int r = 0; r < 4; r++) f();
	unsigned long c = stop(&dst[NB - 1]) / 4;
	printf("RESULT %-6s %6lu cycles  %lu bytes  %lu.%02lu bytes/cycle\n", name, c, bytes, bytes / c, bytes * 100 / c % 100);
}
int main(void) {
	for (int i = 0; i < NB; i++) src[i] = i;
	run("load1", ld1, NB); run("store1", st1, NB); run("copy1", cp1, 2 * NB);
	run("load8", ld8, NB); run("store8", st8, NB); run("copy8", cp8, 2 * NB);
	return 0;
}
