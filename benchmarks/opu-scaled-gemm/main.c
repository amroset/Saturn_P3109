// OPFMACC.S end to end: 64x64x96 GEMMs with P3109 block scales on the 2x2
// kernel, rule A registers (data in an even register, its scales in the next
// one). Each result is compared bit for bit with gen_data.py's reference.
// Three tests:
//   gemm   Normal use. Block 32, data in v2 v4 v6 v8, scales in v3 v5 v7 v9,
//          new scales loaded at every block, so the OPU's scale cache must
//          drop the old ones. Also a zero scale and a NaN scale.
//   evict  A stress test of the scale cache, not a way to write a kernel.
//          Two register sets (v2..v9 and v10..v17) take turns every k step,
//          each with different scales: eight scale registers share the
//          four-entry cache, so every OPFMACC.S evicts entries and refills
//          them. A wrong eviction or a stale entry would apply the other
//          set's scales and change the results.
//   gemm4  (-DFP4, configs with opuFP4) P3109 Binary4p2sf pairs, element-
//          format code 6: each byte holds steps k (bits 3:0) and k+1 (bits
//          7:4), so one OPFMACC.S does two k steps; a zero scale, a NaN.
//   gram   Tests vs1 = vs2. C = (sA x A)(sA x A)^T, so tiles m0 and m3 use
//          one register as both operands, and the cache fills that scale
//          register through both read ports in the same cycle.
// Built with -DSCALEDMODE (configs with scaled mode, docs/scaled-mode.md),
// the same tests use OPFMACC with vtype's scaled bit (bit 11) instead of
// OPFMACC.S, and one more runs:
//   coarse (gemm's kernel) 32 rows (columns) share a scale byte: vtype bits
//          14:12 = 5, so row r takes byte r >> 5, byte 0 of each scale
//          register; the other 31 bytes are random codes the OPU must ignore.
#include <stdio.h>
#include <stdint.h>
#include "rvv_mx.h"
#include "bme.h"

#define M 64
#define N 64
#define K 96
#define BLK 32

// Arrays defined in data.S, which gen_data.py writes; the linker resolves
// them by name. Per test (gemm, evict, gram) and format (0, 1: altfmt):
// _a, _b inputs, _sa, _sb their scales, _c the expected result from
// gen_data.py's exact reference, _out where the kernel stores its result.
#define DATA(t) extern uint8_t t##_a[], t##_b[], t##_sa[], t##_sb[]; extern uint32_t t##_c[], t##_out[];
DATA(gemm0) DATA(gemm1) DATA(evict0) DATA(evict1)
extern uint8_t gram0_a[], gram0_sa[], gram1_a[], gram1_sa[];
extern uint32_t gram0_c[], gram0_out[], gram1_c[], gram1_out[];

#define LD(v, p) asm volatile("vle8.v " v ", (%0)" : : "r"(p))

#ifdef SCALEDMODE
#if defined(FP4)
#error "the FP4 test sets vtype to code 6 without the scaled bit"
#endif
#define MACC OPFMACC
DATA(coarse0) DATA(coarse1)
static unsigned long scale_lg; // log2 of the rows (columns) per scale byte
#else
#define MACC OPFMACCS
#endif
#define K22(A0, A1, B0, B1, k) LD(#A0, &a[M * (k)]); LD(#A1, &a[M * (k) + 32]); \
    LD(#B0, &b[N * (k)]); LD(#B1, &b[N * (k) + 32]); \
    MACC(m0, B0, A0); MACC(m1, B1, A0); MACC(m2, B0, A1); MACC(m3, B1, A1);

// Tiles are cleared and read out through v12..v15. A tile move's vd must be
// below v16: bit 4 of the field is the OPU's column flag, and for OPMVOUT it
// swaps the quadrant indices, so a row moved out to v16..v31 mixes quadrants.
static void start(int altfmt) {
    asm volatile("vsetvli x0, %0, e32, m4, ta, ma\n\tvmv.v.i v12, 0" : : "r"(32));
    OPMVINBCAST(m0, v12); OPMVINBCAST(m1, v12); OPMVINBCAST(m2, v12); OPMVINBCAST(m3, v12);
#ifdef SCALEDMODE
    // e8 m1 ta ma, altfmt, scaled (bit 11), rows per scale byte (bits 14:12)
    unsigned long vt = 0x8C0UL | ((unsigned long)altfmt << 8) | (scale_lg << 12);
    asm volatile("vsetvl x0, %0, %1" : : "r"(32UL), "r"(vt));
#else
    if (altfmt) VSETVLI_ALTFMT_X0(32, SEW_E8, LMUL_M1, 1);
    else        VSETVLI_ALTFMT_X0(32, SEW_E8, LMUL_M1, 0);
#endif
}

// Saves the results from the OPU to memory: row r of each tile is moved into
// v12..v15 (32 FP32 values) and stored in that tile's quadrant of c.
static void finish(uint32_t *c) {
    asm volatile("vsetvli x0, %0, e32, m4, ta, ma" : : "r"(32));
    for (int r = 0; r < 32; r++) {
        VMV_VR(v12, r, m0); asm volatile("vse32.v v12, (%0)" : : "r"(&c[r * N]) : "memory");
        VMV_VR(v12, r, m1); asm volatile("vse32.v v12, (%0)" : : "r"(&c[r * N + 32]) : "memory");
        VMV_VR(v12, r, m2); asm volatile("vse32.v v12, (%0)" : : "r"(&c[(r + 32) * N]) : "memory");
        VMV_VR(v12, r, m3); asm volatile("vse32.v v12, (%0)" : : "r"(&c[(r + 32) * N + 32]) : "memory");
    }
}

static void gemm(const uint8_t *a, const uint8_t *b, const uint8_t *sa, const uint8_t *sb,
                 uint32_t *c, int altfmt) {
    start(altfmt);
    for (int blk = 0; blk < K / BLK; blk++) {
        LD("v3", &sa[M * blk]);
        LD("v5", &sa[M * blk + 32]);
        LD("v7", &sb[N * blk]);
        // Block 1 writes v9 with a register move instead of a load, to test
        // that the OPU's scale cache also sees writes from other units and
        // drops its old copy of v9.
        if (blk == 1) {
            LD("v20", &sb[N * blk + 32]);
            asm volatile("vmv1r.v v9, v20");
        } else {
            LD("v9", &sb[N * blk + 32]);
        }
        for (int k = blk * BLK; k < (blk + 1) * BLK; k++) {
            K22(v2, v4, v6, v8, k)
        }
    }
    finish(c);
}

// sa and sb hold one scale set per block and parity: set 2*blk on even k,
// 2*blk+1 on odd k
static void evict(const uint8_t *a, const uint8_t *b, const uint8_t *sa, const uint8_t *sb,
                uint32_t *c, int altfmt) {
    start(altfmt);
    for (int blk = 0; blk < K / BLK; blk++) {
        const uint8_t *se = &sa[M * 2 * blk], *so = &sa[M * (2 * blk + 1)];
        const uint8_t *te = &sb[N * 2 * blk], *to = &sb[N * (2 * blk + 1)];
        LD("v3", se); LD("v5", se + 32); LD("v7", te); LD("v9", te + 32);
        LD("v11", so); LD("v13", so + 32); LD("v15", to); LD("v17", to + 32);
        for (int k = blk * BLK; k < (blk + 1) * BLK; k += 2) {
            K22(v2, v4, v6, v8, k)
            K22(v10, v12, v14, v16, k + 1)
        }
    }
    finish(c);
}

static void gram(const uint8_t *a, const uint8_t *sa, uint32_t *c, int altfmt) {
    start(altfmt);
    for (int blk = 0; blk < K / BLK; blk++) {
        LD("v3", &sa[M * blk]);
        LD("v5", &sa[M * blk + 32]);
        for (int k = blk * BLK; k < (blk + 1) * BLK; k++) {
            LD("v2", &a[M * k]);
            LD("v4", &a[M * k + 32]);
            MACC(m0, v2, v2);
            MACC(m1, v4, v2);
            MACC(m2, v2, v4);
            MACC(m3, v4, v4);
        }
    }
    finish(c);
}

#ifdef FP4
extern uint8_t gemm4_a[], gemm4_b[], gemm4_sa[], gemm4_sb[];
extern uint32_t gemm4_c[], gemm4_out[];

// a and b hold K/2 rows of packed k pairs; scales per block of 32 k (16 pairs)
static void gemm4(const uint8_t *a, const uint8_t *b, const uint8_t *sa, const uint8_t *sb, uint32_t *c) {
    start(0);
    // e8 m1 with element-format code 6 (vtype[10:9] = 11): FP4 pairs
    asm volatile(".insn i 0x57, 0x7, x0, %0, 0x600" : : "r"(32));
    for (int blk = 0; blk < K / BLK; blk++) {
        LD("v3", &sa[M * blk]);
        LD("v5", &sa[M * blk + 32]);
        LD("v7", &sb[N * blk]);
        LD("v9", &sb[N * blk + 32]);
        for (int kp = blk * BLK / 2; kp < (blk + 1) * BLK / 2; kp++) {
            K22(v2, v4, v6, v8, kp)
        }
    }
    finish(c);
}
#endif

static int check(const char *test, const uint32_t *out, const uint32_t *ref, const char *fmt) {
    int bad = 0;
    for (int i = 0; i < M * N; i++)
        if (out[i] != ref[i] && bad++ < 4)
            printf("MISMATCH %s %s: m=%d n=%d got %08x want %08x\n",
                   test, fmt, i / N, i % N, out[i], ref[i]);
    printf("%s %s: %d of %d differ\n", test, fmt, bad, M * N);
    return bad != 0;
}

int main(void) {
    int failed = 0;
    gemm(gemm0_a, gemm0_b, gemm0_sa, gemm0_sb, gemm0_out, 0);
    failed |= check("gemm", gemm0_out, gemm0_c, "altfmt 0");
    gemm(gemm1_a, gemm1_b, gemm1_sa, gemm1_sb, gemm1_out, 1);
    failed |= check("gemm", gemm1_out, gemm1_c, "altfmt 1");
    evict(evict0_a, evict0_b, evict0_sa, evict0_sb, evict0_out, 0);
    failed |= check("evict", evict0_out, evict0_c, "altfmt 0");
    evict(evict1_a, evict1_b, evict1_sa, evict1_sb, evict1_out, 1);
    failed |= check("evict", evict1_out, evict1_c, "altfmt 1");
    gram(gram0_a, gram0_sa, gram0_out, 0);
    failed |= check("gram", gram0_out, gram0_c, "altfmt 0");
    gram(gram1_a, gram1_sa, gram1_out, 1);
    failed |= check("gram", gram1_out, gram1_c, "altfmt 1");
#ifdef FP4
    gemm4(gemm4_a, gemm4_b, gemm4_sa, gemm4_sb, gemm4_out);
    failed |= check("gemm4", gemm4_out, gemm4_c, "FP4");
#endif
#ifdef SCALEDMODE
    scale_lg = 5;
    gemm(coarse0_a, coarse0_b, coarse0_sa, coarse0_sb, coarse0_out, 0);
    failed |= check("coarse", coarse0_out, coarse0_c, "altfmt 0");
    gemm(coarse1_a, coarse1_b, coarse1_sa, coarse1_sb, coarse1_out, 1);
    failed |= check("coarse", coarse1_out, coarse1_c, "altfmt 1");
#endif
    printf(failed ? "FAILED\n" : "passed\n");
    return failed;
}
