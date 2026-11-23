"""Does the OPU's per-step rounding hurt accuracy? Step 1: kernel level.

    accum_study.py

Quantizes A (M x K) and B (K x N) to P3109 binary8p4 with a power-of-two
Binary8p1uf scale per 32 consecutive k (MX-style: the block's largest value
lands near the format's largest finite value), then computes C five ways:

  true     unquantized A @ B, in float64            (for the quantization error)
  exact    quantized, summed exactly, rounded once to float32
  block    exact BlockDotProduct per block, rounded, then added in float32
  opu      quantized, float32 rounding after every k step (the OPU)
  dot4     quantized, four products summed exactly, then float32 rounding
           (vfdot4: one rounding per four k, added 2026-11-03)
  hopper   14-bit accumulation, promoted to float32 every 128 k (the H800
           behaviour DeepSeek-V3 reports), and without promotion

The OPU emulation is exact: a scaled FP8 product has at most 8 significant
bits, so float64(c) + p is exact and its float32 rounding is the single
rounding the cell performs. check_emulation() confirms it bit for bit
against the reference the RTL is verified with.
"""
import os
import sys

import numpy as np
from gfloat import RoundMode, decode_ndarray, encode_ndarray, round_ndarray
from gfloat.formats import format_info_p3109

HERE = os.path.dirname(os.path.abspath(__file__))
SAT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "benchmarks")
P4 = format_info_p3109(8, 4)  # binary8p4, extended domain
RNE = RoundMode.TiesToEven
BLK = 32
_codes = decode_ndarray(P4, np.arange(256))
MAXF = float(np.max(np.abs(_codes[np.isfinite(_codes)])))


def quantize(X, axis):
    """Block-quantize X along `axis` (the K axis). Returns (values, scale exponents)."""
    Xk = np.moveaxis(X, axis, -1)
    sh = Xk.shape
    blocks = Xk.reshape(*sh[:-1], sh[-1] // BLK, BLK)
    amax = np.max(np.abs(blocks), axis=-1, keepdims=True)
    with np.errstate(divide="ignore"):
        e = np.where(amax > 0, np.ceil(np.log2(amax / MAXF)), 0.0)
    e = np.clip(e, -127, 126)                     # Binary8p1uf codes 1..254
    q = round_ndarray(P4, blocks / 2.0 ** e, RNE, sat=True)
    deq = (q * 2.0 ** e).reshape(sh)              # exact in float64
    return np.moveaxis(deq, -1, axis), np.moveaxis(np.repeat(e, BLK, -1).reshape(sh), -1, axis), \
        np.moveaxis(q.reshape(sh), -1, axis)


def gemm_opu(Ad, Bd):
    C = np.zeros((Ad.shape[0], Bd.shape[1]), np.float32)
    for k in range(Ad.shape[1]):
        C = (C.astype(np.float64) + np.outer(Ad[:, k], Bd[k, :])).astype(np.float32)
    return C


def add_round32(c, s):
    """float32(c + s) rounded once: c float32, s float64 exact. float64(c + s)
    can be inexact; TwoSum gives the lost part, which decides the rare case
    where the float64 sum lands exactly on a float32 midpoint."""
    c = c.astype(np.float64)
    hi = c + s
    bb = hi - c
    lo = (c - (hi - bb)) + (s - bb)
    r = hi.astype(np.float32)
    r64 = r.astype(np.float64)
    up = np.nextafter(r, np.float32(np.inf))
    dn = np.nextafter(r, np.float32(-np.inf))
    r = np.where((hi == (r64 + up.astype(np.float64)) / 2) & (lo > 0), up, r)
    r = np.where((hi == (r64 + dn.astype(np.float64)) / 2) & (lo < 0), dn, r)
    return r


def gemm_dot4(Ad, Bd):
    """vfdot4: the four products of a lane share a block, so their sum is exact in float64."""
    C = np.zeros((Ad.shape[0], Bd.shape[1]), np.float32)
    for k in range(0, Ad.shape[1], 4):
        C = add_round32(C, Ad[:, k:k + 4] @ Bd[k:k + 4, :])
    return C


def gemm_block(Ad, Bd):
    C = np.zeros((Ad.shape[0], Bd.shape[1]), np.float32)
    for k0 in range(0, Ad.shape[1], BLK):
        r = (Ad[:, k0:k0 + BLK] @ Bd[k0:k0 + BLK, :]).astype(np.float32)
        C = (C.astype(np.float64) + r).astype(np.float32)
    return C


def round_bits(x, bits, trunc=False):
    m, e = np.frexp(x)
    m = (np.trunc if trunc else np.round)(m * 2.0 ** bits) / 2.0 ** bits
    return np.ldexp(m, e)


def gemm_hopper(Ad, Bd, interval=128, bits=14, trunc=False):
    C = np.zeros((Ad.shape[0], Bd.shape[1]), np.float32)
    acc = np.zeros(C.shape)
    for k in range(Ad.shape[1]):
        acc = round_bits(acc + np.outer(Ad[:, k], Bd[k, :]), bits, trunc)
        if interval and (k + 1) % interval == 0:
            C = (C.astype(np.float64) + acc).astype(np.float32)
            acc[:] = 0
    if not interval:
        C = acc.astype(np.float32)
    return C


def errors(x, ref, absp):
    d = np.abs(x.astype(np.float64) - ref)
    nz = ref != 0
    rel = d[nz] / np.abs(ref[nz])
    return {"frob": np.linalg.norm(d) / np.linalg.norm(ref), "med": np.median(rel),
            "p99": np.percentile(rel, 99), "max": rel.max(), "norm": (d / absp).max(),
            "same": np.mean(x.astype(np.float64) == ref)}


def study(name, A, B, hopper=True):
    Ad, _, _ = quantize(A, 1)
    Bd, _, _ = quantize(B, 0)
    true = A @ B
    exact64 = Ad @ Bd
    exact = exact64.astype(np.float32).astype(np.float64)
    absp = np.abs(Ad) @ np.abs(Bd)
    rows = {"quantization (exact vs true)": errors(exact.astype(np.float32), true, np.abs(A) @ np.abs(B)),
            "opu vs exact": errors(gemm_opu(Ad, Bd), exact, absp),
            "dot4 vs exact": errors(gemm_dot4(Ad, Bd), exact, absp),
            "block vs exact": errors(gemm_block(Ad, Bd), exact, absp)}
    if hopper:
        rows["hopper, FP32 every 128 vs exact"] = errors(gemm_hopper(Ad, Bd), exact, absp)
        rows["hopper, no promotion vs exact"] = errors(gemm_hopper(Ad, Bd, interval=0), exact, absp)
    print(f"\n== {name}: M={A.shape[0]} N={B.shape[1]} K={A.shape[1]}")
    print(f"   {'':34} {'Frobenius':>10} {'median':>10} {'p99':>10} {'max rel':>10} {'max/S|p|':>10} {'identical':>9}")
    for k, r in rows.items():
        print(f"   {k:34} {r['frob']:10.2e} {r['med']:10.2e} {r['p99']:10.2e} {r['max']:10.2e} {r['norm']:10.2e} {r['same']:9.1%}")
    q, o = rows["quantization (exact vs true)"]["frob"], rows["opu vs exact"]["frob"]
    print(f"   OPU rounding error / quantization error (Frobenius): {o / q:.1e}")
    print(f"   dot4 rounding error / quantization error (Frobenius): {rows['dot4 vs exact']['frob'] / q:.1e}")
    return rows


def check_emulation():
    """The numpy OPU emulation against opu-scaled-gemm's reference (fma_ref), bit for bit."""
    sys.path.insert(0, os.path.join(SAT, "common-data-gen"))
    sys.path.insert(0, os.path.join(SAT, "opu-scaled-gemm", "gen_data"))
    import gen_data as g   # M = N = 64, K = 96, block 32
    rng = np.random.default_rng(7)
    A = rng.standard_normal((g.M, g.K)) * np.exp(rng.standard_normal((g.M, 1)))
    B = rng.standard_normal((g.K, g.N)) * np.exp(rng.standard_normal((1, g.N)))
    Ad, eA, qA = quantize(A, 1)
    Bd, eB, qB = quantize(B, 0)
    a = encode_ndarray(P4, qA).T.astype(int).tolist()      # a[k][i]
    b = encode_ndarray(P4, qB).astype(int).tolist()        # b[k][j]
    sa = (eA + 128).T.astype(int).tolist()
    sb = (eB + 128).astype(int).tolist()
    ref = np.array(g.reference("p3109", P4, a, b, sa, sb), np.uint32).view(np.float32).reshape(g.M, g.N)
    emu = gemm_opu(Ad, Bd)
    same = np.mean(emu == ref)
    print(f"emulation check vs fma_ref ({g.M}x{g.N}x{g.K}): {same:.1%} of outputs bit-identical")
    assert same == 1.0


def check_dot4():
    """The numpy vfdot4 emulation against the reference the vfdot4 lane is verified with
    (models/dot_vectors.py: exact rational arithmetic, one rounding), bit for bit."""
    sys.path.insert(0, os.path.join(SAT, "common-data-gen"))
    sys.path.insert(0, os.path.join(SAT, "..", "models"))
    from gfloat_ref import FP8_STANDARDS
    from dot_vectors import expect, pack
    fi = [f for f in FP8_STANDARDS["p3109"].values() if f.precision == 4][0]
    rng = np.random.default_rng(11)
    M, N, K = 16, 16, 96
    A = rng.standard_normal((M, K)) * np.exp(2 * rng.standard_normal((M, 1)))
    B = rng.standard_normal((K, N)) * np.exp(2 * rng.standard_normal((1, N)))
    Ad, eA, qA = quantize(A, 1)
    Bd, eB, qB = quantize(B, 0)
    ca, cb = encode_ndarray(P4, qA).astype(int), encode_ndarray(P4, qB).astype(int)
    emu = gemm_dot4(Ad, Bd)
    ref = np.zeros((M, N), np.uint32)
    for i in range(M):
        for j in range(N):
            c = 0
            for k in range(0, K, 4):
                c = expect("p3109", fi, pack([int(v) for v in ca[i, k:k + 4]]), pack([int(v) for v in cb[k:k + 4, j]]), c,
                           int(eA[i, k]) + 128, int(eB[k, j]) + 128, 0)
            ref[i, j] = c
    same = np.mean(emu.view(np.uint32) == ref)
    print(f"dot4 emulation check vs dot_vectors.expect ({M}x{N}x{K}): {same:.1%} of outputs bit-identical")
    assert same == 1.0


if __name__ == "__main__":
    check_emulation()
    check_dot4()
    rng = np.random.default_rng(1)
    M = N = 256
    # 1. DeepSeek-V3's case: two random matrices, K = 4096
    study("gaussian", rng.standard_normal((M, 4096)), rng.standard_normal((4096, N)))
    # 2. activation-like outliers: 1% of the K channels 100x larger; small weights
    K = 4096
    A = rng.standard_normal((M, K))
    A[:, rng.choice(K, K // 100, replace=False)] *= 100
    study("outlier channels", A, 0.02 * rng.standard_normal((K, N)))
    # 3. heavy tails on both sides (Student t, 3 degrees of freedom)
    study("student-t", rng.standard_t(3, (M, K)), rng.standard_t(3, (K, N)))
    # 4. no cancellation: all products positive
    study("positive", np.abs(rng.standard_normal((M, K))), np.abs(rng.standard_normal((K, N))))
    # 5. K sweep, gaussian: OPU against H800-style accumulation
    print("\n== K sweep, gaussian (Frobenius relative error vs exact)")
    for K in (512, 1024, 2048, 4096, 8192):
        A, B = rng.standard_normal((128, K)), rng.standard_normal((K, 128))
        Ad, _, _ = quantize(A, 1); Bd, _, _ = quantize(B, 0)
        exact = (Ad @ Bd).astype(np.float32).astype(np.float64)
        absp = np.abs(Ad) @ np.abs(Bd)
        o = errors(gemm_opu(Ad, Bd), exact, absp)
        d4 = errors(gemm_dot4(Ad, Bd), exact, absp)
        h = errors(gemm_hopper(Ad, Bd, interval=0), exact, absp)
        q = errors(exact.astype(np.float32), A @ B, np.abs(A) @ np.abs(B))
        print(f"   K={K:5}: quantization {q['frob']:.2e} | opu {o['frob']:.2e} (max rel {o['max']:.1e}) | dot4 {d4['frob']:.2e} | hopper, no promotion {h['frob']:.2e} (max rel {h['max']:.1e})")
