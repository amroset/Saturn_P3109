// =============================================================================
// IEEE P3109 scalar operations, in software
// =============================================================================
//
// What this file is for
// ---------------------
// The vector unit implements the *arithmetic* on P3109 8-bit numbers:
// conversion (FPConv) and multiply-add (FPFMAPipe). It does not implement the
// rest of what P3109 4.5 asks a conforming implementation to provide:
//
//   - the twelve format-level operations (BitwidthOf, MaxFiniteOf, ...)
//   - the ten predicates (IsZero, IsNaN, IsNormal, ...) and the classifier
//   - Negate and Abs
//   - the five comparisons
//   - the ten minimum/maximum variants
//   - NextGreaterThan / NextLessThan
//
// None of those need hardware: they are bit tests and integer comparisons on
// the 8-bit code point. This header supplies them, so that "the implementation"
// -- hardware plus this library -- covers the required set.
//
// Everything here works on *code points*: the plain integer value of the K bits,
// 0 .. 2^K - 1. No floating-point arithmetic is used, so this is safe to call
// from a bare-metal benchmark.
//
// Observations
// -----------------------------
// 1. "The low bits count upward."
//    In a signed P3109 format the top bit is the sign and the remaining K-1
//    bits are a plain increasing sequence: 0, then the subnormals, then the
//    normals, then (in the extended domain) infinity. So comparing magnitudes
//    is comparing integers. See the value tables in Annex B of the standard.
//
// 2. "There is exactly one zero and exactly one NaN."
//    Zero is code point 0. NaN sits where IEEE-754 would put negative zero,
//    that is 2^(K-1) for a signed format. So there is no -0 to special-case,
//    and a single equality test finds NaN.
//
// Reference: IEEE P3109 Interim Report v4.0.3, sections 3.1, 4.7.2, 4.11-4.14
// and 4.16.

#ifndef __P3109_H
#define __P3109_H

#include <stdint.h>

// -----------------------------------------------------------------------------
// Formats
// -----------------------------------------------------------------------------
// A format is fixed by four numbers (3.1): bitwidth K, precision P, whether it
// is signed, and whether its domain is "extended" (has infinities) or "finite".
// Everything else -- the exponent bias, where NaN lives, the largest finite
// value -- follows from those four.

typedef struct {
  int k;            // bitwidth K
  int p;            // precision P (trailing significand bits, plus one)
  int is_signed;    // 1 = signed, 0 = unsigned
  int is_extended;  // 1 = extended domain (has infinities), 0 = finite
} p3109_fmt_t;

// The formats P3109 4.5 requires, plus the scale format used by block
// operations (5.8). Named as in 3.2: Binary<K>p<P><s|u><e|f>.
#define P3109_BINARY8P4SE ((p3109_fmt_t){8, 4, 1, 1})  // F8, altfmt = 0
#define P3109_BINARY8P3SE ((p3109_fmt_t){8, 3, 1, 1})  // F8, altfmt = 1
#define P3109_BINARY8P4SF ((p3109_fmt_t){8, 4, 1, 0})  // finite-domain variant
#define P3109_BINARY8P3SF ((p3109_fmt_t){8, 3, 1, 0})  // finite-domain variant
#define P3109_BINARY4P2SF ((p3109_fmt_t){4, 2, 1, 0})  // F4
#define P3109_BINARY8P1UF ((p3109_fmt_t){8, 1, 0, 0})  // Fs, the block scale

typedef uint32_t p3109_t;  // a code point, 0 .. 2^K - 1

// -----------------------------------------------------------------------------
// 4.14  Format-level operations
// -----------------------------------------------------------------------------
// These act on the format, not on a value. All are compile-time constants in
// practice; they mirror the P3109FormatInfo case class in the Chisel source.

typedef enum { P3109_SIGNED, P3109_UNSIGNED } p3109_signedness_t;
typedef enum { P3109_FINITE, P3109_EXTENDED } p3109_domain_t;

static inline int p3109_bitwidth_of(p3109_fmt_t f) { return f.k; }
static inline int p3109_precision_of(p3109_fmt_t f) { return f.p; }

static inline p3109_signedness_t p3109_signedness_of(p3109_fmt_t f) {
  return f.is_signed ? P3109_SIGNED : P3109_UNSIGNED;
}

static inline p3109_domain_t p3109_domain_of(p3109_fmt_t f) {
  return f.is_extended ? P3109_EXTENDED : P3109_FINITE;
}

// A signed format spends one bit on the sign, so it has one fewer exponent bit.
static inline int p3109_exponent_bitwidth_of(p3109_fmt_t f) {
  return f.is_signed ? (f.k - f.p) : (f.k - f.p + 1);
}

static inline int p3109_trailing_significand_bitwidth_of(p3109_fmt_t f) {
  return f.p - 1;
}

// 3.1: B = 2^(K-P-1) signed, 2^(K-P) unsigned. This is one more than IEEE-754
// would use for the same shape, which is why the same bits mean half as much.
static inline int p3109_exponent_bias_of(p3109_fmt_t f) {
  return 1 << (f.is_signed ? (f.k - f.p - 1) : (f.k - f.p));
}

// -----------------------------------------------------------------------------
// Special code points (Annex B, Table 3)
// -----------------------------------------------------------------------------

static inline p3109_t p3109_nan_of(p3109_fmt_t f) {
  return f.is_signed ? (1u << (f.k - 1)) : ((1u << f.k) - 1);
}

// Only meaningful in the extended domain.
static inline p3109_t p3109_inf_of(p3109_fmt_t f) {
  return f.is_signed ? ((1u << (f.k - 1)) - 1) : ((1u << f.k) - 2);
}

static inline p3109_t p3109_neg_inf_of(p3109_fmt_t f) {  // signed extended only
  return (1u << f.k) - 1;
}

// The magnitude field: everything below the sign bit.
static inline p3109_t p3109_mag_mask(p3109_fmt_t f) {
  return f.is_signed ? ((1u << (f.k - 1)) - 1) : ((1u << f.k) - 1);
}

// -----------------------------------------------------------------------------
// 4.13  Predicates
// -----------------------------------------------------------------------------

static inline int p3109_is_nan(p3109_fmt_t f, p3109_t x) {
  return x == p3109_nan_of(f);
}

static inline int p3109_is_infinite(p3109_fmt_t f, p3109_t x) {
  if (!f.is_extended) return 0;
  if (!f.is_signed) return x == p3109_inf_of(f);
  return x == p3109_inf_of(f) || x == p3109_neg_inf_of(f);
}

static inline int p3109_is_finite(p3109_fmt_t f, p3109_t x) {
  return !p3109_is_nan(f, x) && !p3109_is_infinite(f, x);
}

static inline int p3109_is_zero(p3109_fmt_t f, p3109_t x) {
  (void)f;
  return x == 0;  // there is exactly one zero, and it is code point 0
}

// 1.0 always sits at the midway code point (Annex A.5): 2^(K-2) signed,
// 2^(K-1) unsigned.
static inline int p3109_is_one(p3109_fmt_t f, p3109_t x) {
  return x == (1u << (f.is_signed ? (f.k - 2) : (f.k - 1)));
}

// NaN is not sign-minus even though its code point has the top bit set.
static inline int p3109_is_sign_minus(p3109_fmt_t f, p3109_t x) {
  if (!f.is_signed || p3109_is_nan(f, x)) return 0;
  return (x >> (f.k - 1)) & 1u;
}

// A value is normal when its exponent field is nonzero, i.e. when its magnitude
// reaches the first code point with a hidden leading one.
static inline int p3109_is_normal(p3109_fmt_t f, p3109_t x) {
  if (p3109_is_zero(f, x) || p3109_is_nan(f, x) || p3109_is_infinite(f, x))
    return 0;
  return (x & p3109_mag_mask(f)) >= (p3109_t)(1u << (f.p - 1));
}

static inline int p3109_is_subnormal(p3109_fmt_t f, p3109_t x) {
  if (p3109_is_zero(f, x) || p3109_is_nan(f, x) || p3109_is_infinite(f, x))
    return 0;
  return !p3109_is_normal(f, x);
}

// 4.13.1  Classifier
typedef enum {
  P3109_CLS_NAN,
  P3109_CLS_NEGATIVE_INFINITY,
  P3109_CLS_NEGATIVE_NORMAL,
  P3109_CLS_NEGATIVE_SUBNORMAL,
  P3109_CLS_ZERO,
  P3109_CLS_POSITIVE_SUBNORMAL,
  P3109_CLS_POSITIVE_NORMAL,
  P3109_CLS_POSITIVE_INFINITY
} p3109_class_t;

static inline p3109_class_t p3109_class(p3109_fmt_t f, p3109_t x) {
  int neg = p3109_is_sign_minus(f, x);
  if (p3109_is_nan(f, x)) return P3109_CLS_NAN;
  if (p3109_is_infinite(f, x))
    return neg ? P3109_CLS_NEGATIVE_INFINITY : P3109_CLS_POSITIVE_INFINITY;
  if (p3109_is_zero(f, x)) return P3109_CLS_ZERO;
  if (p3109_is_normal(f, x))
    return neg ? P3109_CLS_NEGATIVE_NORMAL : P3109_CLS_POSITIVE_NORMAL;
  return neg ? P3109_CLS_NEGATIVE_SUBNORMAL : P3109_CLS_POSITIVE_SUBNORMAL;
}

// -----------------------------------------------------------------------------
// 4.14  Values that depend on the format (returned as code points)
// -----------------------------------------------------------------------------

static inline p3109_t p3109_max_finite_of(p3109_fmt_t f) {
  // The largest code below the special ones. In the extended domain the top
  // magnitude code is infinity, so step one below it.
  p3109_t top = p3109_mag_mask(f);          // signed: 2^(K-1)-1, unsigned: 2^K-1
  if (!f.is_signed) top -= 1;               // unsigned NaN sits at 2^K - 1
  return f.is_extended ? (top - 1) : top;
}

static inline p3109_t p3109_min_finite_of(p3109_fmt_t f) {
  if (!f.is_signed) return 0;
  return p3109_max_finite_of(f) + (1u << (f.k - 1));  // its negation
}

static inline p3109_t p3109_min_positive_of(p3109_fmt_t f) {
  (void)f;
  return 1;  // the code point just above zero
}

// NaN when the format has no subnormals at all (P = 1).
static inline p3109_t p3109_max_subnormal_of(p3109_fmt_t f) {
  if (f.p == 1) return p3109_nan_of(f);
  return (1u << (f.p - 1)) - 1;
}

static inline p3109_t p3109_min_normal_of(p3109_fmt_t f) {
  return 1u << (f.p - 1);
}

// -----------------------------------------------------------------------------
// 4.10.1  Negate, Abs
// -----------------------------------------------------------------------------
// Flip or clear the sign bit -- but NaN has no sign, so leave it alone. (There
// is no -0 to worry about: negating zero gives the code point of NaN if done
// naively, which is why zero is checked too.)
//
// In an *unsigned* format the negation of a nonzero value is not representable.
// 4.10.1 defines Negate as ωProject of -X, and for a finite unsigned format
// 4.7.5 sends an out-of-range value to NaN, so that is what is returned. (This
// case never arises for the required set -- F4 and F8 are all signed -- but the
// scale format Binary8p1uf is unsigned, so it is worth getting right.)

static inline p3109_t p3109_negate(p3109_fmt_t f, p3109_t x) {
  if (p3109_is_nan(f, x) || p3109_is_zero(f, x)) return x;
  if (!f.is_signed) return p3109_nan_of(f);
  return x ^ (1u << (f.k - 1));
}

static inline p3109_t p3109_abs(p3109_fmt_t f, p3109_t x) {
  if (!f.is_signed || p3109_is_nan(f, x)) return x;
  return x & p3109_mag_mask(f);
}

// -----------------------------------------------------------------------------
// 4.12  Comparisons
// -----------------------------------------------------------------------------
// Turn a code point into a signed integer that sorts the same way the values
// do: negative values become negative keys, and the magnitude counts upward.
// Infinity needs no special handling -- it already has the largest magnitude.
// NaN has no key; callers test for it first.

static inline int32_t p3109_order_key(p3109_fmt_t f, p3109_t x) {
  int32_t mag = (int32_t)(x & p3109_mag_mask(f));
  return p3109_is_sign_minus(f, x) ? -mag : mag;
}

// 4.12: every comparison involving NaN is false, including CompareEqual.
#define P3109_CMP_NAN_GUARD(f, x, y) \
  if (p3109_is_nan(f, x) || p3109_is_nan(f, y)) return 0;

static inline int p3109_compare_less(p3109_fmt_t f, p3109_t x, p3109_t y) {
  P3109_CMP_NAN_GUARD(f, x, y)
  return p3109_order_key(f, x) < p3109_order_key(f, y);
}

static inline int p3109_compare_less_equal(p3109_fmt_t f, p3109_t x, p3109_t y) {
  P3109_CMP_NAN_GUARD(f, x, y)
  return p3109_order_key(f, x) <= p3109_order_key(f, y);
}

static inline int p3109_compare_equal(p3109_fmt_t f, p3109_t x, p3109_t y) {
  P3109_CMP_NAN_GUARD(f, x, y)
  return p3109_order_key(f, x) == p3109_order_key(f, y);
}

static inline int p3109_compare_greater(p3109_fmt_t f, p3109_t x, p3109_t y) {
  P3109_CMP_NAN_GUARD(f, x, y)
  return p3109_order_key(f, x) > p3109_order_key(f, y);
}

static inline int p3109_compare_greater_equal(p3109_fmt_t f, p3109_t x,
                                              p3109_t y) {
  P3109_CMP_NAN_GUARD(f, x, y)
  return p3109_order_key(f, x) >= p3109_order_key(f, y);
}

// 4.12.1  Total order: NaN sorts below everything, and the order is total.
static inline int p3109_total_order(p3109_fmt_t f, p3109_t x, p3109_t y) {
  if (p3109_is_nan(f, x)) return 1;
  if (p3109_is_nan(f, y)) return 0;
  return p3109_compare_less_equal(f, x, y);
}

// -----------------------------------------------------------------------------
// 4.11  Minimum and maximum, ten variants
// -----------------------------------------------------------------------------
// The variants differ only in which operands they are willing to ignore:
//
//   plain            NaN wins (the result is NaN)
//   ...Number        NaN is ignored unless both operands are NaN
//   ...Magnitude     compares |x| instead of x, ties broken by value
//   ...Finite        infinities are ignored as well as NaN

static inline p3109_t p3109_minimum(p3109_fmt_t f, p3109_t x, p3109_t y) {
  if (p3109_is_nan(f, x) || p3109_is_nan(f, y)) return p3109_nan_of(f);
  return p3109_order_key(f, x) < p3109_order_key(f, y) ? x : y;
}

static inline p3109_t p3109_maximum(p3109_fmt_t f, p3109_t x, p3109_t y) {
  if (p3109_is_nan(f, x) || p3109_is_nan(f, y)) return p3109_nan_of(f);
  return p3109_order_key(f, x) < p3109_order_key(f, y) ? y : x;
}

static inline p3109_t p3109_minimum_number(p3109_fmt_t f, p3109_t x, p3109_t y) {
  if (p3109_is_nan(f, x) && p3109_is_nan(f, y)) return p3109_nan_of(f);
  if (p3109_is_nan(f, x)) return y;
  if (p3109_is_nan(f, y)) return x;
  return p3109_minimum(f, x, y);
}

static inline p3109_t p3109_maximum_number(p3109_fmt_t f, p3109_t x, p3109_t y) {
  if (p3109_is_nan(f, x) && p3109_is_nan(f, y)) return p3109_nan_of(f);
  if (p3109_is_nan(f, x)) return y;
  if (p3109_is_nan(f, y)) return x;
  return p3109_maximum(f, x, y);
}

static inline p3109_t p3109_minimum_magnitude(p3109_fmt_t f, p3109_t x,
                                              p3109_t y) {
  if (p3109_is_nan(f, x) || p3109_is_nan(f, y)) return p3109_nan_of(f);
  p3109_t mx = x & p3109_mag_mask(f), my = y & p3109_mag_mask(f);
  if (mx < my) return x;
  if (mx > my) return y;
  return p3109_minimum(f, x, y);  // equal magnitude: break the tie by value
}

static inline p3109_t p3109_maximum_magnitude(p3109_fmt_t f, p3109_t x,
                                              p3109_t y) {
  if (p3109_is_nan(f, x) || p3109_is_nan(f, y)) return p3109_nan_of(f);
  p3109_t mx = x & p3109_mag_mask(f), my = y & p3109_mag_mask(f);
  if (mx > my) return x;
  if (mx < my) return y;
  return p3109_maximum(f, x, y);
}

static inline p3109_t p3109_minimum_magnitude_number(p3109_fmt_t f, p3109_t x,
                                                     p3109_t y) {
  if (p3109_is_nan(f, y)) return x;  // note: (NaN, NaN) falls through to NaN
  if (p3109_is_nan(f, x)) return y;
  return p3109_minimum_magnitude(f, x, y);
}

static inline p3109_t p3109_maximum_magnitude_number(p3109_fmt_t f, p3109_t x,
                                                     p3109_t y) {
  if (p3109_is_nan(f, y)) return x;
  if (p3109_is_nan(f, x)) return y;
  return p3109_maximum_magnitude(f, x, y);
}

static inline p3109_t p3109_minimum_finite(p3109_fmt_t f, p3109_t x, p3109_t y) {
  if (p3109_is_nan(f, x) && p3109_is_nan(f, y)) return p3109_nan_of(f);
  if (p3109_is_nan(f, x)) return y;
  if (p3109_is_nan(f, y)) return x;
  int ix = p3109_is_infinite(f, x), iy = p3109_is_infinite(f, y);
  if (ix && iy) return p3109_minimum(f, x, y);  // both infinite: ordinary min
  if (ix) return y;                             // one infinite: take the finite
  if (iy) return x;
  return p3109_minimum(f, x, y);
}

static inline p3109_t p3109_maximum_finite(p3109_fmt_t f, p3109_t x, p3109_t y) {
  if (p3109_is_nan(f, x) && p3109_is_nan(f, y)) return p3109_nan_of(f);
  if (p3109_is_nan(f, x)) return y;
  if (p3109_is_nan(f, y)) return x;
  int ix = p3109_is_infinite(f, x), iy = p3109_is_infinite(f, y);
  if (ix && iy) return p3109_maximum(f, x, y);
  if (ix) return y;
  if (iy) return x;
  return p3109_maximum(f, x, y);
}

// -----------------------------------------------------------------------------
// 4.16  NextGreaterThan, NextLessThan
// -----------------------------------------------------------------------------
// Walking to the neighbouring value is walking to the neighbouring code point,
// because the codes count upward in magnitude. The only care needed is at the
// four edges: NaN, infinity, the largest finite value, and the crossing from
// the smallest negative to zero.
//
// Unlike IEEE-754's nextUp, stepping past infinity gives NaN rather than
// staying put, which is why the standard uses different names.

static inline p3109_t p3109_next_greater_than(p3109_fmt_t f, p3109_t x) {
  p3109_t nan = p3109_nan_of(f);
  if (p3109_is_nan(f, x)) return nan;
  if (f.is_extended && p3109_is_infinite(f, x)) {
    // -Inf steps up to the most negative finite value; +Inf has nowhere to go.
    if (p3109_is_sign_minus(f, x)) return p3109_min_finite_of(f);
    return nan;
  }
  if (x == p3109_max_finite_of(f))
    return f.is_extended ? p3109_inf_of(f) : nan;
  if (p3109_is_sign_minus(f, x)) {
    // Moving up through the negatives means shrinking the magnitude; the
    // smallest negative steps to zero rather than to the NaN code point.
    if (x == (1u << (f.k - 1)) + 1u) return 0;
    return x - 1;
  }
  return x + 1;
}

static inline p3109_t p3109_next_less_than(p3109_fmt_t f, p3109_t x) {
  p3109_t nan = p3109_nan_of(f);
  if (p3109_is_nan(f, x)) return nan;
  if (!f.is_signed && x == 0) return nan;
  if (f.is_extended && p3109_is_infinite(f, x)) {
    if (p3109_is_sign_minus(f, x)) return nan;
    return p3109_max_finite_of(f);
  }
  if (x == p3109_min_finite_of(f) && f.is_signed)
    return f.is_extended ? p3109_neg_inf_of(f) : nan;
  if (x == 0) return (1u << (f.k - 1)) + 1u;  // zero steps to smallest negative
  if (p3109_is_sign_minus(f, x)) return x + 1;
  return x - 1;
}

#endif  // __P3109_H
