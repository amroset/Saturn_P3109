# FP4 in the vector unit

Status: the units are done and verified (rounder, readers, block conversions, the FMA
lanes, unscaled and scaled). Instructions and decode are planned.

P3109's conformance set (Interim Report v4.0.3, §4.5) has one 4-bit format,
Binary4p2sf, beside the two 8-bit formats. Its operations (Convert, Add,
Multiply, FMA, ...) are defined over the union of the two sets. Each result is
rounded once to its own format. The OPU already multiplies FP4 pairs (see
[opfmacc-s.md](opfmacc-s.md#fp4-pairs-element-format-code-6)). This page covers
the vector unit.

## The format

Binary4p2sf has 4 bits, is signed, is in the finite domain, has precision 2
and has bias 2.

| Code | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9..F |
|---|---|---|---|---|---|---|---|---|---|---|
| Value | 0 | 0.25 | 0.5 | 0.75 | 1 | 1.5 | 2 | 3 | NaN | −(code − 8) |

Every value is exact in binary8p4 and in binary8p3. Its field widths are
OCP E2M1's, but the bias differs by one, and the code for −0 is NaN.

## Format code

The element-format code is `vtype[10:9] ## altfmt`:

| Code | Format |
|---|---|
| 0 | binary8p4 |
| 1 | binary8p3 |
| 6 | Binary4p2sf |

Code 6 is the code the OPU uses for FP4 pairs. In `P3109Formats`, `fp4 = true`
adds the format at code 6.

## Storage

- **Packed**, two codes per byte with the low nibble first, in memory and for
  the OPU.
- **Unpacked**, one code in the low nibble of a byte with the high nibble
  zero, for element-wise vector arithmetic at SEW 8.

RVV has no SEW 4. Element-wise arithmetic on packed nibbles would need lanes
of its own.

## Rounding and reading

Every FP4 operand is exact in FP8. So an operation with FP4 inputs could read
them from FP8. What FP8 cannot provide is an FP4 result: rounding to FP8 and
then to FP4 rounds twice. The hardware change is therefore in the rounder.

- `P3109Rounder` takes each format's width K from `P3109Fmt(k, p, finite)`.
  The exponent width (K − P), the bias (2^(K−1−P)), and the sign, NaN and
  maximum codes all follow from K.
  - A 4-bit result is written in the low nibble.
  - For 8-bit formats every constant is unchanged, so the pair's datapath is
    unchanged.
  - Binary4p2sf has precision 2, so it moves the round mask by
    `precShift = 2` and `delta = −6`.
- `p3109ToCore` reads a 4-bit format from the low nibble. E5M3 holds every
  Binary4p2sf value, so the 8-bit FMA core does not change.

## Instructions (planned)

| Instruction | Operation |
|---|---|
| `vfext.vf2` | Packed FP4 to FP8, exact. Zvfofp4min v0.1 encoding: VXUNARY0, vs1 = 10110, SEW 8. Zvfofp4min writes OCP E4M3; here `altfmt` selects binary8p4 or binary8p3. |
| Narrowing inverse (VXUNARY0, vs1 = 10111) | FP8 to packed FP4, rounded once (`frm`), flags set. A floating-point instruction for FS. |
| `vfwcvt.f.f.v`, `vfncvt.f.f.w` at SEW 8, code 6 | Unpacked FP4 to and from BF16. In scaled mode these are ConvertFromBlock and ConvertToBlock. |
| `vfadd`, `vfmul`, `vfmacc`, ... at SEW 8, code 6 | Read unpacked FP4 and round to unpacked FP4. Widening forms write BF16. |

## Verification

`models/run_sweep_check.sh P3109SweepFP4V256D128ShuttleConfig` drives format
code 6 on the pins of FPConvBlock and SegmentedFMAPipe. It covers:

- every FP4 code widened to BF16;
- every BF16 pattern narrowed to FP4, under the five rounding modes and
  round-to-odd, each with SatNone and SatFinite;
- every FP4 operand pair through multiply, add and subtract.

Each result is compared with gfloat (`p3109_k4p2sf`), `rto_ref.py` and
`flags_ref.py`, codes and exception flags both. The same run checks the 8-bit
pair as a regression.
