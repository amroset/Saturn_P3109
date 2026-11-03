# vfdot4: four-element dot product per FP32 lane

`vfdot4` multiplies the four 8-bit elements packed in each 32-bit lane of two
operands and adds the four products to the FP32 accumulator in that lane of
`vd`, with one rounding. In scaled mode ([scaled-mode.md](scaled-mode.md)) the
operands are P3109 blocks, their scales placed by the same rule as every other
scaled instruction. Custom extension, in configs with `P3109Formats(dot = true)`
(`P3109BlockDotV256D128ShuttleConfig` and its `Buf4` and `Shared` variants).

## Encoding

OP-V, funct6 = 0x35 (reserved in RVV 1.0), SEW = 32, any LMUL:

| Form | funct3 | Operands |
|---|---|---|
| `vfdot4.vv vd, vs2, vs1, vm` | OPFVV | lane i of vs2 and of vs1: four elements each |
| `vfdot4.vf vd, vs2, rs1, vm` | OPFVF | lane i of vs2; f[rs1] bits 31:0: four elements, the same for every lane |

Element j of a lane is in bits 8j+7:8j. `altfmt` selects the 8-bit format of
all elements (binary8p4 or binary8p3), as at SEW 8.

## Operation

For each active lane i, with the elements decoded by P3109's BlockDecode
(§5.4.1: zero scale gives zero, or NaN for an infinite or NaN element; scale
255 gives NaN):

```
vd[i] = round( vd[i] + Σ_{j<4} (sA · a[4i+j]) (sB · b[4i+j]) )
```

The sum is exact and rounded once, to binary32, in `frm`. A zero result is +0
(P3109 §4.8.2). `vd` is an FP32 accumulator, never a block.

## Scales

- Without scaled mode, sA = sB = 1.
- In scaled mode (`vtype` bit 11), vs2 is a block: lane i takes byte
  i >> log2(E) of the scale group that follows vs2's data group (vs2 + LMUL),
  as for every scaled instruction. In `.vv`, vs1 likewise (vs1 + LMUL).
- In `.vf`, f[rs1] bits 39:32 hold the scalar's scale: the four elements and
  their scale arrive together. A kernel packs them with integer instructions
  (for MX data: once per block for the scale, once per instruction for the
  elements).

The scales of all four elements of a lane are the same, so a block must hold
whole lanes: a block size that is a multiple of 4.

## Saturn implementation

- **Decode.** `FDOT4` (Instructions.scala), an FMA-family instruction at SEW 32
  that reads vd. At dispatch, in scaled mode, vs2 and vs1 (`.vv`) are block
  operands and vd is not (`FPDot4`, Backend.scala); hazards, scale reads and
  the scale buffers are then those of every scaled instruction.
- **Datapath** (`ScaledDotLane`, ScaledDot.scala), one per FP32 lane of an
  element group, in the FMA unit, with the FMA pipe's latency:
  1. Decode the eight elements (E5M3) and form the four products. Each is an
     integer multiple of 2^-34 below 2^32, so the sum is exact as a 69-bit
     integer.
  2. Convert the sum to a float with a 70-bit significand (exact) and add the
     two scale exponents; widen the accumulator to the same format (exact).
  3. Add them (HardFloat AddRawFN, unrounded).
  4. Round once to binary32.
  Specials (NaN, infinite products, 0 × Inf, Inf − Inf, zero and NaN scales)
  bypass the integer sum.
- **Tests.** `models/run_dot_check.sh`: the lane against exact rational
  arithmetic (`dot_vectors.py`).
