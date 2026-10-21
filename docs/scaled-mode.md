# Scaled mode: block-scaled vector arithmetic and conversions

With `vtype` bit 11 set, the FP8 arithmetic and the FP8 ↔ BF16 conversions
take IEEE P3109 block scales. Each 8-bit register operand is then a block, its
scales in the next register, the same rule as `OPFMACC.S`
([opfmacc-s.md](opfmacc-s.md)). No instruction is added.

Implemented in configs with `P3109Formats.block` (`P3109BlockV256D128ShuttleConfig`,
`P3109BlockFiniteV256D128ShuttleConfig`), which give `vtype` three bits above
`altfmt` (`VTypeFmtBits` = 3). Elsewhere bit 11 is reserved and setting it makes
`vtype` illegal (`vill`).

## Setting the mode

`vtype` bit 11 = 1. `vsetvli`'s immediate stops at bit 10, so use `vsetvl`, e.g.
`vtype` = 0x8C0 for e8, m1, ta, ma, binary8p4 (0x9C0 for binary8p3).

## The rule

In scaled mode, for the instructions below, every 8-bit vector operand vN is a block:

| Register | Contents |
|---|---|
| vN | `x[0..R-1]`, 8-bit elements |
| vN+1 | `s[0..R-1]`: byte i is the Binary8p1uf scale of element i |

With R = VLEN/8. A destination's scales are read, not written: they are the
result block's scale, an input (P3109 §5.7).

- vN must be even, and LMUL 1. Other settings are reserved; Saturn does not trap them.
- A scalar (`.vf`) operand and a BF16 operand are not blocks: their scale is 1.
- Scale codes: 0 is zero, 255 is NaN, c is 2^(c−128) (as for `OPFMACC.S`).

## Instructions (SEW 8)

| Instructions | Blocks | P3109 operation |
|---|---|---|
| `vfadd`, `vfsub`, `vfrsub`, `vfmul` (.vv, .vf) | vs2, vs1 (.vv), vd | BlockAdd, BlockSubtract, BlockMultiply (§5.7) |
| `vfmacc`, `vfnmacc`, `vfmsac`, `vfnmsac`, `vfmadd`, `vfnmadd`, `vfmsub`, `vfnmsub` | vs2, vs1, vd | BlockFMA (§5.7) |
| `vfwadd`, `vfwsub`, `vfwmul`, `vfwmacc` and kin (.vv, .vf) | vs2, vs1 | ScaledAdd, ScaledSubtract, ScaledMultiply (§5.8), result BF16 |
| `vfwadd.w`, `vfwsub.w` | vs1 | the same, vs2 being BF16 |
| `vfwcvtbf16.f.f.v` | vs2 | ConvertFromBlock to BF16 (§5.5.1) |
| `vfncvtbf16.f.f.w`, `.sat` | vd | ConvertToBlock from BF16 (§5.5.2) |

Every other instruction ignores the mode: reductions (they combine elements),
compares, min/max, sign injection, moves, other conversions.

## Operation

For element i, with each block operand decoded by P3109's ωBlockDecode (§5.4.1),
`X = x[i] × 2^(s[i]−128)` (scale 0: zero, or NaN for an infinite or NaN x;
scale 255: NaN):

```
Z = the instruction's operation on the decoded operands, exact
BF16 result:   r = round(Z)
8-bit result:  r = round(Z / 2^(sr−128)),  sr = the scale of element i in vd+1
```

- One rounding, in `frm`'s mode. An 8-bit result follows ωBlockProject (§5.4.2):
  `sr` = 255 gives NaN; `sr` = 0 gives zero for anything but NaN.
- `.sat` (ConvertToBlock) saturates as the unscaled `.sat` form does.
- A zero result is +0 in BF16 (P3109 §4.8.2).
- With every scale 2^0 (code 128), each instruction computes what it computes unscaled.

## Block size

The hardware sees one scale per element, so the block size is software's:
any B, by writing a block's scale into its B bytes (one `vmv.v.x` for B ≥ R, a
`vrgather` from a compact array for smaller B, B = 1 for P3109's scaled
operations). For elementwise operations this is exact: each element depends
only on its own scale (§5.4.1, §5.4.2).

## Saturn implementation

None of this is architectural:

- **Decode.** At dispatch (`Backend.scala`), an instruction of `ScaledMode.insns`
  with the bit set gets `scaled_vs1`, `scaled_vs2`, `scaled_vd`: which operands are blocks.
- **Hazards.** Each block operand adds its scale register (`rs|1`) to the
  instruction's read intents (`IssueQueue`, `ExecuteSequencer`).
- **Scale reads.** The execute sequencer reads the scale registers' element
  groups in a cycle before each issue, on the same three ports; the backend
  holds them until the data issues (`scale_fire`). A scaled instruction
  therefore takes two cycles per issue.
- **FPConv.** The scales reach the existing P3109 block datapath (`P3109Scale.decode`,
  `P3109Scale.project`), spread like the widening data.
- **FMA.** A scaled operand (up to 2^±142) fits none of the FMA's lanes
  (exponents of 5 and 8 bits), so scaled instructions run on separate lanes,
  one per element (`ScaledFMA.scala`): a recoded format with a 10-bit exponent
  and an 8-bit significand, wide enough to hold every scaled operand exactly.
  The scales are added to the operands' exponents, the result scale subtracted,
  and the exact result rounded once.
- **Tests.** `benchmarks/vec-block` runs 17 instructions in scaled mode
  (`vfadd`, `vfsub`, `vfmul`, `vfmacc`, `vfnmsac`, `vfmadd`, `vfmul.vf`,
  `vfrsub.vf`, `vfwadd`, `vfwsub`, `vfwmul`, `vfwmacc`, `vfwmul.vf`,
  `vfwadd.wv`, both conversions and `.sat`), in both 8-bit formats, six of them
  in all five rounding modes: 82 tests of 128 elements, against a gfloat
  reference (`gen_data/gen_data.py`) built on `fma_ref`'s exact arithmetic.
