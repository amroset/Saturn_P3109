# Scaled mode: block-scaled vector arithmetic and conversions

With `vtype` bit 11 set, the FP8 arithmetic and the FP8 ↔ BF16 conversions
take IEEE P3109 block scales. Each 8-bit register operand is then a block, its
scales in the next register, the same rule as `OPFMACC.S`
([opfmacc-s.md](opfmacc-s.md)). No instruction is added.

Implemented in configs with `P3109Formats.block` (`P3109BlockV256D128ShuttleConfig`,
`P3109BlockFiniteV256D128ShuttleConfig`), which give `vtype` six bits above
`altfmt` (`VTypeFmtBits` = 6). Elsewhere bits 14:11 are reserved and setting
one makes `vtype` illegal (`vill`).

## Setting the mode

| `vtype` bits | Field |
|---|---|
| 11 | scaled mode |
| 14:12 | log2 of E, the elements per scale byte: 1, 2, 4, …, 128 |

`vsetvli`'s immediate stops at bit 10, so use `vsetvl`, e.g. `vtype` = 0x58C0
for e8, m1, ta, ma, binary8p4, scaled, E = 32 (0x08C0 for E = 1). A
`vsetvli` or `vsetivli` writes zeros to bits 14:11 and so leaves scaled mode.

The bits only change the instructions listed below, at their SEW (8, or 32 for
`vfdot4`, [vfdot4.md](vfdot4.md)); at another SEW and for every other
instruction they are ignored, not illegal. A nonzero E with bit 11 clear is
ignored too, except by `OPFMACC.S`, which always uses it.

## The rule

In scaled mode, for the instructions below, every 8-bit vector operand vN is a block:

| Register | Contents |
|---|---|
| vN | `x[0..R-1]`, 8-bit elements |
| vN+1 | `s[0..R/E-1]`: byte j is the Binary8p1uf scale of elements jE to jE+E−1 |

With R = VLEN/8: element i takes byte i >> log2(E). Bytes past R/E are ignored;
with E ≥ R, byte 0 scales the whole register. A destination's scales are read,
not written: they are the result block's scale, an input (P3109 §5.7).

- At LMUL L, a block operand's data group vN..vN+L−1 (N a multiple of L) has
  its scales in the group after it, from vN+L (vN+1 at LMUL 1); element i of
  the group takes byte i >> log2(E) of that scale area. Fractional LMUL counts
  as 1.
- Illegal (Saturn raises an illegal-instruction exception): a block operand
  whose N is not a multiple of L; a scale group past v31 (N + 2L > 32); a
  destination group, wide or not, that overlaps any block operand's scale
  group, e.g. `vfadd.vv v9, v8, v10` or `vfwadd.vv v2, v1, v8` at LMUL 1. A
  source may be another operand's scale register (`vfadd.vv v8, v2, v3`): both
  only read it.
- Along the blocks (one scale per E ≥ 16 consecutive elements), LMUL > 1 lets
  one scale-register load serve several blocks (`benchmarks/vec-block-lmul`).
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
`X = x[i] × 2^(s[i>>log2 E]−128)` (scale 0: zero, or NaN for an infinite or NaN x;
scale 255: NaN):

```
Z = the instruction's operation on the decoded operands, exact
BF16 result:   r = round(Z)
8-bit result:  r = round(Z / 2^(sr−128)),  sr = the scale of element i in vd+1
```

- In the destructive forms the result block's scale is also an operand's:
  `vfmacc` and kin have s_c = s_r (vd is the addend), `vfmadd` and kin s_a = s_r
  (vd is a factor), so that operand enters unscaled.
- One rounding, in `frm`'s mode. An 8-bit result follows ωBlockProject (§5.4.2):
  `sr` = 255 gives NaN; `sr` = 0 gives zero for anything but NaN.
- `.sat` (ConvertToBlock) saturates as the unscaled `.sat` form does.
- A zero result is +0 in BF16 (P3109 §4.8.2).
- With every scale 2^0 (code 128), each instruction computes what it computes unscaled.

## Elements per scale byte and block size

E says how the elements of one register share scale bytes; it is not the
block size. A block runs along one dimension of a tensor (MX: 32 consecutive
k), and which register elements it covers depends on how the program walks
the tensor:

- Along the block (a quantizer along a row): the register holds whole blocks,
  and E is the block size (32 for MX), one byte per block.
- Across blocks (one element from each of R rows, as the OPU's operands and
  the quantizer producing them): every element is in a different block, E = 1,
  and the block continues across instructions: software keeps the same scale
  register for as many instructions as the block is long.

E = 1 is also P3109's scaled operations. Block sizes that are not powers of
two use E = 1 with the scale repeated. For elementwise operations each element
depends only on its own block's scale (§5.4.1, §5.4.2), so this is exact.

## The OPU

In scaled mode `OPFMACC` is `OPFMACC.S` ([opfmacc-s.md](opfmacc-s.md)); the
two encodings compute the same thing. Both honor E: row i of the tile takes
byte i >> log2(E) of vs1+1, column j byte j >> log2(E) of vs2+1. E = 1 (the
default) is a scale per row and column, as MX needs; the MX block runs along
k, across instructions. E = 32 gives one scale to the whole tile side, for
coarse scaling such as 128 × 128 weight blocks, and lets the OPU fill only the
scale register's first element group; E = 64 and 128 select the same byte.

## Saturn implementation

None of this is architectural:

- **Decode.** At dispatch (`Backend.scala`), an instruction of `ScaledMode.insns`
  with the bit set gets `scaled_vs1`, `scaled_vs2`, `scaled_vd`: which operands are blocks.
  The illegal register uses are caught earlier, in `EarlyDecode.scala`
  (`scaled_illegal`); `benchmarks/vec-block-illegal` checks them and their
  legal neighbours.
- **Hazards.** Each block operand adds its scale register (`rs|1`) to the
  instruction's read intents (`IssueQueue`, `ExecuteSequencer`).
- **Scale reads.** The execute sequencer reads the scale registers' element
  groups on the same three ports, in a cycle of their own, whenever an issue
  needs a group the backend does not hold (`scale_fire`). Each port keeps one
  entry per element group of a register across instructions, until a
  register-file write to it (`write_snoop`): a reused scale register costs no
  read. Otherwise: once per 8-bit element group with E = 1, once per LMUL-1
  instruction with E ≥ 2.
- **FPConv.** The scales reach the existing P3109 block datapath (`P3109Scale.decode`,
  `P3109Scale.project`), spread like the widening data.
- **FMA.** A scaled operand (up to 2^±142, and 2^±270 with the result scale
  folded in) fits none of the FMA's narrow lanes (exponents of 5 and 8 bits),
  so in scaled-mode configs every lane narrower than 10 exponent bits is
  widened to 10 (`SegmentedFMAPipe.laneType`: FP32 (10,24), FP16 (10,11), BF16
  (10,8), E5M3 (10,4); FP64 keeps its 11). Widening is exact and each lane
  still rounds once, so unscaled results and latency are unchanged. The scales
  are added to the widened operands' exponents, the result scale subtracted,
  and the exact result rounded once (`scaleRec`).
- **Tests.** `benchmarks/vec-block` runs 17 instructions in scaled mode
  (`vfadd`, `vfsub`, `vfmul`, `vfmacc`, `vfnmsac`, `vfmadd`, `vfmul.vf`,
  `vfrsub.vf`, `vfwadd`, `vfwsub`, `vfwmul`, `vfwmacc`, `vfwmul.vf`,
  `vfwadd.wv`, both conversions and `.sat`) with E = 1 in both 8-bit formats,
  six of them in all five rounding modes, and each again with E = 2, 8 and 32,
  the unused scale bytes random: 133 tests of 128 elements, against a gfloat
  reference (`gen_data/gen_data.py`) built on `fma_ref`'s exact arithmetic.
  `benchmarks/vec-block-cover` adds 49 tests of 256 elements on the same
  reference: eight of these instructions at LMUL 2 and 4 with E = 1 and 32,
  three at LMUL 4 with E = 2 (scales in the second scale register), `vfmul.vf`
  at LMUL 8, eight masked (mask undisturbed) and three with a tail (vl 21 of
  32, tail undisturbed), inactive and tail elements checked unchanged. A
  vector instruction with vstart ≠ 0 outside the memory pipeline raises an
  illegal-instruction exception (`PipelinedFaultCheck`), scaled or not.
- **Cost** (`benchmarks/vec-block-timing`, cycles per LMUL-1 instruction):
  one cycle per scale read, none when the buffer holds the group. FP8 → FP8:
  2 unscaled, 4 with E = 1, 3 with E ≥ 2 (destinations rotating); widening and
  conversions with reused scale registers: 4, as unscaled.
- **Against software** (`benchmarks/vec-block-kernels`): dequantize and
  quantize ≈ 2× faster, GEMV 1.16×, all bit-identical.
