# Running the P3109 tests and experiments

This fork adds IEEE P3109 arithmetic, block scaling, FP4 and `vfdot4` to Saturn.
This page lists every test and experiment behind the thesis and the paper, and
how to run each one. The design itself is described in `docs/` (`scaled-mode.md`,
`opfmacc-s.md`, `vfdot4.md`, `fp4-vector.md`).

| Directory | Contents |
|---|---|
| `models/` | Exact references, unit checks of single RTL modules, accuracy studies |
| `benchmarks/` | Programs for the whole chip: correctness tests and timing benchmarks |
| `eval/` | Scripts that build, run and summarize the timing experiments; figure scripts and their data; reference outputs |
| `synth/` | Synthesis flow (Sky130) |

## 1. Setup

```bash
git config --global url."https://github.com/".insteadOf git@github.com:   # submodule URLs are SSH
git clone -b dotprod https://github.com/amroset/Chipyard_P3109.git
cd Chipyard_P3109
./build-setup.sh riscv-tools        # Chipyard's setup: conda env, submodules, toolchain
source env.sh                       # in every new shell

python3.12 -m venv ~/venvs/gfloat   # Python reference (gfloat needs Python >= 3.10)
~/venvs/gfloat/bin/pip install -r generators/saturn/benchmarks/common-data-gen/requirements.txt numpy matplotlib
```

The scripts look for that interpreter at `~/venvs/gfloat/bin/python`; set
`GFLOAT_PYTHON` to use another. Synthesis needs Yosys, OpenROAD and the Sky130
cells (`synth/README.md`, step 1). Run every command below from the Chipyard
root.

Most commands take a Chipyard config. Build what they need first:

```bash
make -C sims/verilator verilog CONFIG=<CONFIG>   # Verilog only: enough for unit checks and synthesis
make -C sims/verilator CONFIG=<CONFIG>           # Verilator simulator: needed for chip-level runs
```

Config names below omit `V256D128ShuttleConfig` (OPU configs: `OPUV256D128…ShuttleConfig`);
all are in `chipyard/SaturnConfigs.scala` and `chipyard/OPUConfigs.scala`.

The results were produced with these versions. Others should give the same
cycle counts as long as the compiler is the same: the compiler sets the code layout.

| Tool | Version |
|---|---|
| riscv64-unknown-elf-gcc | 13.2.0 (Chipyard's conda env) |
| Verilator | 5.022 |
| Python, gfloat, numpy | 3.12.4, 0.5.2, 2.5.3 |
| Yosys, OpenROAD | 0.27+3, 2.0-7070-g0264023b6 |
| Sky130 (open_pdks) | 1.0.457, `sky130_fd_sc_hd`, typical corner |

On a host with glibc newer than Chipyard's default sysroot (2.34), set
`sysroot_linux-64=2.39` in `conda-reqs/chipyard-base.yaml` before `build-setup.sh`.

## 2. Unit checks (`models/`)

Each check drives one module of the chip's generated Verilog with every input
or nearly every input, compares codes and exception flags with exact
references, and ends with `TOTAL MISMATCHES: 0` on success (non-zero exit
otherwise). `models/README.md` has the details.

| Command | Configs | Checks | Time (— not measured) |
|---|---|---|---|
| `validate_rounder.py`, `validate_rto.py`, `validate_flags.py`, `validate_rounder_fma.py` | none (Python) | Python model of the rounder: every BF16 input, round-to-odd, flags, FMA-shaped inputs | minutes |
| `cc -O2 -o /tmp/t generators/saturn/models/p3109_test.c && /tmp/t` | none | `benchmarks/common/p3109.h` helpers | seconds |
| `run_rounder_check.sh <dir>` | none (builds wrappers) | Rounder RTL, both domains, conversion and FMA inputs | 5 min |
| `run_conv_check.sh <CONFIG> <dir>` | P3109, P3109Finite, P3109Block, P3109BlockFinite, P3109BlockDot, P3109BlockDotFP4 | Conversion unit, block conversions with every scale, FP4 | 10 min each |
| `run_fma_check.sh <CONFIG> <dir>` | P3109Block, P3109BlockFinite, P3109BlockDotFP4 | FMA unit in scaled mode: every code pair, sampled accumulators and scales, five modes | — |
| `run_dot_check.sh <CONFIG> <dir>` | P3109BlockDot | One `vfdot4` lane | — |
| `run_opu_check.sh <CONFIG> <dir>` | OPU P3109, P3109Finite, P3109FP4, Mx (the cell takes scales in every P3109 build) | One OPU cell: every code pair (or FP4 quadruple), special accumulators, scales | 1 min each |
| `run_sweep_check.sh <CONFIG> <dir>` | P3109Sweep*, P3109SweepFP4, P3109 | Extra precisions; FP4 on the units' pins | — |

Run the Python validators as `$GFLOAT_PYTHON generators/saturn/models/<script>`
and the others as `generators/saturn/models/run_*.sh`.

## 3. Chip-level tests (`benchmarks/`)

Each program runs on the simulated chip and compares every result with a
reference computed by its generator, then prints `passed` (or `PASSED`) or
the elements that differ. Build with `make -C generators/saturn/benchmarks <name>.riscv`,
or with `generators/saturn/eval/build.sh <name> <out.riscv> <flags>` when
flags are needed, and run with `generators/saturn/eval/run_sims.sh <CONFIG> <binary>`.
The checked-in `data.S` is for a P3109 build. For other builds or options,
regenerate it with the command in the table, run inside the benchmark's `gen_data/`
with `$GFLOAT_PYTHON`.

| Benchmark | Configs | Data / flags | Checks |
|---|---|---|---|
| `vec-mx-unary`, `vec-mx-binary`, `vec-mx-fma` | MX, P3109, P3109Finite | run all steps with `generators/saturn/benchmarks/run_fp8_test.sh <name> [ocp\|p3109\|p3109-finite]` | Unscaled conversions, add/sub/mul, FMA and `.vf` forms, five modes |
| `vec-block` | P3109Block | `gen_data.py --std p3109 -o ..` | 17 instructions in scaled mode, E = 1, 2, 8, 32: 133 tests |
| `vec-block-cover` | P3109Block | same | LMUL 2, 4, 8, masks, tails: 49 tests |
| `vec-block-illegal` | P3109Block | none | Register uses that must trap, and their legal neighbours: 27 cases |
| `vec-dot4` | P3109BlockDot; P3109BlockDotShared with `-DSHARED` | `gen_data.py -o ..` | `vfdot4.vv` and `.vx`, both formats, five modes, E, LMUL, masks, buffer |
| `vtype-fmt` | OPU P3109FP4, and a config without FP4 | none | `vtype` element-format bits kept, or `vill` |
| `opu-scaled-gemm` | OPU P3109Block, P3109, P3109Finite, Mx | `gen_data.py --std <std> ../data.S`; scaled mode: `--coarse` and `-DSCALEDMODE`; FP4: `--fp4` and `-DFP4` on OPU P3109FP4 | 64×64×96 scaled GEMMs, scale-cache eviction, `vs1 = vs2`, coarse scales, FP4 |
| `opu-fp8-gemm` | OPU P3109 | `gen_data.py --std p3109 -o ../data.S` | Unscaled FP8 GEMM |
| `opu-layer` | OPU P3109Block | `-DSMALL -DCHECK` | The transformer layer at reduced size: every intermediate of the hardware pass against the software pass |

## 4. Timing experiments

Verilator runs are deterministic, but a loop's cycle count depends on its
address. Every timing result is therefore measured at eight code layouts and
reported as median and range; a difference counts only if the two ranges do
not overlap. Three scripts in `eval/` do this:

```bash
B=generators/saturn/eval
$B/build_layouts.sh vec-block-explicit runs/a1b ruleA  -DREPS=4           # 8 binaries
$B/build_layouts.sh vec-block-explicit runs/a1b shared -DREPS=4 -DSHARED
JOBS=8 $B/run_sims.sh P3109BlockDot       runs/a1b/ruleA-l*.riscv
JOBS=8 $B/run_sims.sh P3109BlockDotShared runs/a1b/shared-l*.riscv
python3 $B/summarize.py pairing='runs/a1b/ruleA-l*.log' explicit='runs/a1b/shared-l*.log' \
        --compare pairing:ruleA explicit:shared
```

`build_layouts.sh` uses `METHOD=aligned` by default; results marked *pad*
below need `METHOD=pad`. `summarize.py` reads the `RESULT` lines of most
benchmarks, `vec-block-timing`'s table, and fits `opu-timing`'s cycles per
iteration (as `benchmarks/opu-timing/fit.py`). Each benchmark's header lists
its kernels and options.

| Result | Benchmark, flags | Configs |
|---|---|---|
| fig. placement-cost, fig. roofline (vector) | `vec-block-explicit -DREPS=4`, `vec-dot4-gemm`, `vec-dot4-attn`; explicit build adds `-DSHARED`; circles add `-DNOCOPY` | P3109BlockDot, P3109BlockDotBuf4 (larger buffer); P3109BlockDotShared (explicit) |
| fig. placement-cost, fig. roofline (OPU), tab. fp4-opu | `opu-timing -DRULES=6..9`, *pad* (`main.c` says what each prints) | OPU P3109Block, P3109BlockS8, P3109BlockShared, P3109FP4, P3109FP4S8, P3109FP4Shared |
| fig. compute-micro | `vec-block-timing` | P3109BlockDot, P3109BlockDotBuf4, P3109BlockDotShared |
| fig. results-along | `vec-block-lmul`, `-DSHARED` for explicit, *pad* | P3109Block, P3109BlockShared |
| tab. sw-baseline | `opu-timing -DRULES=10 [-DSB1B \| -DSB1C]`, *pad* | OPU P3109Block |
| fig. layer, tab. layer-sizes | `opu-layer` (full size); `-DH=256 -DKV=64 -DF=1408`, `-DH=448 -DKV=64 -DF=2432`; block sizes `-DGEMMONLY -DBS=16\|32\|64\|128` | OPU P3109Block |
| tab. bandwidth | `vec-stream-bw -DNB=16384`, `-DNB=262144` | P3109BlockDot, P3109BlockDotLiq8 |
| Scale-buffer sweep (prose) | as placement-cost; OPU `-DRULES=7` | P3109BlockDotBuf2/4/8; OPU P3109BlockS2/S8/S16 |
| Deeper load queue (prose) | as placement-cost | P3109BlockDotLiq8, P3109BlockDotSharedLiq8, P3109BlockDotBuf4Liq8 |
| Copies and dispatch (prose) | `vec-block-explicit -DWINDOWS`, simulator flag `+dtrace`; `-DSCHED` | P3109BlockDotTrace, P3109BlockDotSharedTrace, P3109BlockDotIQ12, P3109BlockDotSharedIQ12 |
| Misses as throttling (prose) | as placement-cost | P3109BlockDotSharedForceMiss |

## 5. Synthesis

`synth/README.md` gives the flow and the recipes for the rounder comparison
(MX against P3109) and the precision sweep. Areas compare two configs that
differ in one feature, for example P3109 against P3109Block (scaled mode),
P3109BlockDot against P3109BlockDotFP4 (FP4), or `OuterProductCell` of OPU
Mx, P3109Block and P3109FP4. Synthesize the two configs in the same batch, and
compare critical paths only beyond 1 ns, which is how much they vary between
similar builds.

## 6. Accuracy studies

Pure Python. `eval/reference/<script>.txt` holds each script's output as
reported, and the same command reproduces it byte for byte, for example
`$GFLOAT_PYTHON generators/saturn/models/accum_study.py | diff - generators/saturn/eval/reference/accum_study.txt`.

| Result | Command | Time |
|---|---|---|
| tab. accuracy (rounding error in GEMMs) | `$GFLOAT_PYTHON generators/saturn/models/accum_study.py` | 1 min |
| tab. extreme (scaled instructions against software scaling) | `$GFLOAT_PYTHON generators/saturn/models/scaled_accuracy.py` | 15 s |
| tab. kappa (accuracy of OPFMACC.S as BlockDotProduct) | `$GFLOAT_PYTHON generators/saturn/models/opu_kappa.py` | 10 s |

## 7. Figures

`eval/figures/` has the scripts of the thesis and paper figures and the data
they read (`data/*.json`, described in `data/README.md`; OPU and area numbers
are in `constants.py`). `python3 generators/saturn/eval/figures/<name>.py`
writes `generators/saturn/eval/figures/out/<name>.pdf`.

## 8. Results this commit does not reproduce exactly

These were measured on code that has since changed. Rerunning them here gives
the current design's numbers, which may differ slightly.

| Result | Measured on |
|---|---|
| fig. placement-cost, fig. roofline | RTL with separate 8-bit lanes, since replaced by the folded FMA; a rerun on this commit's configs (the buffer sweep) agrees within 0.1 points |
| fig. compute-micro, fig. results-along | Simulators of 2026-10-24/25, before label alignment; compute-micro reproduced by the buffer sweep |
| fig. area, tab. cost | Several synthesis runs, one of a removed config; OPU cell before the exact-product adder (−13.7% since) |
| tab. rounder-area, tab. sweep | Branch `p3109` (2026-10-03), before the rounder took FP4 |
| tab. sw-baseline, tab. opu-baseline, tab. opu-rules (rows A, B, strided) | One layout each, earlier code |
| OPU timing tables | OPU cell before the exact-product adder; cycle counts identical where rerun |
