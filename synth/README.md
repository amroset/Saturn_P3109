# Synthesis of the vector FP units

## Scope

These scripts synthesize two Saturn units onto the SkyWater Sky130 standard
cells (`sky130_fd_sc_hd`, typical corner):

- `FPConvPipe`, the conversion unit;
- `FPFMAPipe`, the multiply-add unit.

Use them to compare the area of two builds, for example
`MXV256D128ShuttleConfig` (OCP FP8) against `P3109V256D128ShuttleConfig`.

The results are cell area and the worst register-to-register path. There is no
place and route: the area is cell area, not die area, and the timing uses the
library's wire-load model. Between similar builds, the critical path varies by
about 1 ns for no design reason. Do not compare timing below that.

## Files

| File | Contents |
|---|---|
| `gen_verilog.sh` | Writes a config's Verilog in a form that Yosys can read |
| `synth_unit.sh` | Synthesizes one unit, flattened: area and critical path |
| `synth_hier.sh` | Synthesizes one unit with its hierarchy kept: area per module |
| `run_all.sh` | Runs `synth_unit.sh` on both units of each config, in parallel |
| `breakdown.py` | Compares the per-module areas of two configs |

The scripts write to `verilog/` and `runs/`, which git ignores. The unit's
module files are found with `../models/collect_hier.py`.

## How to run

Run all commands from the Chipyard root directory.

### 1. Install the tools (once)

The scripts look for the tools at these paths. To use other paths, set the
environment variable.

| Tool | Default path | Variable |
|---|---|---|
| Yosys | `~/.conda-yosys/bin/yosys` | `YOSYS` |
| OpenROAD | `~/.conda-openroad/bin/openroad` | `OPENROAD` |
| Sky130 cells | `~/.conda-sky130/share/pdk/sky130A/libs.ref/sky130_fd_sc_hd` | `PDK` |
| firtool | `.conda-env/riscv-tools/bin/firtool` | `FIRTOOL` |
| Python | `.conda-env/bin/python` | `PYTHON` |

### 2. Generate each config's Verilog

```bash
source env.sh
make -C sims/verilator verilog CONFIG=MXV256D128ShuttleConfig
make -C sims/verilator verilog CONFIG=P3109V256D128ShuttleConfig
```

### 3. Get the area and critical path of each unit

```bash
generators/saturn/synth/run_all.sh MXV256D128ShuttleConfig P3109V256D128ShuttleConfig
```

The multiply-add unit takes the longest, tens of minutes. Pass: one line per
config and unit, with cells, area and critical path. A failed job prints
`FAILED` and the name of its log, and the script exits non-zero.

### 4. See where the area changed (optional)

```bash
generators/saturn/synth/synth_hier.sh MXV256D128ShuttleConfig FPConvPipe
generators/saturn/synth/synth_hier.sh P3109V256D128ShuttleConfig FPConvPipe
python3 generators/saturn/synth/breakdown.py FPConvPipe MXV256D128ShuttleConfig P3109V256D128ShuttleConfig
```

`breakdown.py` lists each module's area in both configs and the difference.
Its total is before buffering and without optimization across modules, so it
is a little larger than the area from step 3.
