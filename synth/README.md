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
| `sweep_table.py` | Tabulates the precision sweep's results (this branch only) |

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

## Precision sweep (this branch only)

The `P3109Sweep*V256D128ShuttleConfig` configs carry extra P3109 precisions
(see `models/README.md`). `sweep_table.py` puts their results side by side,
against the baseline `P3109Sweep34V256D128ShuttleConfig`.

1. Generate the Verilog of each sweep config:

   ```bash
   source env.sh
   for c in 34 234 345 3456 34567 234567 346; do
     make -C sims/verilator verilog CONFIG=P3109Sweep${c}V256D128ShuttleConfig
   done
   ```

2. Get the area and critical path of each config, then the area per module:

   ```bash
   cfgs=$(for c in 34 234 345 3456 34567 234567 346; do echo P3109Sweep${c}V256D128ShuttleConfig; done)
   generators/saturn/synth/run_all.sh $cfgs
   for c in $cfgs; do
     generators/saturn/synth/synth_hier.sh $c FPConvPipe
     generators/saturn/synth/synth_hier.sh $c FPFMAPipe
   done
   ```

3. Print the table:

   ```bash
   python3 generators/saturn/synth/sweep_table.py $cfgs
   ```

   The first config is the baseline. Each column gives the flattened area, its
   change against the baseline, the critical path, and the area per module
   group.
