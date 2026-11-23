# Data behind the figures

The JSON files were exported from the raw simulation logs, which are not in this repository (`notes/scripts/export_paper_data.py` below refers to the authors' copy). `eval/summarize.py` computes the same medians and ranges from new runs (see `REPRODUCE.md`).

- `sweep3-summary.json`: RTL-simulation results of 2026-10-26 (8 code layouts each; median, min, max cycles). Keys: `explicit` (10 kernels, LMUL 1; labels ruleA, shared = explicit best case, unscaled, scalar, baseline = software), `lmul` (along-the-blocks kernels at LMUL 1, 2, 4), `unroll` (N sweep).
- `timing-summary.json`: compute-only microbenchmark (vec-block-timing), cycles per instruction, rule A and explicit builds.
- OPU numbers (opu-timing, cycles per 32-k block of a 2x2-tile GEMM, one layout) and synthesis areas (Sky130, Yosys + ABC) are in `figscripts/constants.py`, with their sources.
- `across-m3.json`: inter-block kernels (M3, 2026-11-05), 8 layouts, every array at the same address in every build, per design point including the four-register-buffer build; exported from the raw logs by `notes/scripts/export_paper_data.py` (fig. placement-cost; formerly fig. results-across).
- `across-e12.json`: the same kernels from E12 (2026-11-03), before data placement was pinned; superseded by `across-m3.json`, no longer used.
- `dot.json`: vfdot4 kernels (M3, 2026-11-05), 8 layouts, same data placement in every build, with per-layout cycles (fig. placement-cost; formerly fig. dot-results).
- `layer.json`: Qwen2.5-0.5B layer, hardware vs software scaling, three sizes (Q1 method, software GEMM of SB1, 2026-11-14) (fig. layer).
- `across-a1b.json`: inter-block element-wise kernels (A1b, 2026-11-10): as M3, with every loop head aligned to 16 bytes in every binary (fig. placement-cost, fig. roofline); supersedes `across-m3.json`.
- `dot-a1.json`: vfdot4.vx kernels (A1, 2026-11-10), loops aligned; keys kernel|build|variant (fig. placement-cost, fig. roofline); supersedes `dot.json`.
- `attn-v1b.json`: vfdot4.vv attention scores (V1b, 2026-11-11), four accumulators, loops aligned (fig. placement-cost, fig. roofline).
- `c1-nocopy.json`: the pairing binaries of A1b, A1 and V1b without their scale copies, on both pairing builds (C1, 2026-11-11); timing only, scaled results wrong by design (fig. placement-cost, circles).
