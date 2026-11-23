"""Measured values not stored in the JSON files, with their sources."""
# opu-timing (notes: opu-scale-isa-design.md), cycles per 32-k block, 2x2-tile GEMM
# Q3/Q4/Q4b (notes/data/q4-opu-cache-2026-11-04, median of 8 layouts, all within +-0.1): unscaled
# double-buffered; pairing with the 4-entry buffer (fastest kernel: single-buffered, scale loads at block
# start) and double-buffered (thrashes); pairing with 8 entries, double-buffered; explicit emulation build
OPU = {"unscaled_db": 528.01, "unscaled_sb": 544.01, "ruleA_cached": 544.08, "ruleA": 560.02,
       "ruleA_db": 724.02, "ruleA_db8": 559.00, "explicit": 544.14, "software": 3008.0}
# Q4c (notes/data/q4c-opu-copies-2026-11-05, median of 8 layouts, range 550.59-550.76): pairing with 8
# entries, double-buffered, the second buffer's scales copied (4 vmv1r.v) instead of loaded
OPU["ruleA_db8_copy"] = 550.66
# Q4e (notes/data/q4d-opu-fp4-2026-11-10, median of 8 layouts, all within 0.1 except the copies: 286.04-287.17):
# the same kernels with FP4 pairs (16 instructions per block); explicit on the FP4Shared build, 8 entries on FP4S8
OPU_FP4 = {"unscaled_db": 264.02, "unscaled_sb": 272.04, "ruleA": 288.00, "ruleA_db": 372.02, "ruleA_db8": 294.94,
           "explicit": 280.01, "ruleA_db8_copy": 286.21}
OPU_MACS = 131072                       # multiply-adds per 32-k block (64 x 64 x 32)
OPU_AI = 31.03                          # multiply-adds per byte of operand traffic
BW, BW32 = 15.79, 11.83                 # B/cycle: LMUL-8 loads; 32-byte loads and stores, overlapped (vec-stream-bw, 16 KB)
# B2 (notes/data/b2-l2-bandwidth-2026-11-11): 32-byte loads alone, 256 KB (L2, as the attention kernel's keys); 6.32 at 16 KB
BW32LD = 7.31

# Synthesis, Sky130 HD, Yosys + ABC, cell area in um^2 (synth/runs; notes 2026-10-26 / 2026-10-27);
# FMA pipe with scaled mode: the widened lanes (Q10, 2026-11-05; separate lanes were 1078402)
VEC = {  # unit: (without scaled mode, with scaled mode)
    "FMA pipe": (670877, 788526), "FPConv": (223064, 255937), "Backend logic": (98438, 127008),
    "FP issue queues": (52480, 70090), "FP sequencer": (30278, 39443), "Integer sequencer": (34421, 46701)}
OPU_AREA = {  # 256 cells, sequencer, top level (scale cache)
    "cell": (47775, 48934), "sequencer": (18554, 22500), "top": (0, 54395)}
