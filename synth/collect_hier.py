"""List the Verilog files that make up one module's hierarchy.

CIRCT writes one module per file in gen-collateral, named after the module, so
the hierarchy is found by following instantiations from the top file.

    python collect_hier.py <gen-collateral dir> <top module>
"""
import os
import re
import sys

gen, top = sys.argv[1], sys.argv[2]
files = {os.path.splitext(f)[0]: os.path.join(gen, f)
         for f in os.listdir(gen) if f.endswith((".sv", ".v"))}

# "  ModuleName instName (" or "  ModuleName #(...) instName ("
inst = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_$]*)\s+(?:#\s*\(.*?\)\s*)?[A-Za-z_][A-Za-z0-9_$]*\s*\(",
                  re.M)

seen, todo = set(), [top]
while todo:
    m = todo.pop()
    if m in seen:
        continue
    if m not in files:
        sys.exit(f"module {m} has no file in {gen}")
    seen.add(m)
    src = open(files[m]).read()
    for name in inst.findall(src):
        if name in files and name not in seen:
            todo.append(name)

for m in sorted(seen):
    print(files[m])
