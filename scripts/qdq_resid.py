#!/usr/bin/env python3
"""Find weight-bearing MatMul/Gemm nodes whose OUTPUT feeds a residual Add.

Round 2 of the explicit-Q/DQ experiment: the round-1 failure mode was that the
per-GEMM output Q/DQ pulled the whole residual stream into int8. This script
locates exactly those GEMMs so the inserter can skip their output-side Q/DQ.

Usage: qdq_resid.py <fp32.onnx> [--json out.json]
"""
import sys, json
from collections import Counter, OrderedDict
import onnx

SRC = sys.argv[1]
JSON_OUT = None
if "--json" in sys.argv:
    JSON_OUT = sys.argv[sys.argv.index("--json") + 1]

print("===== LOADING %s =====" % SRC, flush=True)
m = onnx.load(SRC)
g = m.graph
init_map = {i.name: i for i in g.initializer}
print("nodes=%d initializers=%d" % (len(g.node), len(g.initializer)), flush=True)

prod = {}
for n in g.node:
    for o in n.output:
        prod[o] = n

cons = {}
for n in g.node:
    for i in n.input:
        if i:
            cons.setdefault(i, []).append(n)


def resolve_weight(name, depth=0):
    if name in init_map:
        return name
    if depth > 6:
        return None
    n = prod.get(name)
    if n is None:
        return None
    if n.op_type == "Transpose":
        return resolve_weight(n.input[0], depth + 1)
    return None


PASSTHRU = {"Transpose", "Reshape", "Squeeze", "Unsqueeze", "Identity",
            "Cast", "Flatten", "Slice", "Concat", "Gather", "Expand"}


def walk_to_add(t, depth=0, seen=None):
    """Follow consumers through pass-through ops; return [(add_node, hops)]."""
    if seen is None:
        seen = set()
    if t in seen or depth > 6:
        return []
    seen.add(t)
    out = []
    for c in cons.get(t, []):
        if c.op_type == "Add":
            out.append((c, depth))
        elif c.op_type in PASSTHRU:
            for o in c.output:
                out.extend(walk_to_add(o, depth + 1, seen))
    return out


# ---------------------------------------------------------------- inventory
targets = []
actonly = []
for n in g.node:
    if n.op_type not in ("MatMul", "Gemm"):
        continue
    w = resolve_weight(n.input[1])
    (targets if w else actonly).append((n, w))
print("\nweight-bearing MatMul/Gemm: %d   activation-only: %d"
      % (len(targets), len(actonly)), flush=True)

# ------------------------------------------------------------- all Add nodes
adds = [n for n in g.node if n.op_type == "Add"]
print("\n===== ALL Add NODES (%d) =====" % len(adds), flush=True)


def describe(t):
    """short human description of a tensor: producer op + name"""
    p = prod.get(t)
    if p is None:
        if t in init_map:
            return "INIT:%s" % t
        return "INPUT:%s" % t
    return "%s(%s)" % (p.op_type, p.name)


for a in adds:
    ins = [describe(i) for i in a.input]
    use = [c.op_type for c in cons.get(a.output[0], [])]
    print("  %-58s in=[%s] -> %s" % (a.name, " | ".join(ins), use), flush=True)

# --------------------------------------------------- residual-GEMM detection
resid = OrderedDict()
nonresid = OrderedDict()
for n, w in targets:
    out = n.output[0]
    hits = walk_to_add(out)
    if hits:
        resid[n.name] = {
            "weight": w,
            "out": out,
            "adds": [{"add": a.name, "hops": d,
                      "other_inputs": [describe(x) for x in a.input if x != out]}
                     for a, d in hits],
        }
    else:
        nonresid[n.name] = {"weight": w, "out": out,
                            "consumers": sorted({c.op_type for c in cons.get(out, [])})}

print("\n===== RESIDUAL-FEEDING GEMMs (%d / %d) =====" % (len(resid), len(targets)), flush=True)
for k, v in resid.items():
    hops = ",".join("h%d" % a["hops"] for a in v["adds"])
    print("  %-62s out=%-58s -> %s" % (k, v["out"], hops), flush=True)
    for a in v["adds"]:
        print("        Add %s  other=%s" % (a["add"], a["other_inputs"]), flush=True)

print("\n===== NON-RESIDUAL TARGET GEMMs (%d) =====" % len(nonresid), flush=True)
for k, v in nonresid.items():
    print("  %-62s out=%-58s consumers=%s" % (k, v["out"], v["consumers"]), flush=True)

pref = Counter()
for k in resid:
    parts = k.strip("/").split("/")
    pref["/".join(parts[:3]) if len(parts) >= 3 else k] += 1
print("\n===== RESIDUAL GEMM PREFIX HISTOGRAM =====")
for k, v in sorted(pref.items()):
    print("  %4d  %s" % (v, k))

if JSON_OUT:
    with open(JSON_OUT, "w") as f:
        json.dump({"residual": resid, "non_residual": nonresid,
                   "n_targets": len(targets)}, f, indent=1)
    print("\n### wrote %s" % JSON_OUT)
print("### DONE", flush=True)
