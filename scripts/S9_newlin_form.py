"""Exact form of the newlin mask subgraph: node-by-node chain with literal values.

Prints, for the tensors that feed the encoder layers' attention mask:
  - the producing chain (op_type + name) with every Constant/Initializer value
  - consumer counts
Also prints where each layer's attn mask input comes from, so the shared-vs-per-layer
question is answered from the graph rather than from assumption.
"""
import argparse
import collections

import numpy as np
import onnx
from onnx import numpy_helper

ap = argparse.ArgumentParser()
ap.add_argument("--onnx", required=True)
ap.add_argument("--depth", type=int, default=12)
ap.add_argument("--weights", action="store_true", help="also print ops that carry weights (slow)")
ap.add_argument("--target", action="append", default=None)
a = ap.parse_args()

m = onnx.load(a.onnx, load_external_data=False)
g = m.graph
prod = {}
for n in g.node:
    for o in n.output:
        prod[o] = n
cons = collections.defaultdict(list)
for n in g.node:
    for i in n.input:
        cons[i].append(n)
inits = {i.name: i for i in g.initializer}
gin = {i.name for i in g.input}

HOP = {"Where", "Mul", "Sub", "Add", "Cast", "Expand", "Unsqueeze", "Reshape", "Constant",
       "Transpose", "Concat", "Mul_1", "Slice", "ConstantOfShape", "Equal", "Not", "And",
       "Cast_1", "Squeeze", "Tile", "Range", "Less", "Greater", "Floor", "Neg", "Div"}


def fmt(t, lim=6):
    n = prod.get(t)
    if n is not None and n.op_type == "Constant":
        for at in n.attribute:
            if at.name == "value":
                arr = numpy_helper.to_array(at.t)
                fl = arr.reshape(-1)
                return "Constant shape=%s dtype=%s first=%s" % (
                    tuple(arr.shape), arr.dtype,
                    [float(v) for v in fl[:lim]] if arr.dtype.kind == "f" else fl[:lim].tolist())
            if at.name in ("value_float", "value_floats", "value_int", "value_ints"):
                v = at.f if at.name == "value_float" else (
                    list(at.floats) if at.name == "value_floats" else (
                        at.i if at.name == "value_int" else list(at.ints)))
                return "Constant %s=%r" % (at.name, v)
        return "Constant(no value attr)"
    if t in inits:
        arr = numpy_helper.to_array(inits[t])
        fl = arr.reshape(-1)
        return "Initializer shape=%s dtype=%s first=%s" % (
            tuple(arr.shape), arr.dtype,
            [float(v) for v in fl[:lim]] if arr.dtype.kind == "f" else fl[:lim].tolist())
    if t in gin:
        return "GRAPH_INPUT"
    if n is not None:
        return "%s name=%s" % (n.op_type, n.name)
    return "unresolved"


def walk(t, d=0, seen=None, maxd=12):
    if seen is None:
        seen = set()
    if t in seen or d > maxd:
        return
    seen.add(t)
    p = prod.get(t)
    ncons = len(cons[t])
    print("  %s%-52s %s  [consumers=%d]" % ("  " * d, t, fmt(t), ncons), flush=True)
    if p is None:
        return
    if p.op_type == "Constant" and not a.weights:
        return
    for i in p.input:
        walk(i, d + 1, seen, maxd)


print("### file %s size=%d" % (a.onnx, __import__("os").path.getsize(a.onnx)), flush=True)
where = [n for n in g.node if n.op_type == "Where"]
print("### total Where nodes: %d" % len(where), flush=True)
for n in where:
    print("###   %-42s in=%s out=%s cons=%d" % (n.name, list(n.input), list(n.output),
                                                len(cons[n.output[0]])), flush=True)

cand = list(a.target) if a.target else [
    t for t in ("/encoder/Where_2_output_0", "/encoder/Where_1_output_0")]
cand = [t for t in cand if t in prod]
print("### candidate mask tensors present: %s" % cand, flush=True)
for t in cand:
    print("\n#### CHAIN %s" % t, flush=True)
    walk(t, 0, None, a.depth)
    print("   consumers: %s" % [(c.op_type, c.name) for c in cons[t]], flush=True)

print("\n#### per-layer attn mask source (first 6 and last 6 layers)", flush=True)
layers = sorted({int(n.name.split(".")[1]) for n in g.node
                 if n.name.startswith("/encoder/layers.") and ".attn/" in n.name})
print("### layer indices seen: %s" % layers, flush=True)
for L in layers[:6] + layers[-6:]:
    for suffix in ("attn/Add_2", "attn/Where", "attn/Add_3", "attn/Mul"):
        t = "/encoder/layers.%d/%s" % (L, suffix)
        if t in prod:
            print("   %-40s <- %s in=%s cons=%d" % (t, prod[t].op_type, list(prod[t].input),
                                                    len(cons[prod[t].output[0]])), flush=True)
print("### DONE", flush=True)
