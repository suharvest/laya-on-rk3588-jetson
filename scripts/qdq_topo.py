#!/usr/bin/env python3
"""Structural dump of the explicit-Q/DQ ONNX, focused on the logits path.

Prints, in graph (topological) order:
  * graph inputs / outputs
  * every QuantizeLinear / DequantizeLinear node with its scale and 127*s
  * the node chain under /scorer/ and /act_head/
  * the producer/consumer chain around the prime suspect tensor

Usage: qdq_topo.py <qdq.onnx> [--suspect NAME]
"""
import argparse
import sys
from collections import OrderedDict

import numpy as np
import onnx
from onnx import numpy_helper

ap = argparse.ArgumentParser()
ap.add_argument("onnx")
ap.add_argument("--suspect", default="/scorer/scorer.3/MatMul_output_0")
args = ap.parse_args()

print("===== LOADING", args.onnx, flush=True)
m = onnx.load(args.onnx)
g = m.graph
print("ir_version:", m.ir_version, "opset:", [(o.domain, o.version) for o in m.opset_import])
print("nodes:", len(g.node), "initializers:", len(g.initializer))

print("\n===== GRAPH INPUTS =====")
for i in g.input:
    tt = i.type.tensor_type
    dims = [d.dim_value if d.HasField("dim_value") else (d.dim_param or "?") for d in tt.shape.dim]
    print("  %-24s %-10s %s" % (i.name, onnx.TensorProto.DataType.Name(tt.elem_type), dims))

print("\n===== GRAPH OUTPUTS =====")
for o in g.output:
    tt = o.type.tensor_type
    dims = [d.dim_value if d.HasField("dim_value") else (d.dim_param or "?") for d in tt.shape.dim]
    print("  %-32s %-10s %s" % (o.name, onnx.TensorProto.DataType.Name(tt.elem_type), dims))

init = {i.name: i for i in g.initializer}


def scale_of(node):
    """Return the scalar float scale of a Q/DQ node (per-tensor only)."""
    if len(node.input) < 2:
        return None
    sn = node.input[1]
    if sn not in init:
        return None
    a = numpy_helper.to_array(init[sn]).astype(np.float32)
    if a.size != 1:
        return ("per_channel", tuple(a.shape))
    return float(a.reshape(-1)[0])


prod = {}
cons = {}
for idx, n in enumerate(g.node):
    for o in n.output:
        prod[o] = (idx, n)
    for i in n.input:
        cons.setdefault(i, []).append((idx, n))

print("\n===== ALL Q/DQ NODES (graph order) =====")
qdq_idx = []
for idx, n in enumerate(g.node):
    if n.op_type in ("QuantizeLinear", "DequantizeLinear"):
        qdq_idx.append(idx)
print("QuantizeLinear:", sum(1 for n in g.node if n.op_type == "QuantizeLinear"),
      "DequantizeLinear:", sum(1 for n in g.node if n.op_type == "DequantizeLinear"))

print("\n  idx  op            y  x  scale       127*s    out")
for idx in qdq_idx:
    n = g.node[idx]
    s = scale_of(n)
    sv = ("%.8g" % s) if isinstance(s, float) else str(s)
    sv127 = ("%.6f" % (127.0 * s)) if isinstance(s, float) else "-"
    print("  %4d %-13s %-28s <- %-34s s=%-12s 127s=%-9s" %
          (idx, n.op_type, n.output[0], n.input[0], sv, sv127))

# ---- 127*s in [4.0, 5.2] (the observed saturation floor band)
print("\n===== Q/DQ with 127*s in [4.0, 5.2] =====")
band = []
for idx in qdq_idx:
    n = g.node[idx]
    s = scale_of(n)
    if isinstance(s, float) and 4.0 <= 127.0 * s <= 5.2:
        band.append((n.output[0], n.op_type, s, 127.0 * s))
for name, op, s, v in band:
    print("  %-46s %-13s s=%.8g 127s=%.6f" % (name, op, s, v))

# ---- scorer / act_head chains
for prefix in ("/scorer/", "/act_head/"):
    print("\n===== NODES under %s =====" % prefix)
    for idx, n in enumerate(g.node):
        if n.name.startswith(prefix) or any(o.startswith(prefix) for o in n.output):
            print("  [%4d] %-16s name=%s" % (idx, n.op_type, n.name))
            print("         in : %s" % list(n.input))
            print("         out: %s" % list(n.output))

# ---- chain around the suspect
sus = args.suspect
print("\n===== CHAIN AROUND SUSPECT %s =====" % sus)
if sus in prod:
    idx, n = prod[sus]
    print("  producer [%d] %s name=%s out=%s" % (idx, n.op_type, n.name, list(n.output)))
    for i in n.input:
        if i in init:
            a = numpy_helper.to_array(init[i])
            print("    in %-40s INIT %s %s" % (i, a.dtype, a.shape))
        elif i in prod:
            pi, pn = prod[i]
            print("    in %-40s from [%d] %s %s" % (i, pi, pn.op_type, pn.name))
        else:
            print("    in %-40s <graph input>" % i)
    for k, (ci, cn) in enumerate(cons.get(sus, [])):
        print("  consumer [%d] %s name=%s" % (ci, cn.op_type, cn.name))
        print("    in : %s" % list(cn.input))
        print("    out: %s" % list(cn.output))
else:
    print("  SUSPECT NOT FOUND as a node output")

# ---- graph output producers (logits path)
print("\n===== FINAL NODES (last 25 in graph order) =====")
for idx in range(max(0, len(g.node) - 25), len(g.node)):
    n = g.node[idx]
    print("  [%4d] %-16s out=%s" % (idx, n.op_type, list(n.output)))

print("\n===== DONE =====")
