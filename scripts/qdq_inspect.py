#!/usr/bin/env python3
"""Inspect the s512 ONNX: opset, inputs, GEMM/MatMul nodes, mask-path sentinel, softmax/norm sites."""
import sys, json
from collections import Counter
import numpy as np
import onnx
from onnx import numpy_helper

ONNX = sys.argv[1] if len(sys.argv) > 1 else "/home/harvest/laya-trt/onnx/laya-multilingual.s512.m16.fp32.ng.onnx"

print("===== LOADING =====", flush=True)
m = onnx.load(ONNX)
g = m.graph
print("ir_version:", m.ir_version)
print("producer:", m.producer_name, m.producer_version)
print("opset:", [(o.domain, o.version) for o in m.opset_import])
print("doc_string len:", len(m.doc_string))

print("\n===== GRAPH INPUTS =====")
for i in g.input:
    tt = i.type.tensor_type
    dims = [d.dim_value if d.HasField('dim_value') else (d.dim_param or '?') for d in tt.shape.dim]
    print(f"  {i.name:24s} dtype={onnx.TensorProto.DataType.Name(tt.elem_type):10s} shape={dims}")

print("\n===== GRAPH OUTPUTS =====")
for o in g.output:
    tt = o.type.tensor_type
    dims = [d.dim_value if d.HasField('dim_value') else (d.dim_param or '?') for d in tt.shape.dim]
    print(f"  {o.name:24s} dtype={onnx.TensorProto.DataType.Name(tt.elem_type):10s} shape={dims}")

print("\n===== NODE OP HISTOGRAM =====")
ctr = Counter(n.op_type for n in g.node)
for k, v in sorted(ctr.items(), key=lambda kv: -kv[1]):
    print(f"  {k:24s} {v}")

print("\n===== INITIALIZER COUNT/SIZE =====")
tot = 0
big = []
for init in g.initializer:
    arr = numpy_helper.to_array(init)
    nbytes = arr.nbytes
    tot += nbytes
    if nbytes > 4 * 1024 * 1024:
        big.append((init.name, tuple(arr.shape), arr.dtype.str, nbytes))
print("initializer count:", len(g.initializer), "total bytes:", tot, f"({tot/1024/1024:.1f} MiB)")
big.sort(key=lambda x: -x[3])
print("top-15 largest initializers:")
for n, s, d, b in big[:15]:
    print(f"  {n[:70]:70s} {str(s):22s} {d} {b/1024/1024:8.2f} MiB")

print("\n===== MATMUL/GEMM NODES =====")
mm = [n for n in g.node if n.op_type in ("MatMul", "Gemm")]
print("count:", len(mm))
init_names = {i.name: i for i in g.initializer}
for idx, n in enumerate(mm):
    ins = list(n.input)
    shapes = []
    for i in ins:
        if i in init_names:
            shapes.append(tuple(numpy_helper.to_array(init_names[i]).shape))
        else:
            shapes.append(None)
    print(f"  [{idx:03d}] name={n.name!r} op={n.op_type}")
    print(f"        inputs={ins}")
    print(f"        initshapes={shapes}")
    print(f"        outputs={list(n.output)}")

print("\n===== SENTINEL / MASK-PATH CONSTANTS (value <= -9999) =====")
sent = []
for init in g.initializer:
    arr = numpy_helper.to_array(init)
    if arr.size and arr.size <= 64 and np.issubdtype(arr.dtype, np.floating):
        if np.min(arr) <= -9999:
            sent.append((init.name, arr.flatten()[:8].tolist()))
print("scalar-ish sentinel initializers:", len(sent))
for n, v in sent[:20]:
    print("  ", n, v)

print("\n===== CONSTANT NODES with float sentinel =====")
cnode_sent = []
for n in g.node:
    if n.op_type == "Constant":
        for a in n.attribute:
            if a.name == "value" and a.t.data_type == onnx.TensorProto.FLOAT:
                arr = numpy_helper.to_array(a.t)
                if arr.size <= 64 and arr.size and np.min(arr) <= -9999:
                    cnode_sent.append((n.name, arr.flatten()[:8].tolist(), list(n.output)))
print("constant-node sentinels:", len(cnode_sent))
for n, v, o in cnode_sent[:20]:
    print("  ", n, v, o)

print("\n===== VALUE_INFO present? =====")
print("graph.value_info count:", len(g.value_info))
print("\n===== DONE =====")
