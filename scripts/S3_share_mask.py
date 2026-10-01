"""Step 2: share the per-layer attention mask (single variable).

oldship s90 holds 22 /encoder/layers.N/attn/Where nodes. Step 1 showed they are degenerate
duplicates: X and Y are the literal scalars 0.0 and -inf (one distinct value each across all
22 layers), and each node's condition is one of exactly two shared S x S tensors
(/encoder/Expand_output_0 -> 8 layers, /encoder/Expand_1_output_0 -> 14 layers). So the 22
nodes materialise only 2 distinct S x S masks, 22 times.

This script builds the shared form:
  - one Where(cond, 0.0, -inf) per distinct condition, emitted once at encoder level
  - every layer's mask consumer (Add_2) rewired onto the shared tensor
  - the 22 redundant Where nodes and all 44 per-layer 0.0/-inf Constants removed

Placement: the shared nodes go immediately after the producer of the second condition, because
the constants of the per-layer nodes live at layer depth (layer 1's Constants sit after layer
0's Add_2), so reusing them would break topological order. The shared nodes carry their own
Constants.

Every structural assumption is asserted; a failed assertion aborts before any file is written.
"""
import argparse
import collections
import json
import os
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

ap = argparse.ArgumentParser()
ap.add_argument("--onnx", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--report", default=None)
args = ap.parse_args()

print("### loading %s" % args.onnx, flush=True)
m = onnx.load(args.onnx, load_external_data=False)
g = m.graph
report = {"src": os.path.abspath(args.onnx), "src_size": os.path.getsize(args.onnx)}


def fail(msg):
    print("### ASSERT_FAILED: %s" % msg, flush=True)
    sys.exit(2)


prod = {}
for n in g.node:
    for o in n.output:
        prod[o] = n
cons = collections.defaultdict(list)
for n in g.node:
    for i in n.input:
        cons[i].append(n)


def cval(t):
    n = prod.get(t)
    if n is None or n.op_type != "Constant":
        fail("%s is not produced by a Constant node (%s)" % (t, n.op_type if n else "absent"))
    for a in n.attribute:
        if a.name == "value":
            return numpy_helper.to_array(a.t)
    fail("%s Constant has no value attribute" % t)


# ---------- 1. verify the structure ----------
attn = sorted([n for n in g.node if n.op_type == "Where" and n.name.startswith("/encoder/layers.")],
              key=lambda n: n.name)
print("### attn Where nodes: %d" % len(attn), flush=True)
if len(attn) != 22:
    fail("expected 22 /encoder/layers.*/attn/Where, found %d" % len(attn))

groups = collections.OrderedDict()
for n in attn:
    groups.setdefault(n.input[0], []).append(n)
print("### distinct conditions: %d -> %s"
      % (len(groups), {k: len(v) for k, v in groups.items()}), flush=True)

ref_x = cval(attn[0].input[1])
ref_y = cval(attn[0].input[2])
report["X_value"] = np.asarray(ref_x).tolist()
report["Y_value"] = np.asarray(ref_y).tolist()
if ref_x.dtype != np.float32:
    fail("X dtype is %s, expected float32" % ref_x.dtype)
removed = set()
for n in attn:
    xv, yv = cval(n.input[1]), cval(n.input[2])
    if not (np.array_equal(xv, ref_x) and np.array_equal(yv, ref_y)):
        fail("%s X=%s Y=%s differs from the common X=%s Y=%s" % (n.name, xv, yv, ref_x, ref_y))
    if len(cons[n.output[0]]) != 1:
        fail("%s output has %d consumers: %s"
             % (n.name, len(cons[n.output[0]]), [(c.op_type, c.name) for c in cons[n.output[0]]]))
    removed.add(n.name)
print("### all 22 Where share X=%s Y=%s (dtype %s); each output has exactly 1 consumer"
      % (np.asarray(ref_x).tolist(), np.asarray(ref_y).tolist(), ref_x.dtype), flush=True)

cond_producer_idx = {}
for cond in groups:
    p = prod.get(cond)
    if p is None or not p.name.startswith("/encoder/"):
        fail("condition %s is not produced at encoder level (%s)" % (cond, p.name if p else "GIN"))
    cond_producer_idx[cond] = p.name
    print("### cond %-38s <- %s %s  n_layers=%d"
          % (cond, p.op_type, p.name, len(groups[cond])), flush=True)

# the 44 per-layer constants: only consumer must be the removed Where
const_removed = set()
for n in attn:
    for t in (n.input[1], n.input[2]):
        p = prod.get(t)
        if p is None or p.op_type != "Constant":
            fail("%s is not a Constant node, the shared form would need an initializer path" % t)
        others = [c for c in cons[t] if c.name not in removed]
        if others:
            fail("constant %s is also consumed by %s" % (t, [(c.op_type, c.name) for c in others]))
        const_removed.add(p.name)
print("### per-layer Constants to remove: %d" % len(const_removed), flush=True)

# ---------- 2. build the shared nodes ----------
zero_name = "/encoder/SharedMask_zero"
ninf_name = "/encoder/SharedMask_ninf"
k_zero = helper.make_node("Constant", [], [zero_name + "_output_0"], name=zero_name,
                          value=numpy_helper.from_array(np.asarray(ref_x, np.float32), "v"))
k_ninf = helper.make_node("Constant", [], [ninf_name + "_output_0"], name=ninf_name,
                          value=numpy_helper.from_array(np.asarray(ref_y, np.float32), "v"))
new_nodes = [k_zero, k_ninf]
rewire = {}
for gi, (cond, nodes) in enumerate(groups.items()):
    out_t = "/encoder/SharedMask_%d_output_0" % gi
    new_nodes.append(helper.make_node("Where", [cond, zero_name + "_output_0", ninf_name + "_output_0"],
                                      [out_t], name="/encoder/SharedMask_%d" % gi))
    for n in nodes:
        rewire[n.output[0]] = out_t
    print("### shared mask %d: %s = Where(%s, 0.0, -inf)  replaces %d per-layer nodes"
          % (gi, out_t, cond, len(nodes)), flush=True)

# ---------- 3. rewire consumers ----------
n_rewired = 0
for old, new in rewire.items():
    for c in cons[old]:
        for k, i in enumerate(c.input):
            if i == old:
                c.input[k] = new
                n_rewired += 1
print("### rewired %d consumer inputs" % n_rewired, flush=True)
if n_rewired != 22:
    fail("expected 22 rewired consumer inputs, did %d" % n_rewired)

# ---------- 4. rebuild the node list ----------
drop = removed | const_removed
kept_nodes = [n for n in g.node if n.name not in drop]
dropped_where = sum(1 for n in g.node if n.name in removed)
dropped_const = sum(1 for n in g.node if n.name in const_removed)

# insert after the producer of the second condition (the later of the two)
anchor = max(cond_producer_idx.values(), key=lambda nm: [i for i, n in enumerate(kept_nodes)
                                                        if n.name == nm][0])
idx = [i for i, n in enumerate(kept_nodes) if n.name == anchor][0] + 1
final = kept_nodes[:idx] + new_nodes + kept_nodes[idx:]
print("### dropped %d Where + %d Constant, inserted %d (after %s at %d): %d -> %d nodes"
      % (dropped_where, dropped_const, len(new_nodes), anchor, idx, len(g.node), len(final)),
      flush=True)

# second pass: any Constant left with zero consumers
live = collections.Counter()
for n in final:
    for i in n.input:
        live[i] += 1
for o in g.output:
    live[o.name] += 1
final2 = []
extra = 0
for n in final:
    if n.op_type == "Constant" and live[n.output[0]] == 0:
        extra += 1
        continue
    final2.append(n)
print("### extra unused Constants dropped: %d -> final %d nodes" % (extra, len(final2)), flush=True)

# ---------- 5. explicit topological-order self-check ----------
alive = set()
for n in final2:
    for i in n.input:
        alive.add(i)
for i in g.input:
    alive.add(i.name)
for i in g.initializer:
    alive.add(i.name)
seen = set(i.name for i in g.input) | set(i.name for i in g.initializer)
viol = []
for k, n in enumerate(final2):
    for i in n.input:
        if i and i not in seen:
            viol.append((k, n.name, i))
    seen.update(n.output)
if viol:
    print("### TOPO_VIOLATIONS: %d  first: %s" % (len(viol), viol[:5]), flush=True)
    fail("node list is not topologically sorted")
print("### topo self-check PASS (every input is produced by an earlier node)", flush=True)

# ---------- 6. prune value_info ----------
vi = [v for v in g.value_info if v.name in alive]
dropped_vi = len(g.value_info) - len(vi)

del g.node[:]
g.node.extend(final2)
del g.value_info[:]
g.value_info.extend(vi)

report.update({"out": os.path.abspath(args.out), "n_shared": 2,
               "removed_where": dropped_where, "removed_const": dropped_const + extra,
               "rewired": n_rewired, "nodes_before": 2536, "nodes_after": len(final2),
               "dropped_value_info": dropped_vi,
               "shared": [{"cond": c, "n_layers": len(nodes), "out": "/encoder/SharedMask_%d_output_0" % i}
                          for i, (c, nodes) in enumerate(groups.items())],
               "layers": {c: [n.name for n in nodes] for c, nodes in groups.items()}})

try:
    onnx.checker.check_model(m)
    print("### onnx.checker PASS", flush=True)
    report["checker"] = "PASS"
except Exception as e:
    print("### onnx.checker FAILED: %s: %s" % (type(e).__name__, e), flush=True)
    report["checker"] = "FAIL: %s" % e
    sys.exit(3)

onnx.save(m, args.out)
print("### wrote %s size=%d" % (args.out, os.path.getsize(args.out)), flush=True)
report["out_size"] = os.path.getsize(args.out)
if args.report:
    with open(args.report, "w") as f:
        json.dump(report, f, indent=2)
    print("### wrote report %s" % args.report, flush=True)
print("### S3_DONE", flush=True)
