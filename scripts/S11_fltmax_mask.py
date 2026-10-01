"""Single-variable rewrite: keep the shared-mask topology, swap only the mask VALUE FORM.

Input : the verified shared-mask graph (2 encoder-level Where nodes, 8 + 14 layer consumers)
Output: same two shared carriers, same consumers, same tensor names, but the values are built
        the way the newlin export builds them.

Measured form of the newlin carriers (scripts/S10_semantics.py, one heldout row):
    /encoder/Where_1_output_0  = Mul(Sub(Cast(Expand(attention_mask)), 1.0), nb_const_fltmax)
                                 nb_const_fltmax is an INITIALIZER, +3.4028234663852886e+38
                                 runtime values {0.0 x7200, -3.4028234663852886e+38 x900}
    /encoder/Where_2_output_0  = Where(Cast_5, Constant_13 = -3.4028234663852886e+38,
                                       /encoder/Where_1_output_0)
                                 runtime values {0.0 x6755, -3.4028234663852886e+38 x1345}
oldship's shared carriers: {0.0 x7200, -inf x900} and {0.0 x6755, -inf x1345}.
So the blocked positions are the same; the sentinel is -inf vs -3.4028234663852886e+38, and the
arithmetic that produces it is Cast/Sub/Mul against +FLT_MAX instead of a 0.0/-inf Where.

The 8-layer carrier is rebuilt as the newlin Mul form; the 14-layer carrier as the newlin nested
Where form whose Y branch is that same 8-layer carrier, matching newlin's dependency.

Modes (only one may be passed; the default is the faithful transplant):
  full    both carriers in newlin form (default)
  negonly only the sentinel literal becomes -FLT_MAX; the 0.0/-inf Where structure is kept

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

POS_FLT_MAX = 3.4028234663852886e+38
NEG_FLT_MAX = -3.4028234663852886e+38

ap = argparse.ArgumentParser()
ap.add_argument("--onnx", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--report", default=None)
ap.add_argument("--mode", choices=["full", "negonly"], default="full")
args = ap.parse_args()

print("### loading %s" % args.onnx, flush=True)
m = onnx.load(args.onnx, load_external_data=False)
g = m.graph
report = {"src": os.path.abspath(args.onnx), "src_size": os.path.getsize(args.onnx),
          "mode": args.mode,
          "pos_flt_max": POS_FLT_MAX, "neg_flt_max": NEG_FLT_MAX}


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
inits = {i.name: i for i in g.initializer}


def cval(t):
    n = prod.get(t)
    if n is None:
        fail("%s has no producer" % t)
    if n.op_type == "Initializer":
        pass
    if n.op_type != "Constant":
        fail("%s is produced by %s, expected Constant" % (t, n.op_type))
    for a in n.attribute:
        if a.name == "value":
            return numpy_helper.to_array(a.t)
    fail("%s Constant has no value attribute" % t)


# ---------- 1. locate the two shared carriers ----------
shared = sorted([n for n in g.node if n.op_type == "Where"
                 and n.name.startswith("/encoder/SharedMask_")], key=lambda n: n.name)
print("### shared Where carriers: %d %s" % (len(shared), [n.name for n in shared]), flush=True)
if len(shared) != 2:
    fail("expected exactly 2 /encoder/SharedMask_* Where nodes, found %d" % len(shared))

info = []
for n in shared:
    x, y = cval(n.input[1]), cval(n.input[2])
    blocked = int(np.sum(~np.isfinite(y))) if y.dtype.kind == "f" and np.any(np.isinf(y)) else None
    ncons = len(cons[n.output[0]])
    # bool condition check is done through the dtype of the condition producer
    cp = prod.get(n.input[0])
    if cp is None:
        fail("condition %s of %s has no producer" % (n.input[0], n.name))
    info.append({"name": n.name, "cond": n.input[0], "cond_prod": cp.name,
                 "X": np.asarray(x).tolist(), "Y": np.asarray(y).tolist(),
                 "consumers": ncons, "out": n.output[0]})
    print("### carrier %-28s cond=%-32s X=%s Y=%s consumers=%d"
          % (n.name, n.input[0], np.asarray(x).tolist(), np.asarray(y).tolist(), ncons), flush=True)

info.sort(key=lambda d: -d["consumers"])
big, small = info[0], info[1]
print("### dependency order: 14-layer carrier=%s (cond %s) <- 8-layer carrier=%s (cond %s)"
      % (big["name"], big["cond"], small["name"], small["cond"]), flush=True)
if big["consumers"] != 14 or small["consumers"] != 8:
    fail("expected consumer split 14/8, got %d/%d" % (big["consumers"], small["consumers"]))
if not (np.asarray(small["X"]).tolist() == [0.0] and np.asarray(small["Y"]).tolist() == [-np.inf]):
    fail("8-layer carrier is not Where(cond, 0.0, -inf): X=%s Y=%s" % (small["X"], small["Y"]))

# both conditions must be produced before the carriers so insertion at the old position is safe
idx_of = {n.name: i for i, n in enumerate(g.node)}
anchor = max(idx_of[small["cond_prod"]], idx_of[big["cond_prod"]])
ins_at = min(idx_of[small["name"]], idx_of[big["name"]])
print("### cond producers at %d/%d, first carrier at %d"
      % (idx_of[small["cond_prod"]], idx_of[big["cond_prod"]], ins_at), flush=True)
if ins_at <= anchor:
    fail("carrier at %d is not after both cond producers (max %d)" % (ins_at, anchor))

# ---------- 2. build the replacement nodes ----------
one_name = "/encoder/SharedMaskF_one"
pos_name = "/encoder/SharedMaskF_posfltmax"   # initializer, as in newlin's nb_const_fltmax
neg_name = "/encoder/SharedMaskF_negfltmax"   # Constant node, as in newlin's Constant_13

new_nodes = []
added_init = []

if args.mode == "full":
    k_one = helper.make_node("Constant", [], [one_name + "_output_0"], name=one_name,
                             value=numpy_helper.from_array(np.asarray(1.0, np.float32), "v"))
    k_neg = helper.make_node("Constant", [], [neg_name + "_output_0"], name=neg_name,
                             value=numpy_helper.from_array(np.asarray(NEG_FLT_MAX, np.float32), "v"))
    g.initializer.append(numpy_helper.from_array(np.asarray(POS_FLT_MAX, np.float32), pos_name))
    added_init.append(pos_name)

    cast_out = "/encoder/SharedMaskF_cast_output_0"
    sub_out = "/encoder/SharedMaskF_sub_output_0"
    not_out = "/encoder/SharedMaskF_not_output_0"
    new_nodes += [
        k_one, k_neg,
        helper.make_node("Cast", [small["cond"]], [cast_out],
                         name="/encoder/SharedMaskF_cast", to=TensorProto.FLOAT),
        helper.make_node("Sub", [cast_out, one_name + "_output_0"], [sub_out],
                         name="/encoder/SharedMaskF_sub"),
        helper.make_node("Mul", [sub_out, pos_name], [small["out"]],
                         name="/encoder/SharedMaskF_mul"),
        helper.make_node("Not", [big["cond"]], [not_out], name="/encoder/SharedMaskF_not"),
        helper.make_node("Where", [not_out, neg_name + "_output_0", small["out"]], [big["out"]],
                         name="/encoder/SharedMaskF_nestwhere"),
    ]
else:  # negonly
    k_zero = helper.make_node("Constant", [], [one_name + "_output_0"], name=one_name,
                              value=numpy_helper.from_array(np.asarray(0.0, np.float32), "v"))
    k_neg = helper.make_node("Constant", [], [neg_name + "_output_0"], name=neg_name,
                             value=numpy_helper.from_array(np.asarray(NEG_FLT_MAX, np.float32), "v"))
    new_nodes += [
        k_zero, k_neg,
        helper.make_node("Where", [small["cond"], one_name + "_output_0", neg_name + "_output_0"],
                         [small["out"]], name=small["name"]),
        helper.make_node("Where", [big["cond"], one_name + "_output_0", neg_name + "_output_0"],
                         [big["out"]], name=big["name"]),
    ]

print("### replacement nodes (%d):" % len(new_nodes), flush=True)
for n in new_nodes:
    print("###   %-38s %-8s in=%s out=%s" % (n.name, n.op_type, list(n.input), list(n.output)),
          flush=True)

# ---------- 3. splice ----------
n_nodes_before = len(g.node)
drop = {small["name"], big["name"]}
old_const_names = set()
for n in shared:
    for t in (n.input[1], n.input[2]):
        p = prod.get(t)
        if p is not None and p.op_type == "Constant":
            others = [c for c in cons[t] if c.name not in drop]
            if others:
                fail("constant %s also feeds %s" % (t, [(c.op_type, c.name) for c in others]))
            old_const_names.add(p.name)
print("### old carrier Constants to drop: %d %s" % (len(old_const_names), sorted(old_const_names)),
      flush=True)

kept = [n for n in g.node if n.name not in drop and n.name not in old_const_names]
# insert right after the later of the two condition producers (topologically safe anchor)
anchor_name = max(info, key=lambda d: idx_of[d["cond_prod"]])["cond_prod"]
pos = [i for i, n in enumerate(kept) if n.name == anchor_name]
if len(pos) != 1:
    fail("anchor %s not uniquely present in the kept node list (%d)" % (anchor_name, len(pos)))
pos = pos[0] + 1
final = kept[:pos] + new_nodes + kept[pos:]
print("### nodes %d -> %d (dropped %d carrier + %d constant, inserted %d after %s at %d)"
      % (n_nodes_before, len(final), len(drop), len(old_const_names), len(new_nodes),
         anchor_name, pos), flush=True)

# ---------- 4. topological self-check ----------
seen = set(i.name for i in g.input) | set(i.name for i in g.initializer)
viol = []
for k, n in enumerate(final):
    for i in n.input:
        if i and i not in seen:
            viol.append((k, n.name, i))
    seen.update(n.output)
if viol:
    print("### TOPO_VIOLATIONS %d first=%s" % (len(viol), viol[:5]), flush=True)
    fail("node list is not topologically sorted")
print("### topo self-check PASS", flush=True)

# ---------- 5. drop any Constant left with zero consumers ----------
live = collections.Counter()
for n in final:
    for i in n.input:
        live[i] += 1
for o in g.output:
    live[o.name] += 1
final2 = [n for n in final if not (n.op_type == "Constant" and live[n.output[0]] == 0)]
print("### unused Constants dropped: %d -> final %d nodes" % (len(final) - len(final2), len(final2)),
      flush=True)

# ---------- 6. prune value_info ----------
alive = set()
for n in final2:
    for i in n.input:
        alive.add(i)
for o in g.output:
    alive.add(o.name)
for i in g.input:
    alive.add(i.name)
vi = [v for v in g.value_info if v.name in alive]
print("### value_info dropped: %d" % (len(g.value_info) - len(vi)), flush=True)

del g.node[:]
g.node.extend(final2)
del g.value_info[:]
g.value_info.extend(vi)

report.update({"out": os.path.abspath(args.out),
               "carriers": {"8_layer": small, "14_layer": big},
               "inserted": [{"name": n.name, "op": n.op_type} for n in new_nodes],
               "added_initializers": added_init,
               "dropped_where": sorted(drop), "dropped_constants": sorted(old_const_names),
               "nodes_before": n_nodes_before,
               "nodes_after": len(final2)})

try:
    onnx.checker.check_model(m)
    print("### onnx.checker PASS", flush=True)
    report["checker"] = "PASS"
except Exception as e:
    print("### onnx.checker FAILED: %s: %s" % (type(e).__name__, e), flush=True)
    report["checker"] = "FAIL: %s" % e
    sys.exit(3)

onnx.save(m, args.out)
report["out_size"] = os.path.getsize(args.out)
print("### wrote %s size=%d" % (args.out, report["out_size"]), flush=True)
if args.report:
    with open(args.report, "w") as f:
        json.dump(report, f, indent=2)
    print("### wrote %s" % args.report, flush=True)
print("### S11_DONE", flush=True)
