#!/usr/bin/env python3
"""
Explicit Q/DQ insertion for laya s512 ONNX (TensorRT 10.3 explicit quantization).

Scope (per task book):
  * Quantize ONLY weight-bearing MatMul/Gemm nodes (A activation + B weight + output).
  * EXCLUDE every activation-only MatMul (rotary, QK^T, attn@V, head self_attn internals).
  * EXCLUDE the 5 network inputs (int64/bool).
  * EXCLUDE the mask path (Constant -10000 / -3.4e38 / -inf sentinels) - never fed to Q.
  * EXCLUDE Softmax / LayerNormalization and their elementwise neighbourhoods.

Scales:
  * weight  : per-channel symmetric int8, computed exactly from the fp32 initializer.
  * activation/output : per-tensor symmetric int8, taken from the TensorRT entropy
                        calibration cache produced by the *previous* implicit-PTQ run
                        (same calibrator, same 128-sample calib set).

Usage: qdq_insert.py <in.onnx> <calib.caltable> <out.onnx> [--apply]
"""
import sys, os, json, math, struct, time
from collections import Counter, OrderedDict
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

EPS = 1e-12

def info(*a):
    print(*a, flush=True)

# ---------------------------------------------------------------- caltable
def load_caltable(path):
    scales = OrderedDict()
    bad = OrderedDict()
    with open(path) as f:
        hdr = f.readline().strip()
        info(f"caltable header: {hdr}")
        for line in f:
            line = line.rstrip("\n")
            if not line or ":" not in line:
                continue
            name, _, hexv = line.rpartition(": ")
            try:
                v = struct.unpack(">f", bytes.fromhex(hexv.strip()))[0]
            except Exception:
                bad[name] = hexv
                continue
            if not math.isfinite(v) or v <= 0:
                bad[name] = hexv
                continue
            scales[name] = v
    info(f"caltable: {len(scales)} usable positive finite scales, {len(bad)} unusable")
    if bad:
        info("  unusable examples:", list(bad.items())[:5])
    return scales

# ------------------------------------------------------- weight resolution
def build_producer(g):
    prod = {}
    for n in g.node:
        for o in n.output:
            prod[o] = n
    return prod

def resolve_weight(name, init_map, prod, depth=0):
    """Return (init_name, perm) where perm is the ONNX Transpose perm applied to the
    stored initializer to obtain the tensor actually fed to the op (or None)."""
    if name in init_map:
        return name, None
    if depth > 6:
        return None, None
    n = prod.get(name)
    if n is None:
        return None, None
    if n.op_type == "Transpose":
        inner, p0 = resolve_weight(n.input[0], init_map, prod, depth + 1)
        if inner is None:
            return None, None
        p = None
        for a in n.attribute:
            if a.name == "perm":
                p = list(a.ints)
        if p is None:
            p = list(range(len(init_map[inner].dims)))[::-1]
        if p0 is not None:
            p = [p0[i] for i in p]
        return inner, p
    return None, None

def per_channel_axis(op_type, node, perm, stored_ndim):
    """Output-channel axis in the STORED initializer."""
    if op_type == "MatMul":
        eff_axis = 1                      # B is (K, N) -> N axis = 1
    else:                                 # Gemm
        transB = 0
        for a in node.attribute:
            if a.name == "transB":
                transB = a.i
        eff_axis = 0 if transB else 1
    if perm is None:
        return eff_axis
    return perm[eff_axis]

def quantize_weight(arr, axis):
    """Per-channel symmetric int8. Returns (int8 array, float32 scale 1-D)."""
    a = np.moveaxis(arr.astype(np.float32), axis, 0)
    flat = a.reshape(a.shape[0], -1)
    amax = np.max(np.abs(flat), axis=1)
    amax = np.maximum(amax, EPS)
    scale = (amax / 127.0).astype(np.float32)
    q = np.round(flat / scale[:, None])
    q = np.clip(q, -127, 127).astype(np.int8)
    q = np.moveaxis(q.reshape(a.shape), 0, axis)
    return q, scale

# ------------------------------------------------------------------- main
def main():
    src = sys.argv[1]
    calib_path = sys.argv[2]
    dst = sys.argv[3]
    apply_ = "--apply" in sys.argv

    scales = load_caltable(calib_path)

    info(f"\n===== LOADING {src} =====")
    m = onnx.load(src)
    g = m.graph
    info(f"nodes={len(g.node)} initializers={len(g.initializer)}")

    init_map = {i.name: i for i in g.initializer}
    prod = build_producer(g)

    targets = []          # (node, op_type)
    excluded_actonly = []
    for n in g.node:
        if n.op_type not in ("MatMul", "Gemm"):
            continue
        wname, perm = resolve_weight(n.input[1], init_map, prod)
        if wname is None:
            excluded_actonly.append((n.name, n.op_type, list(n.input)))
        else:
            targets.append((n, wname, perm))

    info(f"\n===== GEMM INVENTORY =====")
    info(f"total MatMul/Gemm nodes           : {sum(1 for n in g.node if n.op_type in ('MatMul','Gemm'))}")
    info(f"weight-bearing (TARGET)           : {len(targets)}")
    info(f"activation-only (EXCLUDED)        : {len(excluded_actonly)}")
    for nm, ot, ins in excluded_actonly[:8]:
        info(f"    excl {nm}  ({ot}) inputs={ins}")
    info(f"    ... ({len(excluded_actonly)} total excluded)")

    # ---------- coverage check ----------
    need_act = []       # (tensor_name, where)
    need_out = []
    seen = set()
    for n, wname, perm in targets:
        a_in = n.input[0]
        if a_in not in seen:
            seen.add(a_in)
            need_act.append((a_in, f"{n.name}[A]"))
        out = n.output[0]
        need_out.append((out, f"{n.name}[out]"))

    miss_a = [(t, w) for t, w in need_act if t not in scales]
    miss_o = [(t, w) for t, w in need_out if t not in scales]
    info(f"\n===== CALIBRATION COVERAGE =====")
    info(f"distinct A activations needed : {len(need_act)}, missing: {len(miss_a)}")
    info(f"GEMM outputs needed           : {len(need_out)}, missing: {len(miss_o)}")
    for t, w in miss_a[:15]:
        info(f"    MISSING(A)   {w:52s} {t}")
    for t, w in miss_o[:15]:
        info(f"    MISSING(out) {w:52s} {t}")

    qmap = {}          # original tensor -> (scale, kind)
    for t, w in need_act:
        if t in scales:
            qmap.setdefault(t, (scales[t], "act"))
    for t, w in need_out:
        if t in scales:
            qmap.setdefault(t, (scales[t], "out"))
    info(f"tensors that WILL be Q/DQ'd: {len(qmap)}")

    plan = {
        "targets": len(targets),
        "excluded_activation_only": len(excluded_actonly),
        "activated_tensors": len(qmap),
        "missing_activation_scales": [t for t, _ in miss_a],
        "missing_output_scales": [t for t, _ in miss_o],
        "weights": [],
    }
    for n, wname, perm in targets:
        init = init_map[wname]
        plan["weights"].append({"node": n.name, "weight": wname,
                                "shape": list(init.dims),
                                "perm": perm, "scale_found": True})
    print("PLAN_JSON " + json.dumps({k: v for k, v in plan.items() if k != "weights"}))

    if not apply_:
        info("\nDRY RUN - not writing. Re-run with --apply")
        return

    if miss_a:
        info(f"\nABORT: {len(miss_a)} A-activation scales missing from caltable; "
             f"refusing to build a partially-quantized graph.")
        sys.exit(3)

    info("\n===== INSERTING Q/DQ =====")
    nodes_orig = list(g.node)
    idx_of = {id(n): i for i, n in enumerate(nodes_orig)}
    head_nodes, pre_nodes, post_nodes = [], {}, {}
    new_inits = []
    qdq_cache = {}       # original tensor name -> replacement (dequantized) tensor name
    n_qd = 0

    def make_qdq(tensor, scale, place):
        """Insert QuantizeLinear -> DequantizeLinear on `tensor`; return the DQ output name.

        `place` is ('head', 0) / ('pre', i) / ('post', i) so the new nodes land next to the
        node that uses them and the assembled list stays in topological order.
        """
        nonlocal n_qd
        n_qd += 1
        tag = f"qdq_{n_qd:04d}"
        s_name = f"{tag}_scale"
        z_name = f"{tag}_zp"
        q_name = f"{tag}_quant"
        dq_name = f"{tag}_dequant"
        new_inits.append(helper.make_tensor(s_name, TensorProto.FLOAT, [], [float(scale)]))
        new_inits.append(helper.make_tensor(z_name, TensorProto.INT8, [], [0]))
        ns = [helper.make_node("QuantizeLinear", [tensor, s_name, z_name], [q_name],
                               name=f"{tag}_QuantizeLinear"),
              helper.make_node("DequantizeLinear", [q_name, s_name, z_name], [dq_name],
                               name=f"{tag}_DequantizeLinear")]
        kind, i = place
        {"head": head_nodes,
         "pre": pre_nodes.setdefault(i, []),
         "post": post_nodes.setdefault(i, [])}[kind].extend(ns)
        return dq_name

    # ---- 1. weights: int8 initializer + DequantizeLinear ----
    replaced_init = {}
    for n, wname, perm in targets:
        if wname in replaced_init:
            continue
        init = init_map[wname]
        arr = numpy_helper.to_array(init)
        axis = per_channel_axis(n.op_type, n, perm, arr.ndim)
        q, scale = quantize_weight(arr, axis)
        tag = f"wq_{len(replaced_init):04d}"
        q_name = f"{tag}_int8"
        s_name = f"{tag}_scale"
        dq_name = f"{tag}_dequant"
        new_inits.append(helper.make_tensor(q_name, TensorProto.INT8, list(q.shape), q.tobytes(), raw=True))
        new_inits.append(helper.make_tensor(s_name, TensorProto.FLOAT, [int(scale.shape[0])], scale.tolist()))
        head_nodes.append(helper.make_node(
            "DequantizeLinear", [q_name, s_name], [dq_name],
            name=f"{tag}_DequantizeLinear", axis=int(axis)))
        replaced_init[wname] = (dq_name, q_name, s_name, axis, float(np.max(scale)))
    info(f"weights quantized: {len(replaced_init)} initializers")

    # rewire every consumer of a replaced initializer
    for n in g.node:
        for i, inp in enumerate(n.input):
            if inp in replaced_init:
                n.input[i] = replaced_init[inp][0]

    drop_init_names = set(replaced_init.keys())

    # ---- 2. GEMM outputs (do this first so chained GEMMs reuse one Q/DQ) ----
    for n, wname, perm in targets:
        out = n.output[0]
        if out in qdq_cache or out not in qmap:
            continue
        rep = make_qdq(out, qmap[out][0], ("post", idx_of[id(n)]))
        for other in nodes_orig:
            for i, inp in enumerate(other.input):
                if inp == out:
                    other.input[i] = rep
        qdq_cache[out] = rep

    # ---- 3. activations (A inputs) ----
    for n, wname, perm in targets:
        a_in = n.input[0]
        if a_in in qdq_cache:
            n.input[0] = qdq_cache[a_in]
        elif a_in in qmap:
            rep = make_qdq(a_in, qmap[a_in][0], ("pre", idx_of[id(n)]))
            qdq_cache[a_in] = rep
            n.input[0] = rep
    info(f"Q/DQ pairs inserted: {n_qd}")

    # ---- 4. assemble in topological order (linear-time Kahn) ----
    final = list(head_nodes)
    for i, n in enumerate(nodes_orig):
        final.extend(pre_nodes.get(i, []))
        final.append(n)
        final.extend(post_nodes.get(i, []))

    avail = {i.name for i in g.initializer} | {i.name for i in g.input}
    avail |= {i.name for i in new_inits}
    consumers, indeg = {}, [0] * len(final)
    for i, n in enumerate(final):
        for t in set(n.input):
            if t and t not in avail:
                consumers.setdefault(t, []).append(i)
                indeg[i] += 1
    from collections import deque
    q = deque([i for i in range(len(final)) if indeg[i] == 0])
    order = []
    while q:
        i = q.popleft()
        order.append(i)
        for t in final[i].output:
            for j in consumers.get(t, ()):
                indeg[j] -= 1
                if indeg[j] == 0:
                    q.append(j)
    if len(order) != len(final):
        info(f"!! toposort produced {len(order)}/{len(final)} nodes - ABORT")
        sys.exit(4)
    del g.node[:]
    g.node.extend([final[i] for i in order])
    info(f"toposorted: {len(order)} nodes")

    del g.initializer[:]
    g.initializer.extend([i for i in init_map.values() if i.name not in drop_init_names])
    g.initializer.extend(new_inits)
    info(f"initializers: {len(g.initializer)} ({len(drop_init_names)} fp32 weights replaced by int8)")

    # ---------- report target distribution ----------
    pref = Counter()
    for n, wname, perm in targets:
        parts = n.name.strip("/").split("/")
        pref["/".join(parts[:3]) if len(parts) >= 3 else n.name] += 1
    info("\n===== TARGET GEMM DISTRIBUTION (by name prefix) =====")
    for k, v in sorted(pref.items()):
        info(f"    {v:4d}  {k}")
    enc_layers = sorted({int(n.name.split("layers.")[1].split("/")[0])
                         for n, _, _ in targets if n.name.startswith("/encoder/layers.")})
    info(f"encoder layers covered: {len(enc_layers)} -> {enc_layers[:3]}...{enc_layers[-3:] if enc_layers else []}")

    info("\n===== SAVING =====")
    t0 = time.time()
    onnx.save(m, dst)
    info(f"saved in {time.time()-t0:.1f}s")
    info(f"wrote {dst}")

if __name__ == "__main__":
    main()
