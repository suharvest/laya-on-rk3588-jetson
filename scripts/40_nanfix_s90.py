"""s90 nanfix: replace IsNaN(x) with Not(Equal(x, x)) for the RKNN build.

Surgery + topology assertion are copied verbatim from scripts/13_nanfix.py (hardcoded to the
s512 pair); only SRC/DST/GOLDEN change. golden_s90/*.npy holds the pre-surgery ORT outputs on
the golden input, so the post-surgery graph is diffed against the pre-surgery ORT run.

IsNaN has no CPU implementation in librknnrt 2.3.2 and every IsNaN in this graph lands on the
CPU side, so the first inference SIGSEGVs. NaN != NaN is exactly the IsNaN predicate.
"""
import hashlib
import os
import sys

import numpy as np
import onnx
import onnxruntime as ort

ONNX_DIR = "/home/harve/laya-rknn/onnx"
GOLDEN = "/home/harve/laya-rknn/golden_s90"
SRC = os.path.join(ONNX_DIR, "laya-multilingual.s90.m16.fp32.onnx")
DST = os.path.join(ONNX_DIR, "laya-multilingual.s90.m16.fp32.nanfix.onnx")
THRESH = 1e-3


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_feeds(sess):
    feeds = {}
    for io in sess.get_inputs():
        feeds[io.name] = np.load(os.path.join(GOLDEN, io.name + ".npy"))
    return feeds


def run_bytes(model_proto, feeds, want=("logits", "act_logits")):
    sess = ort.InferenceSession(model_proto.SerializeToString(),
                                providers=["CPUExecutionProvider"])
    out = sess.run(None, feeds)
    names = [o.name for o in sess.get_outputs()]
    d = dict(zip(names, out))
    return {k: d[k] for k in want if k in d}, sess


def main():
    print("onnxruntime", ort.__version__, flush=True)
    model = onnx.load(SRC)
    print("loaded %s size=%d md5=%s" % (SRC, os.path.getsize(SRC), md5(SRC)), flush=True)

    n_other = {}
    for n in model.graph.node:
        n_other[n.op_type] = n_other.get(n.op_type, 0) + 1
    print("### BEFORE: IsNaN=%d Equal=%d Not=%d" %
          (n_other.get("IsNaN", 0), n_other.get("Equal", 0), n_other.get("Not", 0)), flush=True)

    isnan_nodes = [n for n in model.graph.node if n.op_type == "IsNaN"]
    if not isnan_nodes:
        print("### NOTHING_TO_DO: no IsNaN nodes")
        return 1
    targets = [n.name or n.output[0] for n in isnan_nodes]
    print("### IsNaN nodes (%d):" % len(targets))
    for t in targets:
        print("      ", t, flush=True)

    base_sess = ort.InferenceSession(SRC, providers=["CPUExecutionProvider"])
    feeds = load_feeds(base_sess)
    print("### feeds:", {k: (v.shape, str(v.dtype)) for k, v in feeds.items()}, flush=True)

    # ---- diagnostic only: does the NaN guard ever fire on this input? -------------
    diag = onnx.ModelProto()
    diag.CopyFrom(model)
    existing = {o.name for o in diag.graph.output}
    for n in diag.graph.node:
        if n.op_type == "IsNaN" and n.output[0] not in existing:
            diag.graph.output.extend([onnx.helper.make_tensor_value_info(
                n.output[0], onnx.TensorProto.BOOL, None)])
    try:
        outs, _ = run_bytes(diag, feeds,
                            want=[n.output[0] for n in isnan_nodes if n.output[0] not in existing])
        total_true = 0
        for k, v in outs.items():
            total_true += int(np.count_nonzero(v))
        print("### DIAG IsNaN outputs read=%d total_true_elements=%d" % (len(outs), total_true), flush=True)
    except Exception as e:
        print("### DIAG failed (%s: %s) -- guard firing unknown" % (type(e).__name__, e), flush=True)
    del diag

    # ---- the surgery -------------------------------------------------------------
    shapes = {}
    for vi in list(model.graph.value_info):
        shapes[vi.name] = [d.dim_value for d in vi.type.tensor_type.shape.dim]

    new_nodes = []
    for n in model.graph.node:
        if n.op_type != "IsNaN":
            new_nodes.append(n)
            continue
        orig_out = n.output[0]
        mid = orig_out + "_eqbool"
        eq = onnx.helper.make_node("Equal", [n.input[0], n.input[0]], [mid],
                                   name=(n.name or orig_out) + "_equal")
        n.op_type = "Not"
        n.input[0] = mid
        if n.name:
            n.name = n.name + "_as_not"
        new_nodes.append(eq)
        new_nodes.append(n)
        model.graph.value_info.extend([onnx.helper.make_tensor_value_info(
            mid, onnx.TensorProto.BOOL, shapes.get(orig_out))])
    del model.graph.node[:]
    model.graph.node.extend(new_nodes)

    produced = set(i.name for i in model.graph.initializer)
    produced.update(i.name for i in model.graph.input)
    for n in model.graph.node:
        for inp in n.input:
            if inp and inp not in produced:
                raise AssertionError("NOT TOPOLOGICAL: %s (%s) reads %s before it is produced"
                                     % (n.name, n.op_type, inp))
        produced.update(o for o in n.output if o)
    print("### topological order verified over %d nodes" % len(model.graph.node), flush=True)

    after = {}
    for n in model.graph.node:
        after[n.op_type] = after.get(n.op_type, 0) + 1
    print("### AFTER: IsNaN=%d Equal=%d Not=%d" %
          (after.get("IsNaN", 0), after.get("Equal", 0), after.get("Not", 0)), flush=True)
    assert after.get("IsNaN", 0) == 0, "IsNaN still present"
    assert after.get("Equal", 0) == n_other.get("Equal", 0) + len(targets)
    assert after.get("Not", 0) == n_other.get("Not", 0) + len(targets)

    onnx.save(model, DST)
    size = os.path.getsize(DST)
    print("### wrote %s size=%d md5=%s" % (DST, size, md5(DST)), flush=True)

    # ---- semantic verification ---------------------------------------------------
    sess2 = ort.InferenceSession(DST, providers=["CPUExecutionProvider"])
    outs2 = dict(zip([o.name for o in sess2.get_outputs()], sess2.run(None, feeds)))
    print("### nanfix ORT outputs:", {k: v.shape for k, v in outs2.items()}, flush=True)

    worst_ort = 0.0
    for name in ("logits", "act_logits"):
        ref = np.load(os.path.join(GOLDEN, name + ".npy"))
        d = np.abs(ref.astype(np.float64) - outs2[name].astype(np.float64))
        print("  %-11s vs ORT baseline : max_abs_diff=%.6e  (bitwise_equal=%s)"
              % (name, d.max(), bool(np.array_equal(ref, outs2[name]))), flush=True)
        worst_ort = max(worst_ort, float(d.max()))
        t = np.load(os.path.join(GOLDEN, "torchref_" + name + ".npy"))
        dt = np.abs(t.astype(np.float64) - outs2[name].astype(np.float64))
        print("  %-11s vs torch eager  : max_abs_diff=%.6e" % (name, dt.max()), flush=True)
    print("### VERDICT max_abs_diff vs ORT baseline = %.6e  threshold %.0e -> %s"
          % (worst_ort, THRESH, "PASS" if worst_ort < THRESH else "FAIL"), flush=True)

    # ---- constant-False shortcut: evidence only -----------------------------------
    try:
        m2 = onnx.load(SRC)
        for n in m2.graph.node:
            if n.op_type == "IsNaN":
                n.op_type = "Constant"
                del n.input[:]
                del n.attribute[:]
                n.attribute.extend([onnx.helper.make_attribute(
                    "value", onnx.helper.make_tensor(
                        n.output[0] + "_c", onnx.TensorProto.BOOL, [1], [0]))])
        for n in m2.graph.node:
            if n.op_type == "Constant":
                n.attribute[0].t.name = n.output[0]
        o3, _ = run_bytes(m2, feeds)
        for name in ("logits", "act_logits"):
            d = np.abs(outs2[name].astype(np.float64) - o3[name].astype(np.float64))
            print("### constant-False variant vs nanfix, %-11s max_abs_diff=%.6e" % (name, d.max()), flush=True)
        del m2
    except Exception as e:
        print("### constant-False comparison skipped (%s: %s)" % (type(e).__name__, e), flush=True)

    print("### NANFIX_S90_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
