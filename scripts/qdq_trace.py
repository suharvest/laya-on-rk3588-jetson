#!/usr/bin/env python3
"""Step-0 diagnostic: locate the FIRST tensor that diverges from the fp32 reference.

Method (no engine build, no new files on disk):
  1. Parse the explicit-Q/DQ ONNX and enumerate every DequantizeLinear site
     (its output name, the QuantizeLinear feeding it, the fp32 tensor name being
     quantized, the scale, and 127*s / 128*s).
  2. Add the fp32 twin of every activation site to graph.output *in memory*
     (ORT 1.23 refuses to fetch a name that is not a graph output), serialize the
     ModelProto to bytes and build the session straight from those bytes.
     graph_optimization_level=DISABLE_ALL keeps every Q/DQ a real op.
  3. Same for the Q/DQ model (fetch every DequantizeLinear output + the logits).
  4. Walk the sites in topological order, report the first with
         max|qdq - ref| / (max|ref| + eps) > THRESH
     and, for every site, the saturation ratio max|ref| / (127*s).

Usage:
  qdq_trace.py --qdq qdq/laya.s512.qdq.rs.onnx --ref onnx/...fp32.ng.onnx \
               --heldout calib_s512 --samples 14 --out qdq/out/trace_rs_s14.json
"""
import argparse
import gc
import json
import os
import time

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, numpy_helper

EPS = 1e-12

ap = argparse.ArgumentParser()
ap.add_argument("--qdq", required=True)
ap.add_argument("--ref", required=True)
ap.add_argument("--heldout", required=True)
ap.add_argument("--samples", default="14")
ap.add_argument("--out", required=True)
ap.add_argument("--thresh", type=float, default=0.05)
ap.add_argument("--threads", type=int, default=6)
ap.add_argument("--dump-sites", default="", help="write the Q/DQ site table here (json)")
args = ap.parse_args()


def log(*a):
    print(*a, flush=True)


def sess_opts():
    so = ort.SessionOptions()
    so.intra_op_num_threads = args.threads
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    so.log_severity_level = 3
    return so


def sess_from_names(path, names, tag):
    """Load `path`, add `names` to graph.output in memory, build the session from bytes."""
    t0 = time.perf_counter()
    log("### [%s] onnx.load %s" % (tag, path))
    m = onnx.load(path)
    existing = {o.name for o in m.graph.output}
    added = []
    for n in names:
        if n in existing:
            continue
        vi = helper.ValueInfoProto()
        vi.name = n
        vi.type.tensor_type.elem_type = onnx.TensorProto.FLOAT
        m.graph.output.append(vi)
        added.append(n)
    log("### [%s] added %d graph outputs (%d already present), serialize..." % (tag, len(added), len(names) - len(added)))
    blob = m.SerializeToString()
    log("### [%s] serialized %d bytes, building session" % (tag, len(blob)))
    del m
    gc.collect()
    sess = ort.InferenceSession(blob, sess_opts(), providers=["CPUExecutionProvider"])
    del blob
    gc.collect()
    log("### [%s] session ready in %.1f s" % (tag, time.perf_counter() - t0))
    return sess


def feed(sess, d):
    f = {}
    for i in sess.get_inputs():
        a = np.load(os.path.join(d, i.name + ".npy"))
        t = np.bool_ if "bool" in i.type else (np.int64 if "int64" in i.type else np.float32)
        f[i.name] = a.astype(t)
    return f


# ------------------------------------------------------------------ 1. site table
t0 = time.perf_counter()
log("### scanning Q/DQ sites in %s" % args.qdq)
m = onnx.load(args.qdq)
g = m.graph
init_names = {i.name for i in g.initializer}
init_by_name = {i.name: i for i in g.initializer}

q_of = {}
for n in g.node:
    if n.op_type == "QuantizeLinear":
        q_of[n.output[0]] = n

sites = []
for idx, n in enumerate(g.node):
    if n.op_type != "DequantizeLinear":
        continue
    sc = None
    if len(n.input) >= 2 and n.input[1] in init_by_name:
        sc = np.asarray(numpy_helper.to_array(init_by_name[n.input[1]]), dtype=np.float64)
    qn = q_of.get(n.input[0])
    fp_src = qn.input[0] if qn is not None else None
    kind = "weight" if (fp_src is not None and fp_src in init_names) else "act"
    sites.append({"dq_idx": idx, "dq_out": n.output[0], "q_in": n.input[0], "fp": fp_src,
                  "kind": kind, "per_channel": bool(sc is not None and sc.size > 1),
                  "scale": float(sc.reshape(-1)[0]) if sc is not None and sc.size else float("nan"),
                  "scale_n": int(sc.size) if sc is not None else 0})
sites.sort(key=lambda s: s["dq_idx"])
qdq_outputs = [o.name for o in g.output]
del m, g, init_by_name
gc.collect()

act_sites = [s for s in sites if s["kind"] == "act"]
log("### sites=%d act=%d weight=%d  (%.1f s)"
    % (len(sites), len(act_sites), len(sites) - len(act_sites), time.perf_counter() - t0))
if args.dump_sites:
    with open(args.dump_sites, "w") as f:
        json.dump(sites, f, indent=1)
    log("### wrote %s" % args.dump_sites)

# fp32 twin names must exist in the fp32 graph
t0 = time.perf_counter()
log("### scanning fp32 tensor names in %s" % args.ref)
mr = onnx.load(args.ref)
ref_names = {o for n in mr.graph.node for o in n.output}
ref_names.update(i.name for i in mr.graph.input)
ref_names.update(i.name for i in mr.graph.initializer)
ref_outputs = [o.name for o in mr.graph.output]
del mr
gc.collect()
missing = [s["fp"] for s in act_sites if s["fp"] not in ref_names]
log("### fp32 graph: %d names, outputs %s; act sites missing from it: %d %s (%.1f s)"
    % (len(ref_names), ref_outputs, len(missing), missing[:5], time.perf_counter() - t0))

result = {"qdq": args.qdq, "ref": args.ref, "thresh": args.thresh,
          "n_sites": len(sites), "n_act_sites": len(act_sites),
          "ref_outputs": ref_outputs, "samples": []}

import glob
dirs = sorted(glob.glob(os.path.join(args.heldout, "sample_*")))

for si in [int(x) for x in args.samples.split(",")]:
    d = dirs[si]
    log("\n===== SAMPLE %d (%s) =====" % (si, d))

    ref_fetch = [s["fp"] for s in act_sites if s["fp"] in ref_names]
    s_ref = sess_from_names(args.ref, ref_fetch + ref_outputs, "fp32")
    f = feed(s_ref, d)
    t1 = time.perf_counter()
    vals = s_ref.run(ref_fetch + [o for o in ref_outputs if o not in ref_fetch], f)
    log("### fp32 forward %.1f s" % (time.perf_counter() - t1))
    REF = dict(zip(ref_fetch, vals[: len(ref_fetch)]))
    REFO = dict(zip([o for o in ref_outputs if o not in ref_fetch], vals[len(ref_fetch):]))
    del s_ref, vals
    gc.collect()

    q_fetch = [s["dq_out"] for s in sites] + [o for o in qdq_outputs if o not in {s["dq_out"] for s in sites}]
    s_q = sess_from_names(args.qdq, q_fetch, "qdq")
    f = feed(s_q, d)
    t1 = time.perf_counter()
    q_vals = s_q.run(q_fetch, f)
    log("### qdq forward %.1f s" % (time.perf_counter() - t1))
    Q = dict(zip(q_fetch, q_vals))
    del s_q, q_vals
    gc.collect()

    rows = []
    first_bad = None
    for s in act_sites:
        if s["fp"] not in REF:
            continue
        r = np.asarray(REF[s["fp"]], dtype=np.float64)
        q = np.asarray(Q[s["dq_out"]], dtype=np.float64)
        if r.shape != q.shape:
            log("  shape mismatch %s: %s vs %s" % (s["dq_out"], r.shape, q.shape))
            continue
        amax = float(np.max(np.abs(r)))
        rmin = float(np.min(r))
        rmax = float(np.max(r))
        d = np.abs(q - r)
        rel = float(np.max(d) / (amax + EPS))
        sc = s["scale"]
        nrm = float(np.linalg.norm(r))
        rel_l2 = float(np.linalg.norm(q - r) / nrm) if nrm > 0 else float("nan")
        # fraction of elements the clamp actually moves (|ref| outside [-128s, 127s])
        if sc > 0:
            clip_frac = float(np.mean((r > 127.0 * sc) | (r < -128.0 * sc)))
        else:
            clip_frac = float("nan")
        row = dict(name=s["dq_out"], fp=s["fp"], rel=rel, rel_l2=rel_l2, clip_frac=clip_frac,
                   amax=amax, rmin=rmin, rmax=rmax,
                   scale=sc, s127=127.0 * sc, s128=128.0 * sc,
                   sat_hi=amax / (127.0 * sc) if sc > 0 else float("nan"),
                   sat_lo=abs(rmin) / (128.0 * sc) if sc > 0 else float("nan"),
                   dmax=float(np.max(d)), dmean=float(np.mean(d)),
                   qmin=float(np.min(q)), qmax=float(np.max(q)))
        del d
        rows.append(row)
        if first_bad is None and rel > args.thresh:
            first_bad = row
        del r, q

    log("\n===== SAMPLE %d: FIRST DIVERGENT SITE (rel > %.3f) =====" % (si, args.thresh))
    if first_bad is None:
        log("  NONE -- every activation Q/DQ output tracks fp32 within %.3f" % args.thresh)
    else:
        log("  name      : %s" % first_bad["name"])
        log("  fp tensor : %s" % first_bad["fp"])
        log("  rel=%.6f  max|delta|=%.6g  (ref absmax=%.6g)" % (first_bad["rel"], first_bad["dmax"], first_bad["amax"]))
        log("  ref  min/max : %.6g / %.6g" % (first_bad["rmin"], first_bad["rmax"]))
        log("  qdq  min/max : %.6g / %.6g" % (first_bad["qmin"], first_bad["qmax"]))
        log("  scale=%.8g  127*s=%.6f  128*s=%.6f" % (first_bad["scale"], first_bad["s127"], first_bad["s128"]))
        log("  saturation max|ref|/(127*s)=%.4f   |min|/(128*s)=%.4f" % (first_bad["sat_hi"], first_bad["sat_lo"]))

    log("\n===== SAMPLE %d: DISTRIBUTION OVER ALL %d ACT SITES =====" % (si, len(rows)))
    rl = np.array([r["rel"] for r in rows])
    r2 = np.array([r["rel_l2"] for r in rows])
    cf = np.array([r["clip_frac"] for r in rows])
    sh = np.array([r["sat_hi"] for r in rows])
    log("  rel     : min=%.5f p50=%.5f mean=%.5f max=%.5f   sites>%.3f: %d/%d"
        % (rl.min(), np.median(rl), rl.mean(), rl.max(), args.thresh, int((rl > args.thresh).sum()), len(rows)))
    log("  rel_l2  : min=%.5f p50=%.5f mean=%.5f max=%.5f   sites>0.05: %d/%d"
        % (r2.min(), np.median(r2), r2.mean(), r2.max(), int((r2 > 0.05).sum()), len(rows)))
    log("  clipfrac: min=%.5f p50=%.5f mean=%.5f max=%.5f" % (cf.min(), np.median(cf), cf.mean(), cf.max()))
    log("  sat_hi  : min=%.4f p50=%.4f mean=%.4f max=%.4f   sites>1.0 (clipping): %d/%d"
        % (sh.min(), np.median(sh), sh.mean(), sh.max(), int((sh > 1.0).sum()), len(rows)))

    log("\n===== SAMPLE %d: FIRST 10 SITES IN TOPOLOGICAL ORDER =====" % si)
    for r in rows[:10]:
        log("  rel=%.5f rel_l2=%.5f clip=%.5f sat=%.3f s127=%.5f amax=%9.4f  %s"
            % (r["rel"], r["rel_l2"], r["clip_frac"], r["sat_hi"], r["s127"], r["amax"], r["name"]))

    log("\n===== SAMPLE %d: TOP 15 BY rel =====" % si)
    for r in sorted(rows, key=lambda x: -x["rel"])[:15]:
        log("  rel=%.5f rel_l2=%.5f clip=%.4f sat=%.3f s127=%.5f amax=%9.4f  %s"
            % (r["rel"], r["rel_l2"], r["clip_frac"], r["sat_hi"], r["s127"], r["amax"], r["name"]))
    log("\n===== SAMPLE %d: TOP 15 BY SATURATION max|ref|/(127*s) =====" % si)
    for r in sorted(rows, key=lambda x: -x["sat_hi"])[:15]:
        log("  sat=%.4f satlo=%.4f rel=%.5f rel_l2=%.5f clip=%.4f s127=%.5f amax=%9.4f  %s"
            % (r["sat_hi"], r["sat_lo"], r["rel"], r["rel_l2"], r["clip_frac"], r["s127"], r["amax"], r["name"]))
    log("\n===== SAMPLE %d: TOP 15 BY rel_l2 (energy, not tail) =====" % si)
    for r in sorted(rows, key=lambda x: -x["rel_l2"])[:15]:
        log("  rel_l2=%.5f rel=%.5f clip=%.4f sat=%.3f s127=%.5f amax=%9.4f  %s"
            % (r["rel_l2"], r["rel"], r["clip_frac"], r["sat_hi"], r["s127"], r["amax"], r["name"]))
    order_names = [s["dq_out"] for s in act_sites]
    for probe in ("qdq_0048_dequant",):
        if probe in order_names:
            log("\n  [topo] %s is act site #%d of %d" % (probe, order_names.index(probe), len(order_names)))
    log("\n===== SAMPLE %d: THE SUSPECT SITE =====" % si)
    for r in rows:
        if r["name"] == "qdq_0048_dequant":
            log("  qdq_0048_dequant (/scorer/scorer.3/MatMul_output_0): rel=%.5f rel_l2=%.5f clip=%.5f"
                % (r["rel"], r["rel_l2"], r["clip_frac"]))
            log("    ref min/max/absmax = %.6g / %.6g / %.6g" % (r["rmin"], r["rmax"], r["amax"]))
            log("    qdq min/max        = %.6g / %.6g" % (r["qmin"], r["qmax"]))
            log("    scale=%.8g 127*s=%.6f 128*s=%.6f sat_hi=%.4f sat_lo=%.4f"
                % (r["scale"], r["s127"], r["s128"], r["sat_hi"], r["sat_lo"]))

    log("\n===== SAMPLE %d: LOGITS vs fp32 =====" % si)
    for o in qdq_outputs:
        if o in Q and o in REFO:
            r = np.asarray(REFO[o], dtype=np.float64).reshape(-1)
            q = np.asarray(Q[o], dtype=np.float64).reshape(-1)
            log("  %-12s ref=%s" % (o, np.round(r[:6], 4).tolist()))
            log("  %-12s qdq=%s" % ("", np.round(q[:6], 4).tolist()))
            log("  %-12s rel=%.6f min_ref=%.6f min_qdq=%.6f" % ("", float(np.max(np.abs(q - r)) / (np.max(np.abs(r)) + EPS)), float(np.min(r)), float(np.min(q))))

    result["samples"].append({"idx": si, "first_bad": first_bad, "rows": rows})
    del REF, REFO, Q, f
    gc.collect()

with open(args.out, "w") as f:
    json.dump(result, f, indent=1)
log("\n### wrote %s" % args.out)
