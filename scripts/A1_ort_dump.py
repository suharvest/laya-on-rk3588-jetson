"""ORT fp32 dump over a heldout dir, in the {"samples": [...]} shape that
scripts/46_rank_from_dumps.py consumes.

No surgery and no rknn conversion happen here: this only materialises the fp32 graph's own
output on the same 16 real rows the device will be fed, so the reference and the device dump
are directly comparable by scripts/46_rank_from_dumps.py (which computes the ranking over the
live markers only, and the rel_l2 over that same live slice).

Input names come from the ONNX graph itself, not a hardcoded list, so the s90 five-input
graph and any future packed graph both work.
"""
import argparse
import glob
import hashlib
import json
import os
import sys

import numpy as np
import onnxruntime as ort


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def split_outputs(outs, names):
    """logits is (1,K) and act_logits is (1,2); pick by shape so the order cannot be guessed."""
    logits = act = lg_name = ac_name = None
    for n, o in zip(names, outs):
        a = np.asarray(o)
        if a.ndim == 2 and a.shape[-1] == 2:
            act, ac_name = a, n
        else:
            logits, lg_name = a, n
    if logits is None or act is None:
        raise SystemExit("could not identify logits/act_logits in %s"
                         % [(n, np.shape(o)) for n, o in zip(names, outs)])
    return logits, act, lg_name, ac_name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--heldout", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    in_names = [i.name for i in sess.get_inputs()]
    out_names = [o.name for o in sess.get_outputs()]
    dirs = sorted(glob.glob(os.path.join(args.heldout, "sample_*")))
    if not dirs:
        raise SystemExit("no samples under %s" % args.heldout)

    print("### onnx   %s size=%d md5=%s" % (args.onnx, os.path.getsize(args.onnx),
                                            md5(args.onnx)), flush=True)
    print("### inputs %s" % in_names, flush=True)
    print("### outputs %s" % out_names, flush=True)
    print("### heldout %s samples=%d" % (args.heldout, len(dirs)), flush=True)

    samples = []
    for k, d in enumerate(dirs):
        feeds = {}
        for n in in_names:
            p = os.path.join(d, n + ".npy")
            if not os.path.exists(p):
                raise SystemExit("heldout %s has no %s.npy" % (d, n))
            feeds[n] = np.load(p)
        outs = sess.run(out_names, feeds)
        lg, ac, lg_name, ac_name = split_outputs(outs, out_names)
        row = {"idx": k, "logits": [float(v) for v in np.asarray(lg).reshape(-1)],
               "act_logits": [float(v) for v in np.asarray(ac).reshape(-1)]}
        samples.append(row)
        live = np.asarray(feeds[in_names[3]]).reshape(-1).astype(bool)
        print("### sim[%02d] n_live=%2d logits=%s" % (k, int(live.sum()),
              [round(v, 5) for v in np.asarray(lg).reshape(-1)[live]]), flush=True)

    doc = {"src": os.path.abspath(args.onnx), "onnx_md5": md5(args.onnx),
           "heldout": os.path.abspath(args.heldout), "input_order": in_names,
           "logits_name": lg_name, "act_logits_name": ac_name, "samples": samples}
    with open(args.out, "w") as f:
        json.dump(doc, f, indent=2)
    print("### wrote %s" % args.out, flush=True)
    print("### A1_ORT_DUMP_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
