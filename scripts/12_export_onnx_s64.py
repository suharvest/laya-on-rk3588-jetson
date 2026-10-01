"""Fixed-shape ONNX export of a laya DecisionModel for RKNN (RK3588 NPU).

Differences from the upstream `scripts/export_onnx.py`:

* Dummy inputs use the *target* bucket sizes (default seq=512, markers=16), not 16/2.
  The legacy TorchScript tracer bakes `Reshape` targets in as constants, so the traced
  length has to be the length the converted NPU graph will actually run.
* `dynamic_axes` is removed entirely. rknn-toolkit2 needs every dim to be a concrete
  `dim_value`; a `dim_param` makes the conversion either fail or fall back.
* After export the script asserts that no input or output carries a `dim_param`, so a
  silently dynamic graph cannot be reported as a success.
"""
import argparse
import json
import os

import numpy as np
import torch

from laya.agent import Agent

INPUT_NAMES = ["input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype"]
OUTPUT_NAMES = ["logits", "act_logits"]


def build_inputs(seq_len: int, num_markers: int, seed: int):
    """Deterministic (batch=1) inputs. Same tensors are reused verbatim for the golden run."""
    g = torch.Generator().manual_seed(seed)

    input_ids = torch.randint(0, 1000, (1, seq_len), generator=g, dtype=torch.long)
    input_ids[0, :8] = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8], dtype=torch.long)

    # Real requests are padded; keep a real tail of masked positions so the padding path
    # (src_key_padding_mask / gather) is exercised in the golden too.
    attention_mask = torch.ones((1, seq_len), dtype=torch.long)
    attention_mask[0, seq_len - 12:] = 0

    # First 8 slots are live markers (the mask mixes live and dead entries).
    pos = [1, 5, 17, 33, 64, 100, 200, 300]
    pos += [0] * (num_markers - len(pos))
    # The hardcoded positions above assume the default seq bucket of 512. For a
    # shorter bucket (e.g. 64) DecisionModel would raise: it gathers with
    # marker_pos.clamp(min=0) and no upper clamp, so an index >= seq_len is out of
    # range. Clamp into range for short buckets; for seq=512 this line is a no-op
    # (1,5,17,33,64,100,200,300 are all < 512).
    pos = [p if p < seq_len else seq_len - 1 for p in pos]
    marker_pos = torch.tensor([pos[:num_markers]], dtype=torch.long)

    marker_mask = torch.zeros((1, num_markers), dtype=torch.bool)
    marker_mask[0, :8] = True

    qtype = torch.tensor([1], dtype=torch.long)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "marker_pos": marker_pos,
        "marker_mask": marker_mask,
        "qtype": qtype,
    }


def assert_static(path: str) -> dict:
    """Fail loudly if any input/output dim is symbolic."""
    import onnx

    model = onnx.load(path)
    spec = {"inputs": [], "outputs": []}
    bad = []
    for kind, vinfos in (("inputs", model.graph.input), ("outputs", model.graph.output)):
        for vi in vinfos:
            dims = []
            for d in vi.type.tensor_type.shape.dim:
                if d.HasField("dim_value"):
                    dims.append(int(d.dim_value))
                else:
                    dims.append(d.dim_param or "?")
                    bad.append("%s.%s" % (vi.name, d.dim_param or "?"))
            spec[kind].append({
                "name": vi.name,
                "dtype": onnx.TensorProto.DataType.Name(vi.type.tensor_type.elem_type),
                "shape": dims,
            })
    if bad:
        raise SystemExit("DYNAMIC DIM FOUND, refusing to call this a fixed-shape export: %s" % bad)
    return spec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/home/harve/laya-rknn/models/laya-multilingual")
    ap.add_argument("--output", default="/home/harve/laya-rknn/onnx/laya-multilingual.s512.m16.fp32.onnx")
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--num-markers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--golden-dir", default="/home/harve/laya-rknn/golden",
                    help="must differ from the s512 golden dir; pass golden_s64")
    ap.add_argument("--opset", type=int, default=18)
    ap.add_argument("--dynamo", action="store_true",
                    help="use the torch.export/dynamo path instead of the legacy TorchScript tracer")
    args = ap.parse_args()

    S, M = args.seq_len, args.num_markers
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    os.makedirs(args.golden_dir, exist_ok=True)

    print("torch", torch.__version__)
    print("Loading laya Agent (compile=False, device=cpu) from %s" % args.model)
    agent = Agent(args.model, compile=False, device="cpu")
    model = agent.model
    model.eval()
    print("model class:", type(model).__module__ + "." + type(model).__name__)

    tensors = build_inputs(S, M, args.seed)
    inputs = tuple(tensors[n] for n in INPUT_NAMES)
    for n, t in zip(INPUT_NAMES, inputs):
        print("  dummy %-15s shape=%s dtype=%s" % (n, tuple(t.shape), t.dtype))

    with torch.no_grad():
        ref_logits, ref_act = model(*inputs)
    ref_logits = ref_logits.numpy()
    ref_act = ref_act.numpy()

    print("Exporting (dynamo=%s, opset=%d) -> %s" % (args.dynamo, args.opset, args.output))
    if args.dynamo:
        torch.onnx.export(
            model, inputs, args.output,
            export_params=True, opset_version=args.opset,
            do_constant_folding=True,
            input_names=INPUT_NAMES, output_names=OUTPUT_NAMES,
        )
    else:
        # dynamo=False: the legacy TorchScript tracer, which is what the repo's
        # _DynamicMultiheadAttention was written against. No dynamic_axes.
        torch.onnx.export(
            model, inputs, args.output,
            export_params=True, opset_version=args.opset,
            do_constant_folding=True,
            input_names=INPUT_NAMES, output_names=OUTPUT_NAMES,
            dynamo=False,
        )

    spec = assert_static(args.output)
    print("ONNX spec (all dims concrete):")
    print(json.dumps(spec, indent=2))

    size = os.path.getsize(args.output)
    print("onnx size = %d bytes" % size)

    # Golden tensors saved under the exact ONNX input/output names so the RKNN-side
    # verification script can pair them up without a mapping table.
    for n, t in tensors.items():
        np.save(os.path.join(args.golden_dir, n + ".npy"), t.numpy())

    # The torch reference is written under distinct names: it is the eager-model output,
    # not the ORT output, and must not be mistaken for the golden the NPU gets compared to.
    np.save(os.path.join(args.golden_dir, "torchref_logits.npy"), ref_logits)
    np.save(os.path.join(args.golden_dir, "torchref_act_logits.npy"), ref_act)

    with open(os.path.join(args.golden_dir, "io_spec.json"), "w") as f:
        json.dump({"model": os.path.abspath(args.output), "spec": spec,
                   "seq_len": S, "num_markers": M, "seed": args.seed}, f, indent=2)

    print("torch eager logits[0,:8] =", np.array2string(ref_logits[0, :8], precision=6))
    print("torch eager act_logits    =", np.array2string(ref_act[0], precision=6))
    print("DONE")


if __name__ == "__main__":
    main()
