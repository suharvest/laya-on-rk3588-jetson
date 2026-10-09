#!/usr/bin/env python3
"""Run the edge-workbench native choice gate on a target device.

This is a probe, not a deployment or package installer. It expects the target
device to already provide the vendor runtime (RKNN Lite or TensorRT), the
current Spark native.py, a local tokenizer directory, and a compiled model.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import statistics
import sys
import time
from pathlib import Path

PROMPT = "You are a production command selector. Choose exactly one matching option."
OPTIONS = [
    {"id": "speed", "label": "Set conveyor speed", "description": "Adjust the conveyor speed to the target percentage"},
    {"id": "pause", "label": "Pause equipment", "description": "Safely pause the current production equipment"},
    {"id": "status", "label": "Check equipment status", "description": "Return the current operating status"},
]
CASES = [
    ("en_speed", "Reduce conveyor 3 speed to 60%", "speed"),
    ("en_pause", "Pause the production line now", "pause"),
    ("en_status", "Show camera 2 temperature", "status"),
    ("zh_speed", "把三号输送带速度降到60%", "speed"),
    ("zh_pause", "停止当前生产线", "pause"),
    ("zh_status", "查询设备当前运行状态", "status"),
]


def args_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", choices=("rknn", "tensorrt"), required=False)
    p.add_argument("--model", type=Path)
    p.add_argument("--tokenizer-dir", type=Path)
    p.add_argument("--native", type=Path, required=True, help="Spark edge_workbench/native.py")
    p.add_argument("--seq-len", type=int, default=90)
    p.add_argument("--max-options", type=int, default=16)
    p.add_argument("--core-mask", default="auto", help="RKNN core mask")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--dry-run", action="store_true", help="validate the probe contract without importing a vendor runtime")
    p.add_argument("--output", type=Path)
    return p


def validate_args(args: argparse.Namespace) -> None:
    if args.seq_len <= 0:
        raise ValueError("--seq-len must be positive")
    if args.max_options <= 0:
        raise ValueError("--max-options must be positive")
    if args.warmup < 0 or args.iters <= 0:
        raise ValueError("--warmup must be non-negative and --iters must be positive")
    if args.dry_run:
        return
    if args.backend is None or args.model is None or args.tokenizer_dir is None:
        raise ValueError("--backend, --model and --tokenizer-dir are required unless --dry-run is used")


def dry_result(args: argparse.Namespace) -> dict:
    return {"dry_run": True, "prompt": PROMPT, "options": OPTIONS, "cases": CASES,
            "seq_len": args.seq_len, "max_options": args.max_options,
            "p95_method": "sorted latencies; index=min(n-1, int(n*0.95))"}


def load_native(path: Path):
    spec = importlib.util.spec_from_file_location("edge_workbench_native_probe", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load native module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def run(args: argparse.Namespace) -> dict:
    import numpy as np
    native = load_native(args.native)
    codec = native.LayaCodec(str(args.tokenizer_dir), args.seq_len, args.max_options)
    backend_type = native.RKNNBackend if args.backend == "rknn" else native.TensorRTBackend
    if args.backend == "rknn":
        backend = backend_type(str(args.model), args.seq_len, args.max_options, core_mask=args.core_mask)
    else:
        backend = backend_type(str(args.model), args.seq_len, args.max_options)
    if not backend.ready:
        raise RuntimeError(backend.reason or "backend unavailable")

    def infer(text: str) -> dict:
        req = codec.encode(PROMPT, text, OPTIONS)
        scores = backend.infer_request(req)
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return {"scores": scores, "order": [OPTIONS[i]["id"] for i in order],
                "finite": bool(np.isfinite(scores).all()),
                "input_shapes": {"input_ids": list(req.input_ids.shape), "attention_mask": list(req.attention_mask.shape),
                                  "marker_pos": list(req.marker_pos.shape), "marker_mask": list(req.marker_mask.shape),
                                  "qtype": list(req.qtype.shape)}}

    for _ in range(args.warmup):
        infer("检查设备状态")
    latencies = []
    for _ in range(args.iters):
        start = time.perf_counter(); infer("检查设备状态"); latencies.append((time.perf_counter() - start) * 1000)
    rows = []
    for name, text, expected in CASES:
        row = infer(text); row.update({"name": name, "text": text, "expected": expected, "top1": row["order"][0]})
        rows.append(row)
    backend.release()
    ordered = sorted(latencies)
    return {"backend": args.backend, "model": str(args.model), "model_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
            "model_size": args.model.stat().st_size, "native": str(args.native), "native_sha256": hashlib.sha256(args.native.read_bytes()).hexdigest(), "prompt": PROMPT, "options": OPTIONS,
            "semantic_cases": rows, "bench": {"n": len(latencies), "warmup": args.warmup, "p50_ms": ordered[len(ordered)//2],
            "p95_ms": ordered[min(len(ordered)-1, int(len(ordered)*0.95))], "p95_method": "sorted latencies; index=min(n-1, int(n*0.95))",
            "mean_ms": statistics.mean(latencies)}, "summary": {"semantic_top1": sum(r["top1"] == r["expected"] for r in rows),
            "semantic_total": len(rows), "finite_active": sum(r["finite"] for r in rows)}}


def main() -> int:
    args = args_parser().parse_args()
    try:
        validate_args(args)
        result = dry_result(args) if args.dry_run else run(args)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    print(rendered, end="")
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
