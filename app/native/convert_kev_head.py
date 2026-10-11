#!/usr/bin/env python3
"""Convert the pinned Kev training head checkpoint to the runtime FP32 NPZ contract."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np

EXPECTED = {"q.weight": (256, 1024), "q.bias": (256,), "k.weight": (256, 1024), "k.bias": (256,)}

def convert(source: Path, output: Path, metadata: Path) -> None:
    try:
        import torch
    except ImportError as exc:
        raise SystemExit("conversion requires the pinned source checkpoint environment with torch") from exc
    obj = torch.load(source, map_location="cpu", weights_only=False)
    head = obj.get("head") if isinstance(obj, dict) else None
    if not isinstance(head, dict):
        raise ValueError("checkpoint does not contain a head mapping")
    arrays = {}
    for key, shape in EXPECTED.items():
        if key not in head:
            raise ValueError(f"missing head tensor: {key}")
        value = head[key].detach().cpu().numpy().astype(np.float32, copy=False)
        if tuple(value.shape) != shape or not np.isfinite(value).all():
            raise ValueError(f"invalid {key}: shape={value.shape}")
        arrays[{"q.weight":"q_weight","q.bias":"q_bias","k.weight":"k_weight","k.bias":"k_bias"}[key]] = value
    temperature = float(obj.get("temperature"))
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, **arrays, temperature=np.float32(temperature))
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    head_sha = str(obj.get("init_source", {}).get("head_sha256", ""))
    meta = {"source": str(source), "source_sha256": source_sha, "head_sha256": head_sha,
            "temperature": temperature, "shapes": {k: list(v) for k, v in EXPECTED.items()},
            "dtype": "float32", "format": "kev-fp32-qk-v1"}
    metadata.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("source", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("metadata", type=Path)
    args = ap.parse_args()
    convert(args.source, args.output, args.metadata)
