# Multi-platform runtime contract

This document describes packaging boundaries for the current Workbench source. It does not publish a new model or claim hardware readiness.

## Shared application contract

`app/edge_workbench/native.py` accepts five named fixed-shape inputs: `input_ids`, `attention_mask`, `marker_pos`, `marker_mask`, and `qtype`. `SEQ_LEN` must match the artifact. `MAX_OPTIONS` is capped at 16. The native adapter consumes choice logits; `act_logits` are not an execution permission. Input text is rejected when it exceeds the configured sequence budget.

## Rockchip

The existing `app/docker/Dockerfile` packages RKNN Lite2 2.3.2 for AArch64 and mounts the host `librknnrt.so` read-only. It is used for the validated RK3576 `seq=90` artifact and can be configured for an independently validated RK3588 artifact by supplying its own model, tokenizer, metadata, `SEQ_LEN`, and runtime library. The RK3588 historical `seq=512` artifact and the RK3576 artifact have different ONNX lineages; identical filenames or tokenizer names are not evidence of interchangeability.

## Jetson Orin

`app/docker/Dockerfile.jetson` is a separate candidate. It uses Ubuntu 22.04/Python 3.10 and the locked `jetson` extra for `cuda-python>=12,<13`. TensorRT Python bindings, CUDA libraries, and the serialized engine are host or operator supplied through `docker/docker-compose.jetson.yml`; they are not copied from the Spark build host. The compose sample uses the default s90 shape contract (`[1,90]`, 16 options); set `SEQ_LEN=512` only with a matching inspected engine and metadata.

The Jetson image has not been validated on Orin NX or Orin Nano by this change. A target acceptance must record JetPack/L4T, CUDA, TensorRT version, engine binding names/dtypes/shapes, model/tokenizer provenance, health JSON, and a real decision request.
