#!/bin/bash
# Build the TensorRT fp16 engine for the laya s512 bucket on Jetson Orin.
#
# This wraps the command that actually produced out/laya.s512.fp16.engine
# (650,779,868 B, md5 ae0688b17b297b1474a18a97e9e04c77). There was originally NO build
# script for it -- the engine was built by hand with trtexec, and the command below is
# recovered verbatim from logs/build_s512_fp16_nx.log lines 1 and 197, with the paths
# made relative and parameterised.
#
# Build on Orin NX, not Orin Nano: the 8 GB Nano runs out of memory while profiling.
# Copy the resulting .engine to the Nano afterwards.
#
# Usage:  bash build_fp16_engine.sh <onnx> <out.engine> [workspace_mb]
# Example (the configuration this repository measured):
#         bash build_fp16_engine.sh onnx/laya-multilingual.s512.m16.fp32.ng.onnx \
#                                    out/laya.s512.fp16.engine 2048

set -euo pipefail

TRTEXEC="${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}"
ONNX="${1:?usage: build_fp16_engine.sh <onnx> <out.engine> [workspace_mb]}"
ENGINE="${2:?usage: build_fp16_engine.sh <onnx> <out.engine> [workspace_mb]}"
WORKSPACE_MB="${3:-2048}"
CACHE="${ENGINE%.engine}.timing.cache"

echo "onnx      : $ONNX"
echo "engine    : $ENGINE"
echo "workspace : ${WORKSPACE_MB} MB"
echo "trtexec   : $TRTEXEC"
echo

"$TRTEXEC" \
    --onnx="$ONNX" \
    --fp16 \
    --saveEngine="$ENGINE" \
    --timingCacheFile="$CACHE" \
    --memPoolSize="workspace:${WORKSPACE_MB}"

echo
ls -la "$ENGINE"
if command -v md5sum >/dev/null 2>&1; then md5sum "$ENGINE"; else md5 -q "$ENGINE"; fi
