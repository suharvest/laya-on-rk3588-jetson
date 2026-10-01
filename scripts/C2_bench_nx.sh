#!/bin/bash
# orin-nx: dump TRT layer info (fusion evidence), then TRT bench + rank for both long buckets.
set -u
cd /home/harvest/laya-trt || exit 1
LOG=/home/harvest/laya-trt/logs
: > "$LOG/trt.status"

for S in 2048 8192; do
  echo "BEGIN LAYERINFO S=$S $(date -Is)" >> "$LOG/trt.status"
  /usr/src/tensorrt/bin/trtexec --loadEngine=longseq/laya.s$S.fp16.engine --skipInference \
    --dumpLayerInfo --profilingVerbosity=detailed --exportLayerInfo=$LOG/layers_s$S.json \
    > "$LOG/layerinfo_s$S.log" 2>&1
  echo "END LAYERINFO S=$S RC=$? $(date -Is)" >> "$LOG/trt.status"
done

for S in 2048 8192; do
  echo "BEGIN BENCH S=$S $(date -Is)" >> "$LOG/trt.status"
  python3 laya_trt_run_cudart.py --engine longseq/laya.s$S.fp16.engine \
    --heldout longseq/heldout_s$S --out longseq/trt_nx_s${S}_fp16_bench.json \
    --name trt_nx_s${S}_fp16 --mode bench --iters 50 --warmup 10 \
    > "$LOG/bench_s$S.log" 2>&1
  echo "END BENCH S=$S RC=$? $(date -Is)" >> "$LOG/trt.status"
  free -m >> "$LOG/bench_s$S.log"

  echo "BEGIN RANK S=$S $(date -Is)" >> "$LOG/trt.status"
  python3 laya_trt_run_cudart.py --engine longseq/laya.s$S.fp16.engine \
    --heldout longseq/heldout_s$S --out longseq/trt_nx_s${S}_fp16_rank.json \
    --name trt_nx_s${S}_fp16 --mode rank \
    > "$LOG/rank_s$S.log" 2>&1
  echo "END RANK S=$S RC=$? $(date -Is)" >> "$LOG/trt.status"
done
echo TRT_ALL_DONE $(date -Is) >> "$LOG/trt.status"
