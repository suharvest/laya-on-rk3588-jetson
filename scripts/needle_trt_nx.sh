#!/bin/bash
# orin-nx: TRT fp16 rank dumps over all 15 needle sets + latency benches. Engines reused as-is.
set -u
cd /home/harvest/laya-trt || exit 1
mkdir -p needle/trt logs/needle
tar xf needle_sets.tar -C needle
: >> logs/needle/trt.status

eng() {
  case $1 in
    512)  echo out/laya.s512.fp16.engine ;;
    2048) echo longseq/laya.s2048.fp16.engine ;;
    8192) echo longseq/laya.s8192.fp16.engine ;;
  esac
}

for L in en zh es ja ar; do
  for S in 512 2048 8192; do
    echo "BEGIN RANK $L $S $(date -Is)" >> logs/needle/trt.status
    python3 laya_trt_run_cudart.py --engine "$(eng $S)" --heldout "needle/${L}_s${S}" \
      --out "needle/trt/${L}_s${S}.json" --name "trt_nx_${L}_s${S}" --mode rank \
      > "logs/needle/rank_${L}_s${S}.log" 2>&1
    echo "END RANK $L $S RC=$? $(date -Is)" >> logs/needle/trt.status
  done
done

for S in 512 2048 8192; do
  if [ "$S" = "8192" ]; then IT=100; else IT=200; fi
  echo "BEGIN BENCH $S $(date -Is)" >> logs/needle/trt.status
  python3 laya_trt_run_cudart.py --engine "$(eng $S)" --heldout "needle/en_s${S}" \
    --out "needle/bench_nx_s${S}.json" --name "bench_nx_s${S}" --mode bench \
    --iters "$IT" --warmup 20 > "logs/needle/bench_s${S}.log" 2>&1
  echo "END BENCH $S RC=$? $(date -Is)" >> logs/needle/trt.status
done
echo TRT_NEEDLE_NX_DONE $(date -Is) >> logs/needle/trt.status
