#!/bin/bash
# radxa 上的 bench 驱动。用 nohup 跑，避免 fleet 读 stdout 超时把进程连带杀掉。
# 用法（设备上）: nohup bash run_bench.sh > /dev/null 2>&1 &
PY=/home/radxa/e1-runtime/venv/bin/python
D=/home/radxa/laya-rknn
P=/home/radxa/laya-probe
cd "$D" || exit 1

: > "$D/bench.status"

echo "=== s64 nanfix (attention on NPU) zero-input ===" >> "$D/bench.status"
timeout 900 "$PY" "$P/npu_size_probe.py" \
  --rknn "$D/laya-multilingual.s64.m16.fp16.nanfix.rk3588.rknn" \
  --out "$D/zeros.s64fixed.json" --core-mask 0_1_2 --iters 10 --warmup 2 \
  > "$D/zeros.s64fixed.log" 2>&1
echo "s64_zero RC=$?" >> "$D/bench.status"

echo "=== s512 nanfix with golden, core=auto ===" >> "$D/bench.status"
timeout 900 "$PY" "$D/probe_laya_rknn.py" \
  --rknn "$D/laya-multilingual.s512.m16.fp16.nanfix.rk3588.rknn" \
  --golden-dir "$D/golden" \
  --out "$D/probe.fixed.auto.json" --core-mask auto --iters 3 --warmup 2 \
  > "$D/probe.fixed.auto.log" 2>&1
echo "s512_auto RC=$?" >> "$D/bench.status"

echo "ALL_DONE" >> "$D/bench.status"
