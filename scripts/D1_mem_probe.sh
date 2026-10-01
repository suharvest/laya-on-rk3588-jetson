#!/bin/bash
# orin-nx: sample system memory while running the s8192 engine, and list engine layer names.
set -u
cd /home/harvest/laya-trt || exit 1
LOG=logs/memprobe_s8192.log
: > "$LOG"
echo "=== free -m before ===" >> "$LOG"
free -m >> "$LOG"

( for i in $(seq 1 500); do free -m | awk 'NR==2{print $3, $4, $7}'; sleep 0.25; done ) >> "$LOG" &
SAMP=$!
python3 laya_trt_run_cudart.py --engine longseq/laya.s8192.fp16.engine \
  --heldout longseq/heldout_s8192 --out longseq/memprobe_rank.json --name memprobe --mode rank \
  > logs/memprobe_run.log 2>&1
RC=$?
kill "$SAMP" 2>/dev/null
echo "RUN_RC=$RC"
echo "=== free -m after ===" >> "$LOG"
free -m >> "$LOG"
echo "=== peak used_MB over sampling window ==="
awk '/^(Mem|Swap|total)/{next} NF>=3 && $1 ~ /^[0-9]+$/ {if(NR>25){if($1>mx)mx=$1}} END{print "peak_used_MB="mx}' "$LOG"
echo "=== layer names: count and attention-ish ==="
python3 - <<'PY'
import json, collections
d = json.load(open("/home/harvest/laya-trt/logs/layers_s8192.json"))
layers = d["layers"] if isinstance(d, dict) and "layers" in d else d
print("total layers:", len(layers))
names = [l.get("Name", "") for l in layers]
cnt = collections.Counter()
for n in names:
    if "attn" in n.lower(): cnt["attn"] += 1
    if "softmax" in n.lower(): cnt["softmax"] += 1
    if "myelin" in n.lower(): cnt["myelin"] += 1
print("name buckets:", dict(cnt))
for n in names[:40]:
    print("  ", n)
PY
