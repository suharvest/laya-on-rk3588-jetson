#!/bin/bash
# negfltmax s90: the one allowed follow-up variant.
# Identical shared topology AND identical Where(cond, 0.0, Y) structure as the baseline; the only
# change is the masked sentinel value: Y = -inf -> -3.4028234663852886e+38.  That isolates
# "the sentinel value is out of fp16 range" from "the mask was rebuilt with Cast/Sub/Mul/Not".
#
# Port 8951 and /dev/shm/negfltmax_s90 are this task's own; other sessions' ports/dirs untouched.
set -u
HOST=<BUILD_HOST_LAN>
PORT=8951
LOG=/home/radxa/laya-rknn/negfltmax_s90.log
exec >> "$LOG" 2>&1
echo "===== START negfltmax_s90 $(date -Is) uptime: $(uptime) ====="
df -h / /dev/shm
D=/dev/shm/negfltmax_s90
mkdir -p "$D" || exit 2
cd "$D" || exit 3

pull() {
  local tag="$1" file="$2"
  echo "--- pull $tag <- $file"
  curl -s -m 900 -o "$tag.rknn" -w "$tag http=%{http_code} size=%{size_download} speed_bytes_per_s=%{speed_download}\n" \
    "http://$HOST:$PORT/$file"
  echo "CURL_RC=$?"
  md5sum "$tag.rknn"
  ls -la "$tag.rknn"
}

pull base oldship.s90.m16.fp16.nanguard.sharedmask.rk3588.rknn
pull neg  oldship.s90.m16.fp16.nanguard.sharedmask.negfltmax.rk3588.rknn
df -h / /dev/shm

cd /home/radxa/laya-rknn || exit 4

for tag in neg; do
  echo "===== $tag 47_device_dump ====="
  python3 47_device_dump.py --rknn "$D/$tag.rknn" --name "$tag" \
      --heldout /home/radxa/laya-rknn/heldout_s90 \
      --out "/home/radxa/laya-rknn/negfltmax_s90.${tag}.dump16.json" --core-mask 0_1_2
  echo "RC47_${tag}=$?"
done

for tag in neg base; do
  echo "===== $tag ng_lat iters=48 ====="
  python3 ng_lat.py --rknn "$D/$tag.rknn" --name "$tag" --heldout /home/radxa/laya-rknn/heldout_s90 \
      --out "/home/radxa/laya-rknn/negfltmax_s90.${tag}.lat48.json" --iters 48 --warmup 5 \
      --core-mask 0_1_2
  echo "RC_LAT_${tag}=$?"
done

echo "===== 46 ranking: fp32 vs base vs neg ====="
python3 46_rank_from_dumps.py --heldout /home/radxa/laya-rknn/heldout_s90 \
  --ref fp32=/home/radxa/laya-rknn/batch16_ort_nanfix_ref.json \
  --dump base=/home/radxa/laya-rknn/fltmaxmask_s90.base.dump16.json \
  --dump neg=/home/radxa/laya-rknn/negfltmax_s90.neg.dump16.json \
  --out /home/radxa/laya-rknn/negfltmax_s90.rank.json
echo "RC_RANK=$?"

python3 - <<'PYEOF'
import glob, json
for p in sorted(glob.glob("/home/radxa/laya-rknn/negfltmax_s90.*.lat48.json")):
    j = json.load(open(p))
    print("### %-8s iters=%s n=%s p50=%.2f p95=%.2f min=%.2f mean=%.2f"
          % (j.get("name"), j.get("iters"), j.get("n_latency"), j.get("p50"), j.get("p95"),
             j.get("min"), j.get("mean")))
PYEOF
echo "===== DONE negfltmax_s90 $(date -Is) ====="
df -h / /dev/shm
echo "===== NEGFLTMAX_S90_COMPLETE ====="
