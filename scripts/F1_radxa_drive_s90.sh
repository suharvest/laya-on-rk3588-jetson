#!/bin/bash
# fltmaxmask s90: pull both artifacts over LAN and measure them in ONE process.
# /dev/shm has been observed to be cleared between fleet calls, so the transfer and the
# measurement must not be split across invocations.
#
# The shared-mask baseline is measured in the same session (and again at the end) so a
# machine-drift explanation can be ruled out from these numbers alone.
#
# Port 8951 and /dev/shm/fltmaxmask_s90 are this task's own; :8899/:8931/:8898/:8912/:8917 and
# the other directories belong to concurrent sessions and are not touched.
set -u
HOST=<BUILD_HOST_LAN>
PORT=8951
LOG=/home/radxa/laya-rknn/fltmaxmask_s90.log
exec >> "$LOG" 2>&1
echo "===== START fltmaxmask_s90 $(date -Is) uptime: $(uptime) ====="
df -h / /dev/shm
D=/dev/shm/fltmaxmask_s90
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
pull flt  oldship.s90.m16.fp16.nanguard.sharedmask.fltmaxmask.rk3588.rknn
df -h / /dev/shm

cd /home/radxa/laya-rknn || exit 4

for tag in base flt; do
  echo "===== $tag 47_device_dump ====="
  python3 47_device_dump.py --rknn "$D/$tag.rknn" --name "$tag" \
      --heldout /home/radxa/laya-rknn/heldout_s90 \
      --out "/home/radxa/laya-rknn/fltmaxmask_s90.${tag}.dump16.json" --core-mask 0_1_2
  echo "RC47_${tag}=$?"
done

for tag in base flt base2; do
  f="$D/$tag.rknn"
  [ -f "$f" ] || f="$D/base.rknn"
  echo "===== $tag ng_lat iters=48 (file $f) ====="
  python3 ng_lat.py --rknn "$f" --name "$tag" --heldout /home/radxa/laya-rknn/heldout_s90 \
      --out "/home/radxa/laya-rknn/fltmaxmask_s90.${tag}.lat48.json" --iters 48 --warmup 5 \
      --core-mask 0_1_2
  echo "RC_LAT_${tag}=$?"
done

echo "===== 46 ranking: fp32 reference vs base vs flt ====="
python3 46_rank_from_dumps.py --heldout /home/radxa/laya-rknn/heldout_s90 \
  --ref fp32=/home/radxa/laya-rknn/batch16_ort_nanfix_ref.json \
  --dump base=/home/radxa/laya-rknn/fltmaxmask_s90.base.dump16.json \
  --dump flt=/home/radxa/laya-rknn/fltmaxmask_s90.flt.dump16.json \
  --out /home/radxa/laya-rknn/fltmaxmask_s90.rank.json
echo "RC_RANK=$?"

echo "===== latency summary ====="
python3 - <<'PYEOF'
import glob, json
for p in sorted(glob.glob("/home/radxa/laya-rknn/fltmaxmask_s90.*.lat48.json")):
    j = json.load(open(p))
    print("### %-8s iters=%s n=%s p50=%.2f p95=%.2f min=%.2f mean=%.2f"
          % (j.get("name"), j.get("iters"), j.get("n_latency"), j.get("p50"), j.get("p95"),
             j.get("min"), j.get("mean")))
PYEOF
echo "===== DONE fltmaxmask_s90 $(date -Is) ====="
df -h / /dev/shm
ls -la /home/radxa/laya-rknn/fltmaxmask_s90.* 2>&1
echo "===== FLTMAXMASK_S90_COMPLETE ====="
