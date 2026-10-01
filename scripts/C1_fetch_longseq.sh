#!/bin/bash
# Fetch the long-sequence artifacts from wsl2-local over LAN HTTP.
set -u
SRC=http://<BUILD_HOST_LAN>:8912
D=/home/harvest/laya-trt/longseq
mkdir -p "$D"
cd "$D" || exit 1
for f in heldout_s2048.tar heldout_s8192.tar laya-multilingual.s2048.m16.newlin.fp32.onnx s8192.onnxdir.tar; do
  echo "--- $f"
  curl -s -o "$f" -w 'http=%{http_code} bytes=%{size_download} time=%{time_total}s speed=%{speed_download}B/s\n' "$SRC/$f"
done
echo "=== md5 ==="
md5sum heldout_s2048.tar heldout_s8192.tar laya-multilingual.s2048.m16.newlin.fp32.onnx s8192.onnxdir.tar
echo "=== ls ==="
ls -la
echo "=== df after ==="
df -h / | tail -1
