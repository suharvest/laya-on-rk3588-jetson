#!/bin/bash
# Explicit-Q/DQ driver. Args: nx | nano
# Same protocol as CG_drive.sh / IN_drive.sh so numbers are directly comparable:
#   trtexec --loadEngine --useSpinWait --iterations=200 --warmUp=2000 --avgRuns=100, graph vs nograph
#   laya_trt_run_graph.py --mode bench --iters 200 --warmup 20   (graph: + --graph --graph-warmup 10)
#   laya_trt_run_graph.py --mode rank  over the same 16 heldout rows
set -u
DEV="${1:?usage: QD_drive.sh nx|nano}"
cd /home/harvest/laya-trt || exit 1
mkdir -p qdq/out
case "$DEV" in
  nx)   E=qdq/laya.s512.qdq.engine ;;
  nano) E=qdq/laya.s512.qdq.nano.engine ;;
  *) echo "unknown dev $DEV"; exit 1 ;;
esac
PY=python3
TRTEXEC=/usr/src/tensorrt/bin/trtexec
STATUS=qdq/out/status_${DEV}.txt
: > "$STATUS"

echo "BEGIN QDQ DRIVER $DEV engine=$E $(date -Is)" >> "$STATUS"
ls -la "$E" >> "$STATUS" 2>&1
md5sum "$E" >> "$STATUS" 2>&1

run_trtexec() {  # $1=graph|nograph
  local G="$1" GFLAG=""
  [ "$G" = graph ] && GFLAG="--useCudaGraph"
  echo "BEGIN TRTEXEC qdq $G $(date -Is)" >> "$STATUS"
  $TRTEXEC --loadEngine="$E" --useSpinWait --iterations=200 --warmUp=2000 --avgRuns=100 \
    --exportTimes="qdq/out/trtexec_qdq_${DEV}_${G}.times.json" $GFLAG \
    > "qdq/out/trtexec_qdq_${DEV}_${G}.log" 2>&1
  echo "END TRTEXEC qdq $G RC=$? $(date -Is)" >> "$STATUS"
}

run_runner() {  # $1=tag $2=mode $3... extra
  local TAG="$1" M="$2"; shift 2
  echo "BEGIN RUNNER $TAG $(date -Is)" >> "$STATUS"
  $PY laya_trt_run_graph.py --engine "$E" --heldout calib_s512 \
    --out "qdq/out/${TAG}.json" --name "$TAG" --mode "$M" "$@" \
    > "qdq/out/${TAG}.log" 2>&1
  echo "END RUNNER $TAG RC=$? $(date -Is)" >> "$STATUS"
}

echo "=== trtexec qdq ($DEV) ==="
run_trtexec nograph
run_trtexec graph
echo "=== bench qdq ($DEV) ==="
run_runner bench_qdq_${DEV}_nograph bench --iters 200 --warmup 20
run_runner bench_qdq_${DEV}_graph   bench --iters 200 --warmup 20 --graph --graph-warmup 10
echo "=== rank qdq ($DEV) 16 heldout ==="
run_runner rank_qdq_${DEV}_nograph rank
run_runner rank_qdq_${DEV}_graph   rank --graph --graph-warmup 10
echo "QDQ_ALL_DONE $DEV $(date -Is)" >> "$STATUS"
echo DONE
