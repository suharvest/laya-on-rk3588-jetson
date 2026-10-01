#!/bin/bash
# Round-2 explicit-Q/DQ driver (residual/skip stream kept in fp16). Args: nx | nano
# Same protocol as QD_drive.sh so numbers are directly comparable:
#   trtexec --loadEngine --useSpinWait --iterations=200 --warmUp=2000 --avgRuns=100, graph vs nograph
#   laya_trt_run_graph.py --mode bench --iters 200 --warmup 20   (graph: + --graph --graph-warmup 10)
#   laya_trt_run_graph.py --mode rank  over the same 16 heldout rows
set -u
DEV="${1:?usage: QD_drive2.sh nx|nano}"
cd /home/harvest/laya-trt || exit 1
mkdir -p qdq/out
case "$DEV" in
  nx)   E=qdq/laya.s512.qdq.rs.engine ;;
  nano) E=qdq/laya.s512.qdq.rs.nano.engine ;;
  *) echo "unknown dev $DEV"; exit 1 ;;
esac
PY=python3
TRTEXEC=/usr/src/tensorrt/bin/trtexec
STATUS=qdq/out/status_rs_${DEV}.txt
: > "$STATUS"

echo "BEGIN QDQ-RS DRIVER $DEV engine=$E $(date -Is)" >> "$STATUS"
ls -la "$E" >> "$STATUS" 2>&1
md5sum "$E" >> "$STATUS" 2>&1

run_trtexec() {  # $1=graph|nograph
  local G="$1" GFLAG=""
  [ "$G" = graph ] && GFLAG="--useCudaGraph"
  echo "BEGIN TRTEXEC rs $G $(date -Is)" >> "$STATUS"
  $TRTEXEC --loadEngine="$E" --useSpinWait --iterations=200 --warmUp=2000 --avgRuns=100 \
    --exportTimes="qdq/out/trtexec_rs_${DEV}_${G}.times.json" $GFLAG \
    > "qdq/out/trtexec_rs_${DEV}_${G}.log" 2>&1
  echo "END TRTEXEC rs $G RC=$? $(date -Is)" >> "$STATUS"
}

run_runner() {  # $1=tag $2=mode $3... extra
  local TAG="$1" M="$2"; shift 2
  echo "BEGIN RUNNER $TAG $(date -Is)" >> "$STATUS"
  $PY laya_trt_run_graph.py --engine "$E" --heldout calib_s512 \
    --out "qdq/out/${TAG}.json" --name "$TAG" --mode "$M" "$@" \
    > "qdq/out/${TAG}.log" 2>&1
  echo "END RUNNER $TAG RC=$? $(date -Is)" >> "$STATUS"
}

echo "=== trtexec rs ($DEV) ==="
run_trtexec nograph
run_trtexec graph
echo "=== bench rs ($DEV) ==="
run_runner bench_rs_${DEV}_nograph bench --iters 200 --warmup 20
run_runner bench_rs_${DEV}_graph   bench --iters 200 --warmup 20 --graph --graph-warmup 10
echo "=== rank rs ($DEV) 16 heldout ==="
run_runner rank_rs_${DEV}_nograph rank
run_runner rank_rs_${DEV}_graph   rank --graph --graph-warmup 10
echo "QDQ_RS_ALL_DONE $DEV $(date -Is)" >> "$STATUS"
echo DONE
