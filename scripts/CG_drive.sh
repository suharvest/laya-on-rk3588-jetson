#!/bin/bash
# CUDA-graph probe driver. Args: nx | nano
# Step 1 trtexec graph vs non-graph (same engine, same flags, only --useCudaGraph differs)
# Step 3/4 runner bench + 16-sample rank, graph vs non-graph, same engine/inputs/protocol.
set -u
DEV="${1:?usage: CG_drive.sh nx|nano}"
cd /home/harvest/laya-trt || exit 1
mkdir -p graph
case "$DEV" in
  nx)   E512=out/laya.s512.fp16.engine
        E90=out/laya.s90.fp16.engine ;;
  nano) E512=out/laya.s512.fp16.nxbuilt.engine
        E90=out/laya.s90.fp16.nxbuilt.engine ;;
  *) echo "unknown dev $DEV"; exit 1 ;;
esac
PY=python3
TRTEXEC=/usr/src/tensorrt/bin/trtexec
STATUS=graph/status_${DEV}.txt
: > "$STATUS"

run_trtexec() {  # $1=seq $2=engine $3=graph|nograph
  local S="$1" E="$2" G="$3" GFLAG=""
  [ "$G" = graph ] && GFLAG="--useCudaGraph"
  echo "BEGIN TRTEXEC s$S $G $(date -Is)" >> "$STATUS"
  $TRTEXEC --loadEngine="$E" --useSpinWait --iterations=200 --warmUp=2000 --avgRuns=100 \
    --exportTimes="graph/trtexec_s${S}_${G}.times.json" $GFLAG \
    > "graph/trtexec_s${S}_${G}.log" 2>&1
  echo "END TRTEXEC s$S $G RC=$? $(date -Is)" >> "$STATUS"
}

run_runner() {  # $1=tag $2=engine $3=heldout $4=mode $5=extra args
  local TAG="$1" E="$2" H="$3" M="$4"; shift 4
  echo "BEGIN RUNNER $TAG $(date -Is)" >> "$STATUS"
  $PY laya_trt_run_graph.py --engine "$E" --heldout "$H" \
    --out "graph/${TAG}.json" --name "$TAG" --mode "$M" "$@" \
    > "graph/${TAG}.log" 2>&1
  echo "END RUNNER $TAG RC=$? $(date -Is)" >> "$STATUS"
}

echo "=== STEP 1 trtexec ==="
run_trtexec 512 "$E512" nograph
run_trtexec 512 "$E512" graph
run_trtexec 90  "$E90"  nograph
run_trtexec 90  "$E90"  graph

echo "=== STEP 3 bench ==="
run_runner bench_s512_nograph "$E512" calib_s512      bench --iters 200 --warmup 20
run_runner bench_s512_graph   "$E512" calib_s512      bench --iters 200 --warmup 20 --graph --graph-warmup 10
run_runner bench_s90_nograph  "$E90"  calib_s90/heldout bench --iters 300 --warmup 20
run_runner bench_s90_graph    "$E90"  calib_s90/heldout bench --iters 300 --warmup 20 --graph --graph-warmup 10

echo "=== STEP 4 rank (16 samples) ==="
run_runner rank_s512_nograph  "$E512" calib_s512      rank
run_runner rank_s512_graph    "$E512" calib_s512      rank --graph --graph-warmup 10
run_runner rank_s90_nograph   "$E90"  calib_s90/heldout rank
run_runner rank_s90_graph     "$E90"  calib_s90/heldout rank --graph --graph-warmup 10

echo "CG_ALL_DONE $DEV $(date -Is)" >> "$STATUS"
echo "DONE"
