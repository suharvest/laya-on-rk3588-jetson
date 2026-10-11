#!/usr/bin/env bash
set -u
D=/dev/shm/kev-rkllm/context-8k
M=$D/kev-0.8b-w8a8-rk3588-context8192-shortcalib.rkllm
H=/dev/shm/kev-rkllm/head.bin
B=$D/runtime_context8k
mkdir -p $D/timing
printf 'start_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > $D/timing/timing-status.txt
for spec in 'orig-94 20 50 117 69 76 83 94 101 108 115' 'orig-128 20 50 143 95 102 109 120 127 134 141' 'orig-512 20 50 533 485 492 499 510 517 524 531' 'orig-1024 20 50 1027 979 986 993 1004 1011 1018 1025' 'orig-2048 20 50 2067 2019 2026 2033 2044 2051 2058 2065' 'orig-4096 2 10 4095 4047 4054 4061 4072 4079 4086 4093' 'orig-8000 1 5 8021 7973 7980 7987 7998 8005 8012 8019'; do
 set -- $spec; name=$1; warm=$2; samples=$3; shift 3; ids=$D/ids/$name.txt; log=$D/timing/$name.log
 { echo "start_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) name=$name warmup=$warm samples=$samples"; cat /proc/meminfo | grep -E 'MemAvailable|MemFree'; cat /sys/class/devfreq/fdab0000.npu/cur_freq 2>/dev/null || true; cat /sys/class/devfreq/fdab0000.npu/load 2>/dev/null || true; } >$D/timing/$name.state-before
 WARMUP=$warm SAMPLES=$samples $B $M $H $ids "$@" >$log 2>&1; rc=$?
 { echo "end_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) rc=$rc"; cat /proc/meminfo | grep -E 'MemAvailable|MemFree'; cat /sys/class/devfreq/fdab0000.npu/cur_freq 2>/dev/null || true; cat /sys/class/devfreq/fdab0000.npu/load 2>/dev/null || true; } >$D/timing/$name.state-after
 echo "$name rc=$rc" >>$D/timing/timing-status.txt
 if [ $rc -ne 0 ]; then echo "stop_after=$name" >>$D/timing/timing-status.txt; break; fi
done
echo "end_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>$D/timing/timing-status.txt
