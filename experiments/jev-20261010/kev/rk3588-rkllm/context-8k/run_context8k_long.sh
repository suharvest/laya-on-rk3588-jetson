#!/usr/bin/env bash
set -u
D=/dev/shm/kev-rkllm/context-8k; M=$D/kev-0.8b-w8a8-rk3588-context8192-shortcalib.rkllm; H=/dev/shm/kev-rkllm/head.bin; B=$D/runtime_context8k
printf 'start_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>$D/timing/timing-status-long.txt
for spec in 'orig-2048 2 10 2067 2019 2026 2033 2044 2051 2058 2065' 'orig-4096 2 10 4095 4047 4054 4061 4072 4079 4086 4093' 'orig-8000 1 5 8021 7973 7980 7987 7998 8005 8012 8019'; do
 set -- $spec; name=$1; warm=$2; samples=$3; shift 3; ids=$D/ids/$name.txt; log=$D/timing/$name-long.log
 { echo "start_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) name=$name warmup=$warm samples=$samples"; cat /proc/meminfo | grep -E 'MemAvailable|MemFree'; cat /sys/class/devfreq/fdab0000.npu/cur_freq 2>/dev/null || true; cat /sys/class/devfreq/fdab0000.npu/load 2>/dev/null || true; } >$D/timing/$name-long.state-before
 WARMUP=$warm SAMPLES=$samples $B $M $H $ids "$@" >$log 2>&1; rc=$?
 { echo "end_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) rc=$rc"; cat /proc/meminfo | grep -E 'MemAvailable|MemFree'; cat /sys/class/devfreq/fdab0000.npu/cur_freq 2>/dev/null || true; cat /sys/class/devfreq/fdab0000.npu/load 2>/dev/null || true; } >$D/timing/$name-long.state-after
 echo "$name rc=$rc" >>$D/timing/timing-status-long.txt
 if [ $rc -ne 0 ]; then echo "stop_after=$name" >>$D/timing/timing-status-long.txt; break; fi
done
echo "end_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>$D/timing/timing-status-long.txt
