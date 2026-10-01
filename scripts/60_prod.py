#!/usr/bin/env python3
"""One round of latency, or one 16-row dump, for a single .rknn on the radxa RK3588.

Feed construction, the dtype map and the core-mask enum are imported from 47_device_dump.py, so
this probe feeds the model exactly like the existing accuracy/latency scripts do.

Timing region is r.inference() inclusive of the rknnlite host-side input/output copies, the same
region ng_lat.py times, so numbers stay comparable with the earlier lat48 runs.

Every iteration also records which CPU ran it and that CPU's scaler frequency at that instant,
plus a 5 Hz background sampler for the NPU devfreq node and the NPU/soc thermal zones. That is
what turns a latency distribution into a drift attribution instead of an unexplained spread.
"""
import argparse
import ctypes
import glob
import importlib.util
import json
import os
import threading
import time

import numpy as np
from rknnlite.api import RKNNLite

try:
    _libc = ctypes.CDLL("libc.so.6", use_errno=True)
    _libc.sched_getcpu.restype = ctypes.c_int
except OSError:
    _libc = None


def cur_cpu():
    """os.sched_getcpu is not exposed in this build's os module; go through libc."""
    if _libc is not None:
        return _libc.sched_getcpu()
    with open("/proc/self/stat") as f:
        return int(f.read().rsplit(")", 1)[1].split()[36])

DUMPER = "/home/radxa/laya-rknn/47_device_dump.py"
_spec = importlib.util.spec_from_file_location("dump47", DUMPER)
_m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m)

NPU = "/sys/class/devfreq/fdab0000.npu"
PINS = {"none": None, "little": [0, 1, 2, 3], "big": [4, 5, 6, 7]}


def rd(path):
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return None


def state():
    return {
        "npu_cur_freq": rd(NPU + "/cur_freq"),
        "npu_governor": rd(NPU + "/governor"),
        "npu_trans_total": (rd(NPU + "/trans_stat") or "").splitlines()[-1].strip(),
        "npu_temp_mC": rd("/sys/class/thermal/thermal_zone6/temp"),
        "soc_temp_mC": rd("/sys/class/thermal/thermal_zone0/temp"),
        "big_temp_mC": rd("/sys/class/thermal/thermal_zone1/temp"),
        "fan_cur": rd("/sys/class/thermal/cooling_device5/cur_state"),
        "cpu_freq_kHz": {str(c): rd("/sys/devices/system/cpu/cpu%d/cpufreq/scaling_cur_freq" % c)
                         for c in (0, 4, 6)},
        "cpu_governor": {str(c): rd("/sys/devices/system/cpu/cpu%d/cpufreq/scaling_governor" % c)
                         for c in (0, 4, 6)},
        "loadavg": rd("/proc/loadavg"),
    }


class Sampler(threading.Thread):
    """5 Hz device-state sampler running alongside the timed loop."""

    def __init__(self, hz=5.0):
        super().__init__(daemon=True)
        self.hz = hz
        self.stop = threading.Event()
        self.samples = []

    def run(self):
        while not self.stop.is_set():
            s = state()
            s["t"] = round(time.perf_counter(), 3)
            self.samples.append(s)
            self.stop.wait(1.0 / self.hz)


def sig_of(r):
    n_in, _ = r.rknn_runtime.get_in_out_num()
    sig = []
    for i in range(n_in):
        at = r.rknn_runtime.get_tensor_attr(i, is_output=False)
        sig.append({"index": i,
                    "name": at.name.decode() if isinstance(at.name, bytes) else str(at.name),
                    "type": int(at.type), "dtype": _m.RKNN_TYPE.get(int(at.type)),
                    "dims": [int(d) for d in at.dims][:int(at.n_dims)]})
    return sig


def feed_for(sig, d, warn):
    out = []
    for s in sig:
        t = _m.NP_DTYPE[s["dtype"]]
        p = os.path.join(d, s["name"] + ".npy")
        if os.path.exists(p):
            a = np.load(p)
        elif s["name"] == "packed":
            a = _m.pack_from_five(d)
            warn.append("built packed for %s" % os.path.basename(d))
        else:
            raise SystemExit("no %s.npy in %s" % (s["name"], d))
        a = a.astype(t)
        if list(a.shape) != s["dims"]:
            warn.append("shape %s vs rknn %s for %s" % (list(a.shape), s["dims"], s["name"]))
        out.append(a)
    return out


def w(o, p):
    with open(p, "w") as f:
        json.dump(o, f, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["lat", "dump", "state"])
    ap.add_argument("--rknn")
    ap.add_argument("--name", default="m")
    ap.add_argument("--heldout")
    ap.add_argument("--out")
    ap.add_argument("--iters", type=int, default=120)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--core-mask", default="0_1_2", choices=sorted(_m.CORE))
    ap.add_argument("--pin", default="none", choices=sorted(PINS))
    ap.add_argument("--trace", action="store_true")
    a = ap.parse_args()

    if a.cmd == "state":
        print(json.dumps(state(), indent=2))
        return 0

    res = {"name": a.name, "rknn": a.rknn, "cmd": a.cmd, "ok": False,
           "core_mask": a.core_mask, "pin": a.pin, "iters": a.iters, "warmup": a.warmup,
           "warnings": []}

    # affinity must be set before RKNNLite spins up its worker threads
    if PINS[a.pin]:
        os.sched_setaffinity(0, set(PINS[a.pin]))
    res["affinity_np"] = sorted(os.sched_getaffinity(0))
    res["state_before"] = state()

    r = RKNNLite(verbose=False)
    try:
        res["rknn_bytes"] = os.path.getsize(a.rknn)
        t0 = time.perf_counter()
        if r.load_rknn(a.rknn) != 0:
            res["error"] = "load_rknn failed"; w(res, a.out); print("### LOAD_FAILED"); return 2
        if r.init_runtime(core_mask=_m.CORE[a.core_mask]) != 0:
            res["error"] = "init_runtime failed"; w(res, a.out); print("### INIT_FAILED"); return 3
        res["load_init_s"] = round(time.perf_counter() - t0, 2)
        sig = sig_of(r)
        res["inputs"] = sig
        res["sdk_version"] = str(r.rknn_runtime.get_sdk_version())
        print("### [%s] sdk=%s pin=%s iters=%d inputs=%s"
              % (a.name, res["sdk_version"], a.pin, a.iters, json.dumps(sig)), flush=True)

        dirs = sorted(glob.glob(os.path.join(a.heldout, "sample_*")))
        res["n_samples"] = len(dirs)
        print("### [%s] heldout=%s samples=%d" % (a.name, a.heldout, len(dirs)), flush=True)
        assert len(dirs) == 16, "expected 16 held-out rows, got %d" % len(dirs)
        feeds = [feed_for(sig, d, res["warnings"]) for d in dirs]

        for i in range(a.warmup):
            r.inference(inputs=feeds[i % len(feeds)])

        smp = Sampler()
        smp.start()
        lat, trace, last = [], [], {}
        for i in range(a.iters):
            k = i % len(feeds)
            t0 = time.perf_counter()
            outs = r.inference(inputs=feeds[k])
            dt = (time.perf_counter() - t0) * 1000.0
            lat.append(dt)
            last[k] = outs
            if a.trace:
                c = cur_cpu()
                trace.append({"i": i, "k": k, "ms": round(dt, 3), "cpu": c,
                              "cpu_kHz": rd("/sys/devices/system/cpu/cpu%d/cpufreq/scaling_cur_freq" % max(c, 0)),
                              "npu_kHz": rd(NPU + "/cur_freq")})
        smp.stop.set(); smp.join(timeout=2)
        res["state_after"] = state()
        res["sampler"] = smp.samples
        if a.trace:
            res["trace"] = trace

        s = sorted(lat)
        n = len(s)
        res["n_latency"] = n
        res["p50"] = round(s[n // 2], 2)
        res["p95"] = round(s[min(n - 1, int(0.95 * n))], 2)
        res["min"] = round(s[0], 2)
        res["max"] = round(s[-1], 2)
        res["mean"] = round(sum(s) / n, 2)
        res["p10"] = round(s[int(0.10 * n)], 2)
        res["p90"] = round(s[min(n - 1, int(0.90 * n))], 2)
        res["iqr"] = round(s[int(0.75 * n)] - s[int(0.25 * n)], 2)
        res["latency_ms_all"] = [round(v, 2) for v in lat]
        res["ok"] = True

        for k in sorted(last):
            lg = ac = None
            for o in last[k]:
                arr = np.asarray(o)
                if arr.ndim == 2 and arr.shape[-1] == 2:
                    ac = arr
                else:
                    lg = arr
            res.setdefault("samples", []).append(
                {"idx": k, "logits": [float(v) for v in np.asarray(lg).reshape(-1)],
                 "act_logits": [float(v) for v in np.asarray(ac).reshape(-1)]})
        w(res, a.out)
        print("### LAT %s pin=%s iters=%d n=%d p50=%.2f p95=%.2f min=%.2f max=%.2f mean=%.2f iqr=%.2f load_init=%.2fs"
              % (a.name, a.pin, a.iters, n, res["p50"], res["p95"], res["min"], res["max"],
                 res["mean"], res["iqr"], res["load_init_s"]), flush=True)
        print("### LATENCY_DONE %s %s" % (a.name, a.pin), flush=True)
        return 0
    finally:
        try:
            r.release()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
