#!/usr/bin/env python3
"""Build a TensorRT engine from the explicit-Q/DQ ONNX (no calibrator: scales are in the graph).

Flags deliberately mirror the prior implicit-PTQ build (FP16 + INT8, DETAILED verbosity) so the
only changed variable is the ONNX itself (Q/DQ present vs absent).

Emits machine-checkable int8 evidence: every layer input carries "Format/Datatype" in the engine
inspector JSON, so we can count Int8 vs Half vs Float precisely.
"""
import argparse, json, os, time
from collections import Counter
import tensorrt as trt


class L(trt.ILogger):
    def __init__(self):
        trt.ILogger.__init__(self)

    def log(self, sev, msg):
        print("[TRT] [%s] %s" % (sev.name, msg), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--engine", required=True)
    ap.add_argument("--layerinfo", default=None)
    ap.add_argument("--timing-in", default=None)
    ap.add_argument("--timing-out", default=None)
    ap.add_argument("--workspace-mb", type=int, default=4096)
    a = ap.parse_args()

    logger = L()
    builder = trt.Builder(logger)
    network = builder.create_network()
    parser = trt.OnnxParser(network, logger)

    print("### parsing %s" % a.onnx, flush=True)
    t0 = time.time()
    with open(a.onnx, "rb") as f:
        blob = f.read()
    if not parser.parse(blob):
        for i in range(parser.num_errors):
            print("### PARSE ERROR: %s" % parser.get_error(i), flush=True)
        raise SystemExit("onnx parse failed")
    print("### parsed in %.1fs; inputs=%d outputs=%d layers=%d"
          % (time.time() - t0, network.num_inputs, network.num_outputs, network.num_layers), flush=True)
    for i in range(network.num_inputs):
        ti = network.get_input(i)
        print("###   input %-16s %s %s" % (ti.name, ti.dtype, tuple(ti.shape)), flush=True)

    cfg = builder.create_builder_config()
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, a.workspace_mb << 20)
    cfg.set_flag(trt.BuilderFlag.FP16)
    cfg.set_flag(trt.BuilderFlag.INT8)
    cfg.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    _st = getattr(trt.BuilderFlag, "STRONGLY_TYPED", None)
    print("### flags: INT8=%s FP16=%s stronglyTyped=%s detail=%s"
          % (cfg.get_flag(trt.BuilderFlag.INT8), cfg.get_flag(trt.BuilderFlag.FP16),
             "n/a (absent in TRT %s)" % trt.__version__ if _st is None else cfg.get_flag(_st),
             True), flush=True)

    if a.timing_in and os.path.exists(a.timing_in):
        with open(a.timing_in, "rb") as f:
            cfg.set_timing_cache(cfg.create_timing_cache(f.read()), False)
        print("### loaded timing cache %s" % a.timing_in, flush=True)
    else:
        cfg.set_timing_cache(cfg.create_timing_cache(b""), False)
        print("### no timing cache loaded (fresh)", flush=True)

    print("### building (no calibrator; int8 scales come from the Q/DQ nodes) ...", flush=True)
    t0 = time.time()
    ser = builder.build_serialized_network(network, cfg)
    build_s = time.time() - t0
    if ser is None:
        raise SystemExit("### build_serialized_network returned None")
    print("### built in %.1fs serialized=%d bytes (%.2f MiB)"
          % (build_s, ser.nbytes, ser.nbytes / 1048576.0), flush=True)

    with open(a.engine, "wb") as f:
        f.write(ser)
    print("### wrote engine %s" % a.engine, flush=True)

    if a.timing_out:
        try:
            with open(a.timing_out, "wb") as f:
                f.write(cfg.get_timing_cache().serialize())
            print("### wrote timing cache %s" % a.timing_out, flush=True)
        except Exception as e:
            print("### timing cache write failed: %s" % e, flush=True)

    # ---------------- int8 evidence from the engine that was actually written ----------------
    eng = trt.Runtime(logger).deserialize_cuda_engine(ser)
    insp = eng.create_engine_inspector()
    txt = insp.get_engine_information(trt.LayerInformationFormat.JSON)
    lpath = a.layerinfo or (a.engine + ".layers.json")
    with open(lpath, "w") as f:
        f.write(txt)
    d = json.loads(txt)
    layers = d.get("Layers", [])

    dt_hist = Counter()
    layer_with_int8 = 0
    type_hist = Counter()
    int8_layers = []
    for L_ in layers:
        type_hist[L_.get("LayerType", "?")] += 1
        dts = [i.get("Format/Datatype", "?") for i in L_.get("Inputs", [])]
        dts += [o.get("Format/Datatype", "?") for o in L_.get("Outputs", [])]
        for dt in dts:
            dt_hist[dt] += 1
        if any("Int8" in str(x) for x in dts):
            layer_with_int8 += 1
            int8_layers.append({"name": L_.get("Name"), "type": L_.get("LayerType"),
                                "ins": dts[:3], "tactic": L_.get("TacticName", "")})

    print("### inspector layers total: %d" % len(layers), flush=True)
    print("### tensor datatype histogram (layer inputs+outputs): %s" % dict(dt_hist), flush=True)
    print("### layers with >=1 Int8 tensor: %d / %d  (%.1f%%)"
          % (layer_with_int8, len(layers), 100.0 * layer_with_int8 / max(1, len(layers))), flush=True)
    print("### layer type histogram: %s" % dict(type_hist), flush=True)
    print("### INT8_LAYERS_JSON " + json.dumps(int8_layers[:120]), flush=True)
    print("### DONE", flush=True)


if __name__ == "__main__":
    main()
