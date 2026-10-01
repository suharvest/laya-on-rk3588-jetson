import hashlib, json, os
import numpy as np, onnx
import onnxruntime as ort

GOLDEN = "/home/harve/laya-rknn/golden"
SRC = "/home/harve/laya-rknn/onnx/laya-multilingual.s512.m16.fp32.onnx"
NP = {"BOOL": "bool", "INT64": "int64", "FLOAT": "float32", "INT32": "int32", "FLOAT16": "float16"}

m = onnx.load(SRC)
def collect(vinfos):
    out = []
    for vi in vinfos:
        t = vi.type.tensor_type
        name = onnx.TensorProto.DataType.Name(t.elem_type)
        out.append({
            "name": vi.name,
            "dtype": name,
            "ort_dtype": "tensor(%s)" % name.lower(),
            "numpy_dtype": NP.get(name, "?"),
            "shape": [int(d.dim_value) for d in t.shape.dim],
        })
    return out

spec = {"inputs": collect(m.graph.input), "outputs": collect(m.graph.output)}
with open(os.path.join(GOLDEN, "io_spec.json"), "w") as f:
    json.dump(spec, f, indent=2)
print(json.dumps(spec, indent=2))

sess = ort.InferenceSession(SRC, providers=["CPUExecutionProvider"])
assert [i.name for i in sess.get_inputs()] == [s["name"] for s in spec["inputs"]]
assert [o.name for o in sess.get_outputs()] == [s["name"] for s in spec["outputs"]]

def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()

print("=== golden files (name MUST equal onnx tensor name) ===")
for kind, items in (("INPUT", spec["inputs"]), ("OUTPUT", spec["outputs"])):
    for s in items:
        p = os.path.join(GOLDEN, s["name"] + ".npy")
        a = np.load(p)
        print("%-6s %-15s size=%-9d md5=%s numpy=%-8s shape=%s" % (
            kind, s["name"], os.path.getsize(p), md5(p), str(a.dtype), a.shape))

print("=== all npy in golden ===")
for fn in sorted(os.listdir(GOLDEN)):
    p = os.path.join(GOLDEN, fn)
    print("  %s  %d bytes  md5=%s" % (fn, os.path.getsize(p), md5(p)))
print("DONE")
