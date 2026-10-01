#!/usr/bin/env bash
# Recon for explicit Q/DQ experiment on orin-nx
echo "===== HOST ====="
hostname; uname -a
echo "===== DF ====="
df -h / /home
echo "===== PY3 ====="
which python3; python3 -VV
echo "===== TRT ====="
python3 -c 'import tensorrt as trt; print("TRT", trt.__version__)' 2>&1 | tail -3
echo "===== cuda-python ====="
python3 -c 'import cuda; print("cuda-python OK", cuda.__file__)' 2>&1 | tail -3
echo "===== onnx ====="
python3 -c 'import onnx; print("onnx", onnx.__version__)' 2>&1 | tail -3
echo "===== onnxruntime ====="
python3 -c 'import onnxruntime as ort; print("ort", ort.__version__)' 2>&1 | tail -3
echo "===== onnx-graphsurgeon ====="
python3 -c 'import onnx_graphsurgeon as gs; print("gs OK", gs.__file__)' 2>&1 | tail -3
echo "===== polygraphy ====="
python3 -c 'import polygraphy; print("polygraphy", polygraphy.__version__)' 2>&1 | tail -3
echo "===== modelopt ====="
python3 -c 'import modelopt; print("modelopt", modelopt.__file__)' 2>&1 | tail -3
python3 -c 'import modelopt.onnx; print("modelopt.onnx OK")' 2>&1 | tail -3
echo "===== pip list (quant-related) ====="
python3 -m pip list 2>/dev/null | grep -iE 'onnx|tensorrt|polygraphy|modelopt|nvidia|torch|numpy'
echo "===== trtexec ====="
ls -la /usr/src/tensorrt/bin/trtexec 2>&1
/usr/src/tensorrt/bin/trtexec --version 2>&1 | head -8
echo "===== laya-trt dir ====="
ls -la /home/harvest/laya-trt/ 2>&1 | head -60
echo "===== laya-trt subdirs ====="
ls -la /home/harvest/laya-trt/calib_s512_q/ 2>&1 | head -10
echo "calib count:"; ls /home/harvest/laya-trt/calib_s512_q/ 2>/dev/null | wc -l
echo "===== nvpmodel / jetson_clocks ====="
nvpmodel -q 2>&1 | head -5
cat /sys/devices/platform/gpu.0/railgate_enable 2>/dev/null
echo "===== free ====="
free -g
