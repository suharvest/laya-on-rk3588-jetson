# 重建清单

产出 [ARTIFACTS.md](ARTIFACTS.md) 里那些产物的两个环境的依赖清单，以及构建时的上游版本。

依赖清单是 `pip freeze` 的原文照抄，这份文件的意义就在于此：**Rockchip 交付物的图形态依赖
导出环境**，换环境重建必须重跑 16 条样本的验收，才能声称等价。

## 1. 环境

### 1.1 构建机 `<export-env>/.venv`（RK 链路的导出环境）

Python 版本：

```
Python 3.12.13
```

`python -m pip freeze` **未产出包列表，报错原文如下**：

```
<export-env>/.venv/bin/python: No module named pip
```

该 venv 由 uv 0.11.14 创建（`pyvenv.cfg`，原文见 EVIDENCE），未安装 pip。改用 uv 读取等价清单 `uv pip list --python <export-env>/.venv/bin/python`（uv 路径 `~/.local/bin/uv`，未在 PATH 中，须写全路径）：

```
Package                Version
---------------------- ---------
annotated-doc          0.0.5
anyio                  4.15.1
certifi                2026.7.22
click                  8.5.0
cuda-bindings          13.4.3
cuda-pathfinder        1.8.2
cuda-toolkit           13.0.3.0
filelock               4.0.5
flatbuffers            25.12.19
fsspec                 2026.9.0
h11                    0.16.0
hf-xet                 1.6.0
httpcore               1.0.9
httpx                  0.28.1
huggingface-hub        1.33.0
idna                   3.20
jinja2                 3.1.6
laya                   0.3.21
markdown-it-py         4.2.0
markupsafe             3.0.3
mdurl                  0.1.2
ml-dtypes              0.6.0
mpmath                 1.3.0
networkx               3.7
numpy                  2.5.3
nvidia-cublas          13.1.1.3
nvidia-cuda-cupti      13.0.85
nvidia-cuda-nvrtc      13.0.88
nvidia-cuda-runtime    13.0.96
nvidia-cudnn-cu13      9.24.0.43
nvidia-cufft           12.0.0.61
nvidia-cufile          1.15.1.6
nvidia-curand          10.4.0.35
nvidia-cusolver        12.0.4.66
nvidia-cusparse        12.6.3.3
nvidia-cusparselt-cu13 0.8.1
nvidia-nccl-cu13       2.30.7
nvidia-nvjitlink       13.4.92
nvidia-nvshmem-cu13    3.4.5
nvidia-nvtx            13.0.85
onnx                   1.23.0
onnx-ir                1.0.0
onnxruntime            1.30.0
onnxscript             0.7.2
onnxsim                0.7.3
packaging              26.3
protobuf               7.36.2
pygments               2.21.0
pyyaml                 6.0.3
regex                  2026.9.10
rich                   15.0.0
safetensors            0.8.0
setuptools             84.0.0
shellingham            1.5.4
sympy                  1.14.0
tokenizers             0.23.2
torch                  2.14.0
tqdm                   4.70.1
transformers           5.17.0
triton                 3.8.0
typer                  0.27.2
typing-extensions      4.16.0
```

已知预期核实结果：**torch 2.14.0+cu130 与 transformers 5.17.0 均核实为真。**

- `uv pip list` 显示 `torch 2.14.0`（不含本地版本后缀），运行时自报为 `torch 2.14.0+cu130`，`torch.version.cuda` = `13.0`。
- `transformers` 运行时自报 `5.17.0`。
- CUDA 13 运行时栈齐备（`cuda-toolkit 13.0.3.0`、`nvidia-cublas 13.1.1.3`、`nvidia-cudnn-cu13 9.24.0.43`、`nvidia-nccl-cu13 2.30.7`）。

另：site-packages 的 `dist-info` 目录清单与上表逐项一致（原始输出见 EVIDENCE），可交叉印证。

### 1.2 构建机 `<convert-env>`（RKNN 转换环境）

Python 版本：

```
Python 3.12.3
```

`pip freeze` 完整输出：

```
annotated-doc==0.0.5
anyio==4.15.1
certifi==2026.7.22
charset-normalizer==3.5.1
click==8.5.0
fast-histogram==0.14
filelock==3.32.5
flatbuffers==25.12.19
fsspec==2026.7.0
h11==0.16.0
hf-xet==1.6.0
httpcore==1.0.9
httpx==0.28.1
huggingface_hub==0.36.2
idna==3.20
Jinja2==3.1.6
markdown-it-py==4.2.0
MarkupSafe==3.0.3
mdurl==0.1.2
ml_dtypes==0.5.4
mpmath==1.3.0
networkx==3.6.1
numpy==1.26.4
nvidia-cublas-cu12==12.1.3.1
nvidia-cuda-cupti-cu12==12.1.105
nvidia-cuda-nvrtc-cu12==12.1.105
nvidia-cuda-runtime-cu12==12.1.105
nvidia-cudnn-cu12==9.1.0.70
nvidia-cufft-cu12==11.0.2.54
nvidia-curand-cu12==10.3.2.106
nvidia-cusolver-cu12==11.4.5.107
nvidia-cusparse-cu12==12.1.0.106
nvidia-nccl-cu12==2.20.5
nvidia-nvjitlink-cu12==12.9.86
nvidia-nvtx-cu12==12.1.105
onnx==1.16.1
onnxruntime==1.26.0
opencv-python==4.11.0.86
packaging==26.3
pillow==12.3.0
protobuf==4.25.4
psutil==7.2.2
Pygments==2.21.0
PyYAML==6.0.3
regex==2026.9.29
requests==2.34.2
rich==15.0.0
rknn-toolkit2 @ file://<local-wheel-dir>/rknn-toolkit2/packages/x86_64/rknn_toolkit2-2.3.2-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl#sha256=504d7fcf6a792dfa874e25155de0afb5d6650f04a1a0d49b0886934bec4741ef
ruamel.yaml==0.19.1
safetensors==0.8.0
scipy==1.17.1
setuptools==80.10.2
shellingham==1.5.4
sympy==1.14.0
tokenizers==0.21.4
torch==2.4.0
tqdm==4.70.0
transformers==4.48.0
triton==3.0.0
typer==0.27.2
typing_extensions==4.16.0
urllib3==2.8.0
```

已知预期核实结果：**rknn-toolkit2 2.3.2 与 transformers 4.48.x 均核实为真。**

- `rknn-toolkit2` 以本地 wheel 安装，版本段为 `rknn_toolkit2-2.3.2-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl`，路径 `<local-wheel-dir>/rknn-toolkit2/packages/x86_64/`。
- `transformers==4.48.0`（运行时自报 `4.48.0`）。
- `torch==2.4.0`，运行时自报 `2.4.0+cu121`，`torch.version.cuda` = `12.1`。

> `rknn-toolkit2` 是以本地 wheel 安装的，不在 PyPI 上。复现时需自备该 wheel 或改用官方发布渠道。

### 1.3 `<export-env>/pyproject.toml` 与 `uv.lock`

| 文件 | 字节数 | md5 |
|---|---|---|
| `<export-env>/pyproject.toml` | 236 | `501482b5a610f2a626196bb60aebfbf1` |
| `<export-env>/uv.lock` | 89518 | `a7d6189e7b75b55b3e6f398b0a5ba352` |

`pyproject.toml` 全文：

```toml
[project]
name = "env-laya"
version = "0.1.0"
requires-python = ">=3.12,<3.13"
dependencies = [
    "laya @ file://<workspace>/laya",
    "torch",
    "onnx",
    "onnxruntime",
    "onnxscript",
    "onnxsim",
    "numpy",
]
```

> 该 `pyproject.toml` 依赖 `laya @ file://<workspace>/laya`（源码目录软引用，非 PyPI）。重建导出环境需要该 laya 源码目录仍在。

---

## 2. 上游出处

### 2.1 HuggingFace `convaiinnovations/laya-multilingual`

| 字段 | 值 |
|---|---|
| `sha`（当前 revision） | `e4e9ddf21a7b1903b7acffd8814ad4307bf63a67` |
| `cardData.license` | `apache-2.0` |
| `lastModified` | `2026-09-24T05:39:24.000Z` |
| `createdAt` | `2026-09-19T03:23:34.000Z` |
| `gated` | `false` |

取值来源说明：`hf-mirror.com/api/models/...` 本次返回 **HTTP 308**（重定向、空 body），未取到 JSON；实际数值取自 `https://huggingface.co/api/models/convaiinnovations/laya-multilingual`（HTTP 200）。任务书给的那条命令经镜像端点未能出结果。

### 2.2 GitHub `NandhaKishorM/laya`

| 字段 | 值 |
|---|---|
| `main` HEAD sha | `6d942c92081fbc139e736bbd9ac0023223c29b7f` |
| HEAD commit 日期 | `2026-09-29T17:36:09Z` |
| HEAD commit 首行 | `chore: bump version to 0.3.22` |
| license | `Apache-2.0`（Apache License 2.0） |
| `default_branch` | `main` |
| `pushed_at` | `2026-09-29T17:46:22Z` |

该仓库另有分支（非 HEAD）：`refs/heads/research` = `28d43add7e47ce502489c9433310d55276c64e0f`，以及两条 dependabot 分支。

---

