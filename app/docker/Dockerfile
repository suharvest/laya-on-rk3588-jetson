ARG PYTHON_IMAGE=python:3.11-slim-bookworm

FROM ${PYTHON_IMAGE} AS builder

ARG UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
ENV UV_INDEX_URL=${UV_INDEX_URL} \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_HTTP_TIMEOUT=120 \
    UV_CONCURRENT_DOWNLOADS=2

WORKDIR /src
RUN python -m venv /opt/uv-bootstrap \
    && /opt/uv-bootstrap/bin/pip install --no-cache-dir --index-url "${UV_INDEX_URL}" uv
ENV PATH=/opt/uv-bootstrap/bin:${PATH}
ARG BUILD_KEV_RKLLM=0
ARG KEV_SDK_DIR=docker/vendor/rkllm-v1.3.1
COPY pyproject.toml uv.lock ./
COPY docker/vendor ./docker/vendor
COPY native/kev_runtime.cpp ./native/kev_runtime.cpp
RUN uv sync --frozen --no-dev --no-install-project
COPY docker/vendor/rknn_toolkit_lite2-2.3.2-cp311-cp311-manylinux_2_17_aarch64.manylinux2014_aarch64.whl /tmp/
RUN uv pip install --python /opt/venv/bin/python --no-deps /tmp/rknn_toolkit_lite2-2.3.2-cp311-cp311-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
RUN mkdir -p /out/kev \
    && if [ "${BUILD_KEV_RKLLM}" = "1" ]; then \
         apt-get update \
         && apt-get install -y --no-install-recommends g++ \
         && test -f /src/${KEV_SDK_DIR}/rkllm.h \
         && g++ -std=c++17 -O2 -fPIC -shared /src/native/kev_runtime.cpp \
              -I/src/${KEV_SDK_DIR} -ldl -pthread -o /out/kev/libkev_runtime.so \
         && rm -rf /var/lib/apt/lists/*; \
       else \
         printf "Kev helper not built; set BUILD_KEV_RKLLM=1 with pinned SDK headers.\n"; \
       fi

FROM ${PYTHON_IMAGE} AS runtime

ARG BUILD_KEV_RKLLM=0
RUN if [ "${BUILD_KEV_RKLLM}" = "1" ]; then \
      apt-get update \
      && apt-get install -y --no-install-recommends libstdc++6 libgomp1 \
      && rm -rf /var/lib/apt/lists/*; \
    fi

ENV PATH=/opt/venv/bin:${PATH} \
    PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    LD_LIBRARY_PATH=/opt/kev:/opt/rknn:/usr/lib/aarch64-linux-gnu:/usr/lib \
    DECISION_BACKEND=rknn \
    MODEL_PATH=/models/model.rknn \
    TOKENIZER_PATH=/models/tokenizer \
    MODEL_METADATA_PATH= \
    SEQ_LEN=90 \
    MAX_OPTIONS=16 \
    RKNN_CORE=auto \
    MODEL_ID=laya \
    KEV_HEAD_PATH= \
    RKLLM_LIB_PATH=/opt/kev/librkllmrt.so \
    RKLLM_HELPER_PATH=/opt/kev/libkev_runtime.so \
    MODEL_SERVICES_JSON=

WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /out/kev /opt/kev
COPY edge_workbench ./edge_workbench
COPY web ./web
COPY docker/healthcheck.py ./docker/healthcheck.py
COPY docker/entrypoint.py ./docker/entrypoint.py
COPY docker/vendor/RKNN-TOOLKIT2-LICENSE /usr/share/doc/rknn-toolkit-lite2/LICENSE
COPY docker/vendor/PROVENANCE.txt /usr/share/doc/rknn-toolkit-lite2/PROVENANCE.txt
COPY docker/vendor/rkllm-v1.3.1/RKLLM-SDK-LICENSE /usr/share/doc/rkllm/RKLLM-SDK-LICENSE
COPY docker/vendor/rkllm-v1.3.1/NOTICE /usr/share/doc/rkllm/NOTICE
EXPOSE 18768
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD ["python", "/app/docker/healthcheck.py"]
ENTRYPOINT ["python", "/app/docker/entrypoint.py"]
