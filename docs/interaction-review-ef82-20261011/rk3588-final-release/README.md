# RK3588 release verification (2026-10-11)

This directory records the final stock deployment of source `ef82f8360c2b787f97c958f89c92fe569c62ecb1` using the ready inventory at HF revision `db0545065d4da3598d009407fc3bd0c7632ca199`.

- Service URL: `http://<device-ip>:18768`
- Image tag: `edge-command-workbench:rockchip-rkllm-0.3.1-review`
- Archive SHA-256: `251cb7fef080f489bf311b37c5bddbba1cd72e887e71f10ce12fc25a1fe64f77`
- Image config SHA-256: `d7a81c4f8ef3eaa2c21560b2d37939c6088357ec0c1d41c3089ff5fbc3836de9`
- Bootstrap exit: `0`
- Artifact accounting: 13 target-filtered files reused from the SHA-verified cache and 1 new image archive downloaded from `https://hf-mirror.com`; this is not a fresh 14-file download.
- Health reported `model_ready=true`, `backend=rkllm`, `model_id=kev`, `max_sequence_tokens=8192`, `max_options=16`.
- The same short decision request returned HTTP 200 for `model_id=kev` (`rkllm`, 8192) and `model_id=laya` (`rknn`, 90). The Laya service was an existing remote profile at `http://127.0.0.1:18771`.
- The helper was verified inside the image at `/opt/kev/libkev_runtime.so`, SHA-256 `022e0102f9dd4c8839ba7acbc68f03d8811bf020cde84d6cf1343829cd66f919`; no external helper mount was used.

The earlier `a556`/`d870` 14-file bootstrap record remains historical. Voice and ASR were not part of this validation. The evidence contains selected metadata and raw request/response/timing files; it excludes model weights and image archives.

## Current support scope

RK3576 RKNN and the existing RK3576 Kev 2K path remain supported; the 8K cat-remote run is recorded separately because its guard rejects over-limit input after 2088 actual tokens. RK3588 Kev 8K is now verified by the stock release evidence in this directory. Jetson remains a historical native Kev path and is not claimed as a ready deployment target here.
