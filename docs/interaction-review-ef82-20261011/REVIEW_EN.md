# ef82 Independent Runtime Review

Review time: 2026-10-11 12:20–12:25 (Asia/Shanghai)
Runtime URL: <http://100.77.150.16:18772/>
Deployment status: the first browser route review used d870; that phase did not update the 18772 container or publish UI. The later real-container QA used the ef82 candidate on 18775, as documented below.

## Version and route limitation

- Source file: `/tmp/edge-workbench-web-ef82f83/app.js`
- SHA-256: `02c44a4da2b0a9e298aec9ef9154a5c5488e29875aff729fd0f26348dda0ed9e`
- Only the isolated review browser routed `/app.js` to this file.
- Real `/api/models`, `/api/decide`, and `/api/health/remote` calls remained on port 18772.
- This report verifies the route-replaced browser behavior. It does not prove that the d870 container contains ef82.

## Candidate build association

- Local candidate source commit: `ef82f8360c2b787f97c958f89c92fe569c62ecb1`.
- Candidate image: `edge-command-workbench:rockchip-rkllm-0.3.1-review`; image/config blob SHA-256: `d7a81c4f8ef3eaa2c21560b2d37939c6088357ec0c1d41c3089ff5fbc3836de9`.
- Candidate archive SHA-256: `251cb7fef080f489bf311b37c5bddbba1cd72e887e71f10ce12fc25a1fe64f77`.
- The real API calls in the first route-review phase used the d870 deployment on port 18772; the container-QA section below started only an independent QA container on 18775 and did not update the 18772 production service.

## Confirmed

- ZH/EN result labels switch correctly. Existing results remain visible and language switching does not add a history entry. Evidence: [ef82-zh-result.yaml](./ef82-zh-result.yaml), [ef82-en-result.yaml](./ef82-en-result.yaml).
- Both real models completed decisions: `model.rknn → laya` (90 tokens) and `model.rkllm → kev` (8192 tokens). Evidence: [ef82-zh-result.yaml](./ef82-zh-result.yaml), [ef82-rkllm-result-en.yaml](./ef82-rkllm-result-en.yaml).
- Test and connect shows failure for an unreachable address; clearing it and retrying restores the connected state. Evidence: [ef82-connect-fail.yaml](./ef82-connect-fail.yaml), [ef82-connect-recovered.yaml](./ef82-connect-recovered.yaml).
- Mouse double-click protection was re-tested: the first click disables the button, the second is rejected, and request count is 1. The previous mouse duplicate-click finding is withdrawn. Evidence: [ef82-mouse-click-repro.txt](./ef82-mouse-click-repro.txt).
- Two consecutive Ctrl+Enter submissions produce only one `/api/decide` request. Evidence: [ef82-ctrl-enter-repro.txt](./ef82-ctrl-enter-repro.txt).
- During rapid model switching, stale health requests are aborted and final metadata matches the last selection. Evidence: [ef82-modelhealth-race-network.log](./ef82-modelhealth-race-network.log), [ef82-modelhealth-race.txt](./ef82-modelhealth-race.txt).

## Coverage boundary

- No real voice service was configured. The recording entry remained disabled. This review did not treat a synthetic WebSocket final message as real ASR evidence; real ASR requires a connected service.
- The 502 Console entry after connection recovery was produced by the intentionally unreachable test address and is diagnostic evidence, not a release-state claim.

## Evidence artifacts

- Full-page completed-result screenshots (Kev / rkllm / 8192): [ef82-completed-zh.png](./ef82-completed-zh.png), [ef82-completed-en.png](./ef82-completed-en.png)
- Two-frame completed-result language-switch GIF, non-realtime: [ef82-completed-zh-en.gif](./ef82-completed-zh-en.gif)
- Completed response, result text, and history counts: [ef82-completed-zh-runtime.txt](./ef82-completed-zh-runtime.txt), [ef82-completed-en-runtime.txt](./ef82-completed-en-runtime.txt)
- Raw request payload: [ef82-request-payload.txt](./ef82-request-payload.txt)
- Console: [ef82-console-latest.log](./ef82-console-latest.log)

SHA-256:

```text
ef82-completed-zh.png  fc330da7cbb79a1e219a1199551aa2a3f531e2552210c5c26a382e52a082f16b
ef82-completed-en.png  9ae42e4d7b0ba2390426f47639482426474531003ac5d272d97cd8d8efdee958
ef82-completed-zh-en.gif  324c79531ad390f01888756aee950ce830fdda186629cda424da516c124e5696
```

## Real container QA

- QA container: `edge-workbench-rkllm-ef82-qa`, port `18775`, independent tag `edge-command-workbench:rockchip-rkllm-0.3.1-review`; no existing container was stopped or replaced. See [container-info.txt](./container-qa/container-info.txt).
- Container health returned `model_ready=true`, `model_id=kev`, `backend=rkllm`, and `max_sequence_tokens=8192`; the catalog returned both ready Kev/RKLLM 8192 and Laya/RKNN 90 profiles. Raw responses: [health.json](./container-qa/health.json), [models.json](./container-qa/models.json).
- The same short payload produced HTTP 200 for both routes: Kev selected `cmd-3` with actual `model_id=kev`; Laya selected `cmd-3` with actual `model_id=laya`. Raw requests/responses: [kev-request.json](./container-qa/kev-request.json), [kev-response.json](./container-qa/kev-response.json), [laya-request.json](./container-qa/laya-request.json), [laya-response.json](./container-qa/laya-response.json).
- Measured `inference_ms` was `1338.0779240396805` for Kev/RKLLM and `457.5591559987515` for Laya/RKNN. HTTP timing and curl exit records: [kev-http.txt](./container-qa/kev-http.txt), [laya-http.txt](./container-qa/laya-http.txt), [kev-start.txt](./container-qa/kev-start.txt), [kev-end.txt](./container-qa/kev-end.txt), [laya-start.txt](./container-qa/laya-start.txt), [laya-end.txt](./container-qa/laya-end.txt).
- The archive config blob SHA is `d7a81c4f8ef3eaa2c21560b2d37939c6088357ec0c1d41c3089ff5fbc3836de9`; the Docker 29 loaded image ID is `sha256:8acbe96dc6548ef18eb424f405d68d812a0245e02006dbfd1f9801f600820f16`. The difference is recorded in [archive-config-vs-loaded.txt](./container-qa/archive-config-vs-loaded.txt).
- The candidate `/app/web/app.js` SHA is `02c44a4da2b0a9e298aec9ef9154a5c5488e29875aff729fd0f26348dda0ed9e`; the helper SHA is `022e0102f9dd4c8839ba7acbc68f03d8811bf020cde84d6cf1343829cd66f919`. See [embedded-sha.txt](./container-qa/embedded-sha.txt). Startup logs are in [container.log](./container-qa/container.log).
- This QA used existing Radxa model/tokenizer/head/runtime files as read-only mounts. It deployed only the independent QA container; it did not update the production service, publish to HF, or change source/math.
## Direct browser re-test (18775)

- This evidence comes from a fresh browser session against `http://100.77.150.16:18775/` with no Playwright route replacement. The `direct-browser/` directory contains the actual page assets, snapshots, network/request records, and completed-result screenshots.
- Actual `app.js` SHA-256: `02c44a4da2b0a9e298aec9ef9154a5c5488e29875aff729fd0f26348dda0ed9e`; actual `index.html` SHA-256: `75f2779f55ba3b2af3030e3739986b8217af7e69c7bc691e2a4f99795e4676ff`.
- The Kev Chinese completed decision returned HTTP 200 with actual model `kev`, backend `rkllm`, and max context `8192`; switching to EN preserved the result without increasing history. The Laya real decision returned HTTP 200 with actual model `laya`, backend `rknn`, and max context `90`.
- The Ctrl+Enter pending guard showed `Running · 0s`; the button was disabled after the first keypress and the second keypress produced no extra request (request count `1`).
- This candidate was tested only in the current QA container and was not published to HF. Real ASR was not configured or validated. The GIF is a non-realtime montage of two completed screenshots and does not represent live recording or inference duration.

Evidence entry: [direct-browser/REVIEW.md](./direct-browser/REVIEW.md).

