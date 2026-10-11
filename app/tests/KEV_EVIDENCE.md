# Kev integration evidence

- Pinned tokenizer: `/home/harvest/tmp/jev-20261010/kev/base/tokenizer.json`
- Tokenizer SHA256: `fe000e3ed39ed12b8d2481d527d44f93c65d37e87645d2dcc80d1bf9d50d2927`
- Pinned SDK header: `/home/harvest/tmp/jev-20261010/kev-rkllm/rknn-llm-v1.3.1/rkllm-runtime/Linux/librkllm_api/include/rkllm.h`
- Reference source: `experiments/jev-20261010/kev/rk3588-rkllm/runtime_batch.cpp`
- C bridge compile check:
  `g++ -std=c++17 -Wall -Wextra -Werror -fPIC -shared -I /home/harvest/tmp/jev-20261010/kev-rkllm/rknn-llm-v1.3.1/rkllm-runtime/Linux/librkllm_api/include -o /tmp/libkev_runtime.so app/native/kev_runtime.cpp -ldl`
- Python evidence: `uv run pytest -q tests/test_kev.py tests/test_native.py tests/test_decision_service.py tests/test_healthcheck.py` -> 29 passed.
- Full app evidence: uv run pytest -q -> 35 passed, 39 warnings (FastAPI shutdown hook deprecation).
- Device/NPU validation is not covered here; it requires root's hardware worker.

- Actual official-source encoder parity used pinned source /tmp/kev-source-audit-20261010/kev/model.py and tokenizer /home/harvest/tmp/jev-20261010/kev/runtime/checkpoint/tokenizer.json; five frozen cases matched exact IDs, option_end_indices, and decision_index: app/tests/kev_goldens.json.
- Head source: /home/harvest/tmp/jev-20261010/kev/model/head.pt, SHA256 f400bd12802b2b105ae45d6b03774a158a3db4fccff42413734ddca2e5c920b6; converted artifact /home/harvest/tmp/jev-20261010/kev-rkllm/app-artifacts/head.npz, SHA256 533a21e940e69a570d5cf12f50f1bef27bdb2847395a288fa6cbfc3f9a1a7a7b, temperature 2.3510958125672174, Q/K (256,1024), bias (256,).
- SDK library for Radxa QA: /home/harvest/tmp/jev-20261010/kev-rkllm/rknn-llm-v1.3.1/rkllm-runtime/Linux/librkllm_api/aarch64/librkllmrt.so, SHA256 f25e9b099db08aaacd0a3ac62b4697d3951d6ae61ae41ea09f6702cfa89eb32c.
- Actual dynamic bridge lifecycle built a fake SDK against the pinned header, loaded it through kev_open, ran full hidden callback, then called kev_close twice: official header fake SDK bridge lifecycle passed 3072.

- Cleanup regression evidence: full app pytest -> 34 passed, 37 warnings (FastAPI shutdown hook deprecation). Tests cover head-load failure releasing an opened runtime, application shutdown release, remote catalog identity mismatch, and explicit remote response identity mismatch.
- Current source fingerprints: kev_runtime.cpp ddad908b6bffb7cac599c43614cacd5eb95d0f668d79ff30278ea7e8c73a3131; decision.py a980e8d060c95c4f7793e7ad124ec588ae4d30b0c6ed08d1b5c72a0d867e4341; app.py de17de2db663fbfeffdae9820c0fca34ca7ecba00792972f8aad1f25fc92d565; convert_kev_head.py 49e6b641da565ef3f7d52bafb12eb42e3b7351693ce486ab5ab63757a3f046fb.

- Fallback regression: DecisionService initialization failure now initializes _profiles; /api/models returns a valid fallback catalog.
- Updated fingerprints: app.py d4fa04bb7e0ea6a5e936f028681b10ce7ca6e343c5c43af3f593b6ce65d3338a; test_decision_service.py 1a8e9569b1739e5670ffa38274b41c7192970f2c6e53fa0d57cf1e043f333200.
