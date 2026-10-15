# Candidate deployment evidence (sanitized)

- Candidate commit: `d5527aa5b155c23c9f03c61d4dc0ccc8936261ae`
- Deployment ID: `1b90b989-5ea0-4702-8b1d-c30673585060`
- Engine control plane: Spark `sensecraft-engine 0.0.3`, contract `10`; daemon status also reported engine `0.0.3`, instance route profile `agent`, candidate solutions directory loaded.
- The earlier failed deployment event metadata discrepancy (`summary.engine_version=0.1.0` versus Spark binary/daemon `0.0.3`) remains documented in the prior evidence; the successful logs do not contain a conflicting engine version field.
- Deployment command: `sensecraft-engine deploy --params /tmp/deploy-candidate-body.json --skip-verify --yes --detach`; terminal `status ... --json` exit status was `0`.
- Terminal status: `completed`, exit status `0`, started `2026-10-15T05:42:58.737668`, completed `2026-10-15T05:43:58.049134` (about 59.3 seconds).
- Steps: connect, network, OS, Docker, prepare, bootstrap, upload, image reuse, start, health check, after actions all completed. Engine event says `All services healthy` and `Model ready: backend=rkllm model_id=kev`.
- Health: `status=ok`, `model_ready=true`, `backend=rkllm`, `model_id=kev`, `max_sequence_tokens=8192`, `max_options=16`.
- Models: Kev local/default ready (`rkllm`, 8192); retained Laya profile ready (`rknn`, 90, existing local service URL omitted).
- Golden set: 5 fixed IDs, all HTTP 200 and expected selection matched: `jev-001-en set_speed`, `jev-001-zh set_speed`, `jev-002-en status`, `jev-002-zh status`, `jev-003-en start`.
- Overlength guard: the deployed container's own `KevCodec` preflight counted the fixture as `40025` tokens against `8192`; the same request returned HTTP `422 input_too_long` with `actual_tokens=40025` and `max_tokens=8192`, before NPU inference.
- The sanitized deployment request contains `auto_replace_containers=true`, the Spark key-file placeholder, and the existing Laya profile; it contains no `network_region` or `force_offline` request override. Those values were inherited from the candidate device YAML (`network_region: global`, `docker.offline.force_offline: false`).
- Bootstrap provenance correction: the successful round's bootstrap evidence is derived only from the 1b90 engine event stream: seq 38 is the archive checksum message; seq 40-64 are 13 model/shared-file checksum messages; seq 66/68 contain image/config verification and the exact text `skipping docker load`; seq 70 records pinned artifacts ready; seq 74 records before actions completed. Historical bootstrap captures are not assigned to any current deployment ID and are excluded from this public archive. No archive download or `Loaded image` claim is made for the successful round.
- Rollback backup and old image tag remain retained; current service is the candidate image and healthy. No further retry is authorized or performed.
- After confirming no active deployments, the Spark managed daemon was restored with CLI to the pre-operation solutions path `<SPARK_SOLUTIONS_PATH>/solutions`; its runtime now reports that path, `resource_dir: null`, and no busy deployment. Radxa service was not stopped.

Raw private files remain local/Spark-only. Public copies replace host, home, key path, and URL values.
