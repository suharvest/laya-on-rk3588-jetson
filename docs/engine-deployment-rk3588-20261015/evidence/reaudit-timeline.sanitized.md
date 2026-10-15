# Reaudited deployment timeline (sanitized)

Source: Spark `sensecraft-engine status` and `sensecraft-engine logs <deployment_id> --since 0`, collected after deployment completion. Host, home, key paths redacted. No new deployment or device write was performed during this audit.

## b49ae360-af9d-421f-89dc-d0a5fe724b2f

- Status: `failed`.
- Deployment started: `2026-10-15T03:57:20.093833`.
- Deployment completed: `2026-10-15T03:57:27.742147`.
- Device checks completed: connect, network, OS, Docker, disk.
- Confirmed device changes in engine events: created remote directory (`seq 24-26`); copied Kev manifest (`seq 30`), bootstrap/verifier (`seq 32`), and archive verifier (`seq 34`).
- `seq 36`: started `Prepare pinned Kev image and model artifacts`.
- `seq 38/40`: bootstrap stderr reports permission denied writing `.cache/rkllm-artifact-plan.tsv` at script line 87.
- `seq 42`: action failed with the same permission error; `seq 48/49`: device/deployment failed.
- No engine event proves an offline-package gate for this deployment. No retained CLI output with a deployment ID proves which command produced the earlier offline-gate text; that prior attribution is therefore `unknown/unattributed` and removed from this timeline.
- No engine event proves any permission repair after this failure. The fact that the next deployment reached bootstrap does not identify who or what changed the permission state.

Raw re-audit event SHA256: `1e4b5e782124e24b1d25af16f786f328e37bf08464a3f48c28d270c36dbb37b1`

## a93c7ccd-ab7a-43f2-80a8-b61d3007eb49

- Status: `failed`.
- Deployment started: `2026-10-15T03:57:57.932663`.
- Deployment completed: `2026-10-15T03:58:27.640163`.
- Device checks completed: connect, network, OS, Docker, disk.
- `seq 38-64`: bootstrap reports archive checksum plus 13 model/shared-file checksum matches.
- `seq 66/68`: exact local image tag/config verified; engine message says `skipping docker load`.
- `seq 76/78`: engine stopped and removed existing containers because `auto_replace_containers=true`.
- `seq 80-84`: uploaded Docker assets.
- `seq 86`: images already existed locally.
- `seq 90`: service start failed with the raw truncated daemon error `No such image: docker.m.daocloud.io/library/edge-command-workbench:rockchip-rkllm-`.
- `seq 96/97`: device/deployment failed.
- The rollback that restored the old service was performed outside these engine events; its separate backup/health evidence remains in the existing private/sanitized records. No rollback command is inferred from this event stream.

Raw re-audit event SHA256: `1ee81d9cc065fc783c25c5e0acc2119894811e90bdc90ca30755aeb2ac456c67`

## 1b90b989-5ea0-4702-8b1d-c30673585060

- Status: `completed`, exit status `0`.
- Deployment started: `2026-10-15T05:42:58.737668`.
- Deployment completed: `2026-10-15T05:43:58.049134`.
- All engine steps completed, including health and after-actions; final messages were `All services healthy` and `Model ready: backend=rkllm model_id=kev`.
- Success engine events are retained in `deploy-1b90b989-events.compact.ndjson`; raw SHA256 `ddd891ecfba626936fc0a5d9eb4067492a442bf211081c3c9dfa8e03f2bb6805`.

## Engine metadata

Spark CLI/binary/daemon reported engine `0.0.3`, contract `10`. The earlier failed deployment create metadata reported `summary.engine_version=0.1.0`; no source currently proves whether that field is a daemon metadata bug or a different internal component, so the discrepancy remains explicit.
