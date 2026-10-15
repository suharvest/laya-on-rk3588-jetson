# RK3588 Edge Command Router deployment evidence

This report records the sanitized deployment of candidate `d5527aa5b155c23c9f03c61d4dc0ccc8936261ae` to a Radxa RK3588 device through `sensecraft-engine` 0.0.3. The Spark daemon also reported engine 0.0.3. Deployment ID `1b90b989-5ea0-4702-8b1d-c30673585060` completed in about 59.3 seconds with terminal exit status 0.

The successful run reused the prepared archive cache and recorded 13 model/shared-file checksum matches. Its engine events also verify the exact image tag and config digest and record that Docker load was skipped. This report does not describe the run as a fresh model download or cold-start build. The candidate image reached `model_ready=true`, `backend=rkllm`, `model_id=kev`, `max_sequence_tokens=8192`, and `max_options=16`. The existing Laya profile remained available. The retained Laya path is a compatibility profile, not a new RK3588 accuracy claim.

## Validation covered

- Five fixed smoke requests returned HTTP 200 and matched the expected selection: `jev-001-en`/`jev-001-zh` → `set_speed`, `jev-002-en`/`jev-002-zh` → `status`, and `jev-003-en` → `start`.
- The 40025-token fixture was rejected by the deployed container before NPU inference with HTTP 422, `input_too_long`, `actual_tokens=40025`, and `max_tokens=8192`.
- These five smoke requests are deployment checks, not a model accuracy benchmark. ASR and voice behavior were not tested.

## Failure and rollback history

The first attempt (`b49ae360`) reached bootstrap but failed because `.cache/rkllm-artifact-plan.tsv` could not be written (`Permission denied`); no offline-package gate is attributed to this deployment. The second attempt (`a93c7ccd`) verified the archive and 13 model/shared-file checksums and skipped Docker load, then failed when Docker mirror rewriting changed the local image reference to a nonexistent `docker.m.daocloud.io/library/...` path at service start. The permission repair between attempts is not established by the retained events. The old service was observed healthy after rollback; no rollback command is inferred here. The successful candidate run used the corrected local-archive handling and was followed by daemon restoration; no further retry was performed.

The successful deployment request contains no `network_region` or `force_offline` override. Those values come from the candidate device YAML. The deployment used the RK3588 remote preset and retained the existing Laya service profile. This report covers RK3588 only; it is not a Jetson-native or RK3576 hardware result.

## Sanitized evidence

- [Successful deployment events](evidence/deploy-1b90b989-events.compact.ndjson)
- [Successful bootstrap events](evidence/bootstrap-1b90-engine-derived.ndjson)
- [Candidate request](evidence/deploy-candidate-request.sanitized.json)
- [Health result](evidence/health.final.json) and [model list](evidence/models.final.json)
- [Five smoke requests](evidence/goldens5.requests.sanitized.ndjson) and [responses](evidence/goldens5.responses.ndjson)
- [Overlength guard evidence](evidence/overlength8192.evidence.txt)
- [Reaudited failure timeline](evidence/reaudit-timeline.sanitized.md), [b49 bootstrap events](evidence/b49ae360.reaudit.sanitized.ndjson), and [a93 bootstrap/image events](evidence/a93c7ccd.reaudit.sanitized.ndjson)
- [Sanitized deployment report](evidence/deployment-1b90b989-report.sanitized.md)

Hostnames, private paths, credentials, and raw/private captures are excluded from this archive.
