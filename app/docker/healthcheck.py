from __future__ import annotations

import json
import os
import sys
from urllib.error import URLError
from urllib.request import urlopen

EXPECTED_INPUTS = {
    "input_ids": lambda seq, options: [1, seq],
    "attention_mask": lambda seq, options: [1, seq],
    "marker_pos": lambda seq, options: [1, options],
    "marker_mask": lambda seq, options: [1, options],
    "qtype": lambda seq, options: [1],
}


def _expected(name: str, fallback: str | None = None) -> str | None:
    value = os.getenv(name)
    return value if value not in (None, "") else fallback


def validate_health(data: object, *, expected_backend: str | None, expected_seq: str | None,
                    expected_options: str | None, metadata: object = None) -> str | None:
    if not isinstance(data, dict):
        return "health response is not an object"
    if data.get("status") != "ok" or data.get("model_ready") is not True:
        return f"status={data.get('status')!r}, model_ready={data.get('model_ready')!r}"
    for key, expected in (("backend", expected_backend), ("max_sequence_tokens", expected_seq),
                          ("max_options", expected_options)):
        if expected is not None and str(data.get(key)) != expected:
            return f"{key}={data.get(key)!r}, expected {expected!r}"
    if metadata is not None:
        if not isinstance(metadata, dict) or not isinstance(metadata.get("inputs"), dict):
            return "target metadata has no inputs object"
        try:
            seq = int(expected_seq or metadata["seq_len"])
            options = int(expected_options or metadata["max_options"])
        except (KeyError, TypeError, ValueError):
            return "target metadata has no usable seq/options"
        inputs = metadata["inputs"]
        if set(inputs) != set(EXPECTED_INPUTS):
            return f"target metadata input names={sorted(inputs)!r}"
        for name, shape_fn in EXPECTED_INPUTS.items():
            if inputs.get(name) != shape_fn(seq, options):
                return f"target metadata shape mismatch for {name}: {inputs.get(name)!r}"
    return None


def main() -> int:
    port = os.getenv("PORT", "18768")
    try:
        with urlopen(f"http://127.0.0.1:{port}/api/health", timeout=4) as response:
            data = json.load(response)
        metadata = None
        metadata_path = os.getenv("TARGET_METADATA_PATH")
        if metadata_path:
            with open(metadata_path, encoding="utf-8") as f:
                metadata = json.load(f)
    except (OSError, ValueError, URLError) as exc:
        print(f"healthcheck failed: {exc}", file=sys.stderr)
        return 1
    reason = validate_health(
        data,
        expected_backend=_expected("EXPECTED_BACKEND", os.getenv("DECISION_BACKEND")),
        expected_seq=_expected("EXPECTED_SEQ_LEN", os.getenv("SEQ_LEN")),
        expected_options=_expected("EXPECTED_MAX_OPTIONS", os.getenv("MAX_OPTIONS")),
        metadata=metadata,
    )
    if reason:
        print(f"healthcheck failed: {reason}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
