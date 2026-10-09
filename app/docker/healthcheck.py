from __future__ import annotations

import json
import os
import sys
from urllib.error import URLError
from urllib.request import urlopen


def main() -> int:
    port = os.getenv("PORT", "18768")
    try:
        with urlopen(f"http://127.0.0.1:{port}/api/health", timeout=4) as response:
            data = json.load(response)
    except (OSError, ValueError, URLError) as exc:
        print(f"healthcheck failed: {exc}", file=sys.stderr)
        return 1
    required = {
        "status": "ok",
        "backend": "rknn",
        "model_ready": True,
        "max_sequence_tokens": 90,
        "max_options": 16,
    }
    for key, expected in required.items():
        if data.get(key) != expected:
            print(f"healthcheck failed: {key}={data.get(key)!r}, expected {expected!r}", file=sys.stderr)
            return 1
    return 0


raise SystemExit(main())
