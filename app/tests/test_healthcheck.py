from docker.healthcheck import validate_health


def good():
    return {"status": "ok", "model_ready": True, "backend": "tensorrt",
            "max_sequence_tokens": 90, "max_options": 16}


def metadata():
    return {"seq_len": 90, "max_options": 16, "inputs": {
        "input_ids": [1, 90], "attention_mask": [1, 90],
        "marker_pos": [1, 16], "marker_mask": [1, 16], "qtype": [1]}}


def test_healthcheck_requires_ready_model():
    data = good(); data["model_ready"] = False
    assert validate_health(data, expected_backend="tensorrt", expected_seq="90", expected_options="16")


def test_healthcheck_rejects_backend_and_shape_mismatch():
    assert validate_health(good(), expected_backend="rknn", expected_seq="90", expected_options="16")
    assert validate_health(good(), expected_backend="tensorrt", expected_seq="512", expected_options="16")


def test_healthcheck_rejects_missing_or_wrong_input_shape():
    bad = metadata(); del bad["inputs"]["qtype"]
    assert validate_health(good(), expected_backend="tensorrt", expected_seq="90", expected_options="16", metadata=bad)
    bad = metadata(); bad["inputs"]["input_ids"] = [1, 512]
    assert validate_health(good(), expected_backend="tensorrt", expected_seq="90", expected_options="16", metadata=bad)
