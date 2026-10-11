from pathlib import Path
p=Path("/home/cat/kev-rkllm/persist/evidence/test_stepwise_rkllm.py")
s=p.read_text()
repls=[
("""        ("skip_special_token", ctypes.c_bool),
        ("is_async", ctypes.c_bool),""","""        ("skip_special_token", ctypes.c_bool),
        ("ignore_eos_token", ctypes.c_bool),
        ("is_async", ctypes.c_bool),"""),
("""        ("prompt_cache_params", ctypes.c_void_p),
        ("keep_history", ctypes.c_int),""","""        ("prompt_cache_params", ctypes.c_void_p),
        ("sampling_params", ctypes.c_void_p),
        ("keep_history", ctypes.c_int),
        ("max_new_tokens", ctypes.c_int32),"""),
("""RKLLM_CALLBACK = ctypes.CFUNCTYPE(
    None, ctypes.POINTER(RKLLMResult), ctypes.c_void_p, ctypes.c_int
)""","""RKLLM_CALLBACK = ctypes.CFUNCTYPE(
    ctypes.c_int, ctypes.POINTER(RKLLMResult), ctypes.c_void_p, ctypes.c_int
)

class RKLLMCallback(ctypes.Structure):
    _fields_ = [
        ("result_callback", RKLLM_CALLBACK),
        ("result_userdata", ctypes.c_void_p),
        ("tokenizer_callback", ctypes.c_void_p),
        ("tokenizer_userdata", ctypes.c_void_p),
        ("embed_callback", ctypes.c_void_p),
        ("embed_userdata", ctypes.c_void_p),
    ]"""),
("""    except Exception as e:
        print("CB error: {}".format(e), flush=True)


cb = RKLLM_CALLBACK(callback)""","""    except Exception as e:
        print("CB error: {}".format(e), flush=True)
    return 0


cb = RKLLM_CALLBACK(callback)
cb_struct = RKLLMCallback()
cb_struct.result_callback = cb"""),
("""        RKLLM_CALLBACK,
    ]""","""        ctypes.POINTER(RKLLMCallback),
    ]"""),
("""    rkllm.rkllm_clear_kv_cache.argtypes = [RKLLM_Handle_t]""","""    rkllm.rkllm_clear_kv_cache.argtypes = [RKLLM_Handle_t, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p]"""),
("""    ret = rkllm.rkllm_init(ctypes.byref(handle), ctypes.byref(param), cb)""","""    ret = rkllm.rkllm_init(ctypes.byref(handle), ctypes.byref(param), ctypes.byref(cb_struct))"""),
("""rkllm.rkllm_clear_kv_cache(handle)""","""rkllm.rkllm_clear_kv_cache(handle, 0, None, None)"""),
("""    out = "/home/cat/kev-rkllm/persist/evidence/rkllm_stepwise_results.json""","""    out = "/home/cat/kev-rkllm/persist/evidence/rkllm_stepwise_results-v131.json"""),
]
for a,b in repls:
    if a not in s:
        raise SystemExit("missing replacement: "+repr(a[:60]))
    s=s.replace(a,b)
p.write_text(s)
print("patched",p)
