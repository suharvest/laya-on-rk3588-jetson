#include <cstddef>
#include <rkllm.h>
#include <dlfcn.h>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <vector>

namespace {
using CreateFn = decltype(&rkllm_createDefaultParam);
using InitFn = decltype(&rkllm_init);
using RunFn = decltype(&rkllm_run);
using ClearFn = decltype(&rkllm_clear_kv_cache);
using DestroyFn = decltype(&rkllm_destroy);

struct Context {
  void* sdk = nullptr;
  LLMHandle handle = nullptr;
  CreateFn create = nullptr;
  InitFn init = nullptr;
  RunFn run = nullptr;
  ClearFn clear = nullptr;
  DestroyFn destroy = nullptr;
  std::mutex mu;
  std::vector<float> hidden;
  int n_tokens = 0;
  int dim = 0;
  int error = 0;
};

int on_result(RKLLMResult* result, void* userdata, LLMCallState state) {
  auto* ctx = static_cast<Context*>(userdata);
  if (!ctx || !result || state == RKLLM_RUN_ERROR) return state == RKLLM_RUN_ERROR ? EIO : 0;
  const auto& hidden = result->last_hidden_layer;
  if (state == RKLLM_RUN_NORMAL && hidden.hidden_states && hidden.num_tokens > 0 && hidden.embd_size > 0) {
    if (hidden.embd_size != 1024 || hidden.num_tokens > 8192) {
      ctx->error = EOVERFLOW;
      return EOVERFLOW;
    }
    ctx->n_tokens = hidden.num_tokens;
    ctx->dim = hidden.embd_size;
    const size_t count = static_cast<size_t>(ctx->n_tokens) * ctx->dim;
    ctx->hidden.assign(hidden.hidden_states, hidden.hidden_states + count);
    for (float value : ctx->hidden) {
      if (!std::isfinite(value)) { ctx->error = ERANGE; return ERANGE; }
    }
  }
  return 0;
}

template <typename T> T symbol(void* handle, const char* name) {
  return reinterpret_cast<T>(dlsym(handle, name));
}
}

extern "C" int kev_open(const char* model_path, int32_t context_len, void** out_context) {
  if (!model_path || !out_context || context_len <= 0 || context_len > 8192) return EINVAL;
  *out_context = nullptr;
  const char* sdk_path = std::getenv("RKLLM_LIB_PATH");
  if (!sdk_path || !*sdk_path) return ENOENT;
  auto ctx = std::make_unique<Context>();
  ctx->sdk = dlopen(sdk_path, RTLD_NOW | RTLD_LOCAL);
  if (!ctx->sdk) return ENOENT;
  ctx->create = symbol<CreateFn>(ctx->sdk, "rkllm_createDefaultParam");
  ctx->init = symbol<InitFn>(ctx->sdk, "rkllm_init");
  ctx->run = symbol<RunFn>(ctx->sdk, "rkllm_run");
  ctx->clear = symbol<ClearFn>(ctx->sdk, "rkllm_clear_kv_cache");
  ctx->destroy = symbol<DestroyFn>(ctx->sdk, "rkllm_destroy");
  if (!ctx->create || !ctx->init || !ctx->run || !ctx->clear || !ctx->destroy) {
    dlclose(ctx->sdk);
    ctx->sdk = nullptr;
    return ENOSYS;
  }
  RKLLMParam param = ctx->create();
  param.model_path = model_path;
  param.max_context_len = context_len;
  param.max_new_tokens = 1;
  param.extend_param.base_domain_id = 1;
  param.extend_param.enabled_cpus_num = 4;
  RKLLMCallback callback{};
  callback.result_callback = on_result;
  callback.result_userdata = ctx.get();
  const int rc = ctx->init(&ctx->handle, &param, &callback);
  if (rc != 0 || !ctx->handle) {
    if (ctx->handle) ctx->destroy(ctx->handle);
    dlclose(ctx->sdk);
    return rc ? rc : EIO;
  }
  *out_context = ctx.release();
  return 0;
}

extern "C" int kev_clear_cache(void* opaque) {
  auto* ctx = static_cast<Context*>(opaque);
  if (!ctx || !ctx->handle) return ENODEV;
  std::lock_guard<std::mutex> lock(ctx->mu);
  return ctx->clear(ctx->handle, 0, nullptr, nullptr);
}

extern "C" int kev_infer(void* opaque, const int32_t* input_ids, int32_t n_tokens,
                         float* hidden_out, int32_t capacity_tokens, int32_t hidden_dim) {
  auto* ctx = static_cast<Context*>(opaque);
  if (!ctx || !ctx->handle || !input_ids || !hidden_out || n_tokens <= 0 ||
      capacity_tokens < n_tokens || hidden_dim != 1024) return EINVAL;
  std::lock_guard<std::mutex> lock(ctx->mu);
  ctx->hidden.clear(); ctx->n_tokens = ctx->dim = ctx->error = 0;
  int rc = ctx->clear(ctx->handle, 0, nullptr, nullptr);
  if (rc != 0) return rc;
  RKLLMInput input{};
  input.role = "user";
  input.enable_thinking = false;
  input.input_type = RKLLM_INPUT_TOKEN;
  input.token_input.input_ids = const_cast<int32_t*>(input_ids);
  input.token_input.n_tokens = static_cast<size_t>(n_tokens);
  RKLLMInferParam infer{};
  infer.mode = RKLLM_INFER_GET_LAST_HIDDEN_LAYER;
  infer.keep_history = 0;
  infer.max_new_tokens = 1;
  rc = ctx->run(ctx->handle, &input, &infer, ctx);
  if (rc != 0) return rc;
  if (ctx->error != 0) return ctx->error;
  if (ctx->n_tokens != n_tokens || ctx->dim != hidden_dim) return EPROTO;
  const size_t count = static_cast<size_t>(n_tokens) * hidden_dim;
  std::memcpy(hidden_out, ctx->hidden.data(), count * sizeof(float));
  for (size_t i = 0; i < count; ++i) if (!std::isfinite(hidden_out[i])) return ERANGE;
  return 0;
}

extern "C" void kev_close(void* opaque) {
  auto* ctx = static_cast<Context*>(opaque);
  if (!ctx) return;
  {
    std::lock_guard<std::mutex> lock(ctx->mu);
    if (ctx->handle && ctx->destroy) ctx->destroy(ctx->handle);
    ctx->handle = nullptr;
    if (ctx->sdk) dlclose(ctx->sdk);
    ctx->sdk = nullptr;
    ctx->create = nullptr;
    ctx->init = nullptr;
    ctx->run = nullptr;
    ctx->clear = nullptr;
    ctx->destroy = nullptr;
  }
  delete ctx;
}
