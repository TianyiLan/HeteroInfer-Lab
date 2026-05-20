"""内部模块：共享推理 + telemetry 测量（不直接对外暴露）。

本模块从原 profiling.py 抽取，是 memory_profiler 与 efficiency_profiler
共用的底层实现。每次推理一次性产出 memory + efficiency 全部字段，
由上层 profiler 按字段子集过滤后写入对应 CSV。

公开函数（package-internal，前缀不带下划线但视为内部用）：
  - measure_text_single  : 纯文本推理，返回完整 telemetry dict
  - measure_image_single : 视觉-语言推理，返回完整 telemetry dict（含视觉分项）

外部代码请通过 edge_llm_systems.memory_profiler / efficiency_profiler 调用，
不要直接 import 本模块。
"""

from __future__ import annotations

import gc
import time
from typing import Any

import torch

from edge_llm_systems.kv_cache import (
    estimate_kv_cache_mb,
    kv_cache_size_from_past_key_values_mb,
)
from edge_llm_systems.cuda_utils import synchronize_if_cuda, reset_peak_memory_stats
from edge_llm_systems.memory import get_peak_gpu_memory_mb
from edge_llm_systems.metrics import tokens_per_second

# ──────────────────────────────────────────────────────────────────────────────
# OOM 阶段枚举
# ──────────────────────────────────────────────────────────────────────────────
OOM_STAGES = [
    "none",
    "model_load",
    "image_preprocess",
    "vision_encode",
    "prefill",
    "decode",
    "kv_extract",
    "unknown",
]

_REFUSAL_PHRASES = [
    "I cannot",
    "I can't",
    "I'm unable",
    "As an AI",
    "I apologize, but",
    "I'm not able to",
]

_FILLER_TEXT = "The quick brown fox jumps over the lazy dog. "


def _run_standard_cleanup() -> None:
    """每次测量前的标准 4 行清理序列。"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    reset_peak_memory_stats()
    synchronize_if_cuda()


def _detect_refusal(text: str) -> bool:
    for phrase in _REFUSAL_PHRASES:
        if phrase in text:
            return True
    return False


def _make_oom_result(
    oom_stage: str,
    actual_prompt_len: int = 0,
    model_load_mem_mb: float = 0.0,
) -> dict:
    """OOM 时的标准返回字典，所有数值字段填 None。"""
    return {
        "ttft_ms": None,
        "tpot_ms": None,
        "total_latency_ms": None,
        "tokens_s": None,
        "model_load_mem_mb": model_load_mem_mb,
        "peak_mem_mb": None,
        "kv_pkv_prefill_mb": None,
        "kv_pkv_final_mb": None,
        "kv_est_mb": None,
        "kv_payload_ratio": None,
        "actual_prompt_len": actual_prompt_len,
        "actual_gen_len": 0,
        "finish_reason": "error",
        "output_text": "",
        "output_length": 0,
        "output_nonempty": False,
        "refusal_detected": False,
        "status": "oom",
        "oom_stage": oom_stage,
        "message_zh": f"OOM 发生于 {oom_stage} 阶段",
    }


def _vision_extra_defaults(
    image_resolution: int = 0,
    image_count: int = 1,
    image_token_count: int = 0,
    total_input_tokens: int = 0,
    image_preprocess_ms: float | None = None,
    vision_encode_ms: float | None = None,
    projector_ms: float | None = None,
    text_prefill_ms: float | None = None,
) -> dict:
    """视觉任务专属字段的默认值字典（OOM 返回时填充）。"""
    return {
        "image_count": image_count,
        "image_resolution": image_resolution,
        "image_token_count": image_token_count,
        "total_input_tokens": total_input_tokens,
        "image_preprocess_ms": image_preprocess_ms,
        "vision_encode_ms": vision_encode_ms,
        "projector_ms": projector_ms,
        "text_prefill_ms": text_prefill_ms,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 公开接口（package-internal）
# ──────────────────────────────────────────────────────────────────────────────

def measure_text_single(
    model: Any,
    tokenizer: Any,
    device: str,
    prompt_len: int,
    gen_len: int,
    model_load_mem_mb: float,
) -> dict:
    """单次纯文本推理，返回完整 telemetry dict（memory + efficiency 全字段）。

    Args:
        model: 已加载的语言模型
        tokenizer: 对应分词器
        device: "cuda" 或 "cpu"
        prompt_len: 目标 prompt token 数
        gen_len: 目标生成 token 数
        model_load_mem_mb: 模型加载后的基线显存（MB）

    Returns:
        包含 ttft_ms / tpot_ms / peak_mem_mb / kv_* 等所有字段的字典。
        OOM 时各数值字段为 None，status="oom"，oom_stage 标记发生阶段。
    """
    _run_standard_cleanup()

    repeated_text = _FILLER_TEXT * (prompt_len // len(_FILLER_TEXT.split()) + 10)
    try:
        inputs = tokenizer(
            repeated_text,
            return_tensors="pt",
            truncation=True,
            max_length=prompt_len,
        )
    except Exception as e:
        return {**_make_oom_result("unknown", 0, model_load_mem_mb),
                "message_zh": f"Tokenize 失败: {e}"}

    input_ids = inputs["input_ids"].to(device)
    actual_prompt_len = input_ids.shape[-1]

    # ── Prefill ──
    try:
        synchronize_if_cuda()
        t0 = time.perf_counter()

        with torch.no_grad():
            outputs = model(input_ids, use_cache=True)

        synchronize_if_cuda()
        ttft_ms = (time.perf_counter() - t0) * 1000.0

        past_kv = outputs.past_key_values
        try:
            kv_pkv_prefill_mb = kv_cache_size_from_past_key_values_mb(past_kv)
        except Exception:
            kv_pkv_prefill_mb = None

        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            return _make_oom_result("prefill", actual_prompt_len, model_load_mem_mb)
        return {**_make_oom_result("unknown", actual_prompt_len, model_load_mem_mb),
                "message_zh": f"Prefill 异常: {e}"}

    # ── Decode ──
    generated_ids: list[int] = []
    step_times: list[float] = []
    actual_gen_len = 0
    finish_reason = "max_tokens"

    try:
        for step in range(gen_len):
            synchronize_if_cuda()
            t_step_start = time.perf_counter()

            with torch.no_grad():
                step_outputs = model(
                    input_ids=next_token,
                    past_key_values=past_kv,
                    use_cache=True,
                )

            synchronize_if_cuda()
            step_time_ms = (time.perf_counter() - t_step_start) * 1000.0
            step_times.append(step_time_ms)

            past_kv = step_outputs.past_key_values
            next_token = step_outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            token_id = next_token.item()
            generated_ids.append(token_id)
            actual_gen_len += 1

            if tokenizer.eos_token_id is not None and token_id == tokenizer.eos_token_id:
                finish_reason = "natural"
                break

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            return _make_oom_result("decode", actual_prompt_len, model_load_mem_mb)

    # ── 汇总 ──
    try:
        kv_pkv_final_mb = kv_cache_size_from_past_key_values_mb(past_kv)
    except Exception:
        kv_pkv_final_mb = None

    tpot_ms = (sum(step_times) / len(step_times)) if step_times else None
    total_latency_ms = ttft_ms + (tpot_ms or 0.0) * actual_gen_len

    try:
        kv_est_mb = estimate_kv_cache_mb(model, actual_prompt_len + actual_gen_len)
    except Exception:
        kv_est_mb = None

    peak_mem_mb = get_peak_gpu_memory_mb() if torch.cuda.is_available() else None

    if peak_mem_mb and peak_mem_mb > 0 and kv_pkv_final_mb is not None:
        kv_payload_ratio = kv_pkv_final_mb / peak_mem_mb
    else:
        kv_payload_ratio = 0.0

    output_text = tokenizer.decode(generated_ids, skip_special_tokens=True) if generated_ids else ""
    tokens_s = tokens_per_second(tpot_ms) if tpot_ms and tpot_ms > 0 else None

    return {
        # 基础性能（efficiency）
        "ttft_ms": round(ttft_ms, 3),
        "tpot_ms": round(tpot_ms, 3) if tpot_ms is not None else None,
        "total_latency_ms": round(total_latency_ms, 3),
        "tokens_s": round(tokens_s, 2) if tokens_s is not None else None,
        # 显存（memory）
        "model_load_mem_mb": model_load_mem_mb,
        "peak_mem_mb": round(peak_mem_mb, 1) if peak_mem_mb is not None else None,
        "kv_pkv_prefill_mb": round(kv_pkv_prefill_mb, 3) if kv_pkv_prefill_mb is not None else None,
        "kv_pkv_final_mb": round(kv_pkv_final_mb, 3) if kv_pkv_final_mb is not None else None,
        "kv_est_mb": round(kv_est_mb, 3) if kv_est_mb is not None else None,
        "kv_payload_ratio": round(kv_payload_ratio, 4) if kv_payload_ratio is not None else None,
        # Token 统计
        "actual_prompt_len": actual_prompt_len,
        "actual_gen_len": actual_gen_len,
        # 输出质量
        "finish_reason": finish_reason,
        "output_text": output_text,
        "output_length": len(output_text),
        "output_nonempty": len(output_text) > 0,
        "refusal_detected": _detect_refusal(output_text),
        # 状态
        "status": "success",
        "oom_stage": "none",
        "message_zh": "成功",
    }


def measure_image_single(
    model: Any,
    processor: Any,
    device: str,
    image: Any,                  # PIL.Image
    image_resolution: int,
    gen_len: int,
    model_load_mem_mb: float,
    image_count: int = 1,
) -> dict:
    """单次视觉-语言推理，返回完整 telemetry dict（含视觉分项计时）。

    在 text-only 基础上额外分项计时：
      - 图像预处理（processor 调用）
      - 视觉编码器 forward（forward hook）
      - 多模态投影器 forward（forward hook）
      - 纯文本 prefill 耗时（总 prefill - 视觉部分）
    """
    _run_standard_cleanup()

    text_prompt = "Describe this image briefly."

    # ── 图像预处理 ──
    try:
        synchronize_if_cuda()
        t_preprocess_start = time.perf_counter()

        inputs = processor(
            images=image,
            text=text_prompt,
            return_tensors="pt",
        )

        synchronize_if_cuda()
        image_preprocess_ms = (time.perf_counter() - t_preprocess_start) * 1000.0

        inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            return {**_make_oom_result("image_preprocess", 0, model_load_mem_mb),
                    **_vision_extra_defaults(image_resolution=image_resolution, image_count=image_count)}
        return {**_make_oom_result("unknown", 0, model_load_mem_mb),
                **_vision_extra_defaults(image_resolution=image_resolution, image_count=image_count),
                "message_zh": f"图像预处理失败: {e}"}

    input_ids = inputs.get("input_ids")
    text_prompt_token_count = input_ids.shape[-1] if input_ids is not None else 0
    actual_prompt_len = text_prompt_token_count

    try:
        image_token_id = processor.tokenizer.convert_tokens_to_ids("<|image|>")
        if input_ids is not None:
            image_token_count = (input_ids == image_token_id).sum().item()
        else:
            image_token_count = 0
    except Exception:
        image_token_count = 0

    total_input_tokens = image_token_count + text_prompt_token_count

    # ── 注册分项计时 hook ──
    hook_times: dict[str, float] = {}
    hooks = []

    def _make_pre_hook(name: str):
        def hook(module, args, kwargs=None):
            synchronize_if_cuda()
            hook_times[f"{name}_start"] = time.perf_counter()
        return hook

    def _make_post_hook(name: str):
        def hook(module, input, output):
            synchronize_if_cuda()
            start = hook_times.get(f"{name}_start", time.perf_counter())
            hook_times[name] = (time.perf_counter() - start) * 1000.0
        return hook

    vision_module = None
    for attr in ("vision_tower", "vision_model", "vision_encoder"):
        if hasattr(model, attr):
            vision_module = getattr(model, attr)
            break
    if vision_module is not None:
        hooks.append(vision_module.register_forward_pre_hook(_make_pre_hook("vision_encode")))
        hooks.append(vision_module.register_forward_hook(_make_post_hook("vision_encode")))

    projector_module = None
    for attr in ("multi_modal_projector", "mm_projector", "vision_projection"):
        if hasattr(model, attr):
            projector_module = getattr(model, attr)
            break
    if projector_module is not None:
        hooks.append(projector_module.register_forward_pre_hook(_make_pre_hook("projector")))
        hooks.append(projector_module.register_forward_hook(_make_post_hook("projector")))

    # ── Prefill ──
    try:
        synchronize_if_cuda()
        t_prefill_start = time.perf_counter()

        with torch.no_grad():
            outputs = model(**inputs, use_cache=True)

        synchronize_if_cuda()
        ttft_ms = (time.perf_counter() - t_prefill_start) * 1000.0

        past_kv = outputs.past_key_values

        try:
            kv_pkv_prefill_mb = kv_cache_size_from_past_key_values_mb(past_kv)
        except Exception:
            kv_pkv_prefill_mb = None

        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    except RuntimeError as e:
        for h in hooks:
            h.remove()
        if "out of memory" in str(e).lower():
            return {**_make_oom_result("prefill", actual_prompt_len, model_load_mem_mb),
                    **_vision_extra_defaults(image_resolution=image_resolution,
                                             image_count=image_count,
                                             image_token_count=image_token_count,
                                             total_input_tokens=total_input_tokens,
                                             image_preprocess_ms=image_preprocess_ms)}
        return {**_make_oom_result("unknown", actual_prompt_len, model_load_mem_mb),
                **_vision_extra_defaults(image_resolution=image_resolution, image_count=image_count),
                "message_zh": f"Prefill 异常: {e}"}
    finally:
        for h in hooks:
            h.remove()

    vision_encode_ms = hook_times.get("vision_encode")
    projector_ms = hook_times.get("projector")
    if vision_encode_ms is not None and projector_ms is not None:
        text_prefill_ms = ttft_ms - vision_encode_ms - projector_ms
    else:
        text_prefill_ms = ttft_ms

    # ── Decode ──
    generated_ids: list[int] = []
    step_times: list[float] = []
    actual_gen_len = 0
    finish_reason = "max_tokens"

    try:
        for step in range(gen_len):
            synchronize_if_cuda()
            t_step = time.perf_counter()

            with torch.no_grad():
                step_outputs = model(
                    input_ids=next_token,
                    past_key_values=past_kv,
                    use_cache=True,
                )

            synchronize_if_cuda()
            step_times.append((time.perf_counter() - t_step) * 1000.0)

            past_kv = step_outputs.past_key_values
            next_token = step_outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            token_id = next_token.item()
            generated_ids.append(token_id)
            actual_gen_len += 1

            eos_id = processor.tokenizer.eos_token_id
            if eos_id is not None and token_id == eos_id:
                finish_reason = "natural"
                break

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            return {**_make_oom_result("decode", actual_prompt_len, model_load_mem_mb),
                    **_vision_extra_defaults(
                        image_resolution=image_resolution,
                        image_count=image_count,
                        image_token_count=image_token_count,
                        total_input_tokens=total_input_tokens,
                        image_preprocess_ms=image_preprocess_ms,
                        vision_encode_ms=vision_encode_ms,
                        projector_ms=projector_ms,
                        text_prefill_ms=text_prefill_ms,
                    )}

    # ── 汇总 ──
    try:
        kv_pkv_final_mb = kv_cache_size_from_past_key_values_mb(past_kv)
    except Exception:
        kv_pkv_final_mb = None

    tpot_ms = (sum(step_times) / len(step_times)) if step_times else None
    total_latency_ms = ttft_ms + (tpot_ms or 0.0) * actual_gen_len

    try:
        kv_est_mb = estimate_kv_cache_mb(model, total_input_tokens + actual_gen_len)
    except Exception:
        kv_est_mb = None

    peak_mem_mb = get_peak_gpu_memory_mb() if torch.cuda.is_available() else None

    if peak_mem_mb and peak_mem_mb > 0 and kv_pkv_final_mb is not None:
        kv_payload_ratio = kv_pkv_final_mb / peak_mem_mb
    else:
        kv_payload_ratio = 0.0

    output_text = processor.tokenizer.decode(generated_ids, skip_special_tokens=True) if generated_ids else ""
    tokens_s = tokens_per_second(tpot_ms) if tpot_ms and tpot_ms > 0 else None

    return {
        # efficiency
        "ttft_ms": round(ttft_ms, 3),
        "tpot_ms": round(tpot_ms, 3) if tpot_ms is not None else None,
        "total_latency_ms": round(total_latency_ms, 3),
        "tokens_s": round(tokens_s, 2) if tokens_s is not None else None,
        # memory
        "model_load_mem_mb": model_load_mem_mb,
        "peak_mem_mb": round(peak_mem_mb, 1) if peak_mem_mb is not None else None,
        "kv_pkv_prefill_mb": round(kv_pkv_prefill_mb, 3) if kv_pkv_prefill_mb is not None else None,
        "kv_pkv_final_mb": round(kv_pkv_final_mb, 3) if kv_pkv_final_mb is not None else None,
        "kv_est_mb": round(kv_est_mb, 3) if kv_est_mb is not None else None,
        "kv_payload_ratio": round(kv_payload_ratio, 4) if kv_payload_ratio is not None else None,
        # token 统计
        "actual_prompt_len": actual_prompt_len,
        "actual_gen_len": actual_gen_len,
        # 输出
        "finish_reason": finish_reason,
        "output_text": output_text,
        "output_length": len(output_text),
        "output_nonempty": len(output_text) > 0,
        "refusal_detected": _detect_refusal(output_text),
        # 状态
        "status": "success",
        "oom_stage": "none",
        "message_zh": "成功",
        # 视觉专属
        "image_count": image_count,
        "image_resolution": image_resolution,
        "image_token_count": image_token_count,
        "total_input_tokens": total_input_tokens,
        "image_preprocess_ms": round(image_preprocess_ms, 3),
        "vision_encode_ms": round(vision_encode_ms, 3) if vision_encode_ms is not None else None,
        "projector_ms": round(projector_ms, 3) if projector_ms is not None else None,
        "text_prefill_ms": round(text_prefill_ms, 3) if text_prefill_ms is not None else None,
    }
