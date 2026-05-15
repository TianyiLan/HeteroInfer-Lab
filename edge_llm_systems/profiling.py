"""单次推理性能测量：TTFT / TPOT / KV Cache / 显存。

提供两个主要测量函数：
- measure_text_single：纯文本推理，测量 prefill / decode 阶段耗时
- measure_image_single：视觉-语言推理，额外分项计时视觉编码和投影器

所有 OOM 错误被捕获并记录，函数不抛出异常，确保实验循环不中断。
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
# OOM 阶段枚举：标记 OOM 发生在推理的哪个阶段
# ──────────────────────────────────────────────────────────────────────────────
OOM_STAGES = [
    "none",            # 无 OOM
    "model_load",      # 模型加载时 OOM
    "image_preprocess",# 图像预处理时 OOM
    "vision_encode",   # 视觉编码时 OOM
    "prefill",         # Prefill 阶段 OOM
    "decode",          # Decode 阶段 OOM
    "kv_extract",      # KV cache 提取时 OOM
    "unknown",         # 未知阶段 OOM
]

# 拒绝词列表：检测模型是否拒绝回答
_REFUSAL_PHRASES = [
    "I cannot",
    "I can't",
    "I'm unable",
    "As an AI",
    "I apologize, but",
    "I'm not able to",
]

# 用于生成 prompt 的填充文本
_FILLER_TEXT = "The quick brown fox jumps over the lazy dog. "


def _run_standard_cleanup() -> None:
    """每次测量前的标准 4 行清理序列。

    确保上一次测量的内存统计不污染本次结果：
    1. Python 垃圾回收
    2. 清空 CUDA 显存缓存
    3. 重置峰值显存统计
    4. CUDA 同步（确保所有异步操作完成）
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    reset_peak_memory_stats()
    synchronize_if_cuda()


def _detect_refusal(text: str) -> bool:
    """检测输出文本中是否含有拒绝词。"""
    for phrase in _REFUSAL_PHRASES:
        if phrase in text:
            return True
    return False


def _make_oom_result(
    oom_stage: str,
    actual_prompt_len: int = 0,
    model_load_mem_mb: float = 0.0,
) -> dict:
    """构造 OOM 时的标准返回字典，所有数值字段填 None。"""
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


def measure_text_single(
    model: Any,
    tokenizer: Any,
    device: str,
    prompt_len: int,
    gen_len: int,
    model_load_mem_mb: float,
) -> dict:
    """测量单次纯文本推理的性能指标。

    分别测量 prefill（TTFT）和 decode（TPOT）两个阶段，
    并收集 KV cache 大小和显存占用信息。

    Args:
        model: 已加载的语言模型
        tokenizer: 对应的分词器
        device: 推理设备，如 "cuda"
        prompt_len: 目标 prompt token 数
        gen_len: 目标生成 token 数
        model_load_mem_mb: 模型加载后的基线显存（MB），原样记录

    Returns:
        包含完整性能指标的字典，OOM 时各数值字段为 None
    """
    # ── 步骤 1：标准清理 ──────────────────────────────────────────────────────
    _run_standard_cleanup()

    # ── 步骤 2：构造 prompt ───────────────────────────────────────────────────
    # 重复填充文本直到足够长，再用 tokenizer 截断到精确长度
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
    actual_prompt_len = input_ids.shape[-1]  # 实际 tokenize 后的 prompt 长度

    # ── 步骤 3：Prefill（计时 TTFT）─────────────────────────────────────────
    try:
        synchronize_if_cuda()
        t0 = time.perf_counter()

        with torch.no_grad():
            # use_cache=True 让模型返回 past_key_values 供后续 decode 复用
            outputs = model(input_ids, use_cache=True)

        synchronize_if_cuda()
        ttft_ms = (time.perf_counter() - t0) * 1000.0

        past_kv = outputs.past_key_values
        # 获取 prefill 后的 KV cache 大小（纯 tensor payload）
        try:
            kv_pkv_prefill_mb = kv_cache_size_from_past_key_values_mb(past_kv)
        except Exception:
            kv_pkv_prefill_mb = None

        # 下一个 token 是 prefill 输出 logits 中概率最高的 token
        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            return _make_oom_result("prefill", actual_prompt_len, model_load_mem_mb)
        return {**_make_oom_result("unknown", actual_prompt_len, model_load_mem_mb),
                "message_zh": f"Prefill 异常: {e}"}

    # ── 步骤 4：Decode 循环（计时 TPOT）──────────────────────────────────────
    generated_ids: list[int] = []
    step_times: list[float] = []
    actual_gen_len = 0
    finish_reason = "max_tokens"

    try:
        for step in range(gen_len):
            synchronize_if_cuda()
            t_step_start = time.perf_counter()

            with torch.no_grad():
                # 每步只传入上一个 token，利用 past_key_values 避免重复计算
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

            # 遇到 EOS token 则提前停止
            if tokenizer.eos_token_id is not None and token_id == tokenizer.eos_token_id:
                finish_reason = "natural"
                break

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            # Decode 阶段 OOM：保留已完成的步数信息
            return _make_oom_result("decode", actual_prompt_len, model_load_mem_mb)

    # ── 步骤 5：KV cache 最终大小 ─────────────────────────────────────────────
    try:
        kv_pkv_final_mb = kv_cache_size_from_past_key_values_mb(past_kv)
    except Exception:
        kv_pkv_final_mb = None

    # ── 步骤 6：汇总指标 ──────────────────────────────────────────────────────
    tpot_ms = (sum(step_times) / len(step_times)) if step_times else None
    total_latency_ms = ttft_ms + (tpot_ms or 0.0) * actual_gen_len

    # 估算 KV cache 理论大小（基于模型配置）
    try:
        kv_est_mb = estimate_kv_cache_mb(model, actual_prompt_len + actual_gen_len)
    except Exception:
        kv_est_mb = None

    # 峰值显存（重置统计后的最大值）
    peak_mem_mb = get_peak_gpu_memory_mb() if torch.cuda.is_available() else None

    # KV payload 占峰值显存的比例
    if peak_mem_mb and peak_mem_mb > 0 and kv_pkv_final_mb is not None:
        kv_payload_ratio = kv_pkv_final_mb / peak_mem_mb
    else:
        kv_payload_ratio = 0.0

    # 解码生成的文本
    output_text = tokenizer.decode(generated_ids, skip_special_tokens=True) if generated_ids else ""

    # tokens/s 基于 TPOT 计算
    tokens_s = tokens_per_second(tpot_ms) if tpot_ms and tpot_ms > 0 else None

    return {
        "ttft_ms": round(ttft_ms, 3),
        "tpot_ms": round(tpot_ms, 3) if tpot_ms is not None else None,
        "total_latency_ms": round(total_latency_ms, 3),
        "tokens_s": round(tokens_s, 2) if tokens_s is not None else None,
        "model_load_mem_mb": model_load_mem_mb,
        "peak_mem_mb": round(peak_mem_mb, 1) if peak_mem_mb is not None else None,
        "kv_pkv_prefill_mb": round(kv_pkv_prefill_mb, 3) if kv_pkv_prefill_mb is not None else None,
        "kv_pkv_final_mb": round(kv_pkv_final_mb, 3) if kv_pkv_final_mb is not None else None,
        "kv_est_mb": round(kv_est_mb, 3) if kv_est_mb is not None else None,
        "kv_payload_ratio": round(kv_payload_ratio, 4) if kv_payload_ratio is not None else None,
        "actual_prompt_len": actual_prompt_len,
        "actual_gen_len": actual_gen_len,
        "finish_reason": finish_reason,
        "output_text": output_text,
        "output_length": len(output_text),
        "output_nonempty": len(output_text) > 0,
        "refusal_detected": _detect_refusal(output_text),
        "status": "success",
        "oom_stage": "none",
        "message_zh": "成功",
    }


def measure_image_single(
    model: Any,
    processor: Any,
    device: str,
    image: Any,               # PIL.Image
    image_resolution: int,
    gen_len: int,
    model_load_mem_mb: float,
) -> dict:
    """测量单次视觉-语言推理的性能指标。

    在 text-only 基础上额外分项计时：
    - 图像预处理（processor 调用）
    - 视觉编码器 forward（hook 方式）
    - 多模态投影器 forward（hook 方式）
    - 纯文本 prefill 耗时（总 prefill - 视觉部分）

    Args:
        model: 已加载的视觉-语言模型
        processor: 对应的 AutoProcessor
        device: 推理设备
        image: PIL.Image 对象，已 resize 到 image_resolution
        image_resolution: 图像分辨率（正方形边长），用于记录
        gen_len: 目标生成 token 数
        model_load_mem_mb: 模型加载后基线显存

    Returns:
        包含完整性能指标的字典（含视觉分项计时）
    """
    # ── 步骤 1：标准清理 ──────────────────────────────────────────────────────
    _run_standard_cleanup()

    # 固定 prompt，视觉任务统一使用简短描述性问题
    text_prompt = "Describe this image briefly."

    # ── 步骤 2：图像预处理（计时）────────────────────────────────────────────
    try:
        synchronize_if_cuda()
        t_preprocess_start = time.perf_counter()

        # processor 同时处理图像和文本，返回模型所需的所有 tensor
        inputs = processor(
            images=image,
            text=text_prompt,
            return_tensors="pt",
        )

        synchronize_if_cuda()
        image_preprocess_ms = (time.perf_counter() - t_preprocess_start) * 1000.0

        # 将所有 tensor 移动到目标设备
        inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            return {**_make_oom_result("image_preprocess", 0, model_load_mem_mb),
                    **_vision_extra_defaults()}
        return {**_make_oom_result("unknown", 0, model_load_mem_mb),
                **_vision_extra_defaults(),
                "message_zh": f"图像预处理失败: {e}"}

    # ── 步骤 3：统计文本 prompt 和 image token 数量 ────────────────────────────
    input_ids = inputs.get("input_ids")
    text_prompt_token_count = input_ids.shape[-1] if input_ids is not None else 0
    actual_prompt_len = text_prompt_token_count

    # 获取 LLaMA 3.2 Vision 使用的 image token id
    try:
        image_token_id = processor.tokenizer.convert_tokens_to_ids("<|image|>")
        if input_ids is not None:
            image_token_count = (input_ids == image_token_id).sum().item()
        else:
            image_token_count = 0
    except Exception:
        image_token_count = 0

    total_input_tokens = image_token_count + text_prompt_token_count

    # ── 步骤 4：注册 forward hook 分项计时 ────────────────────────────────────
    # hook 存储各阶段的起止时间，测量后立即移除
    hook_times: dict[str, float] = {}
    hooks = []

    def _make_pre_hook(name: str):
        """创建 pre-forward hook，记录该模块开始时间。"""
        def hook(module, args, kwargs=None):
            synchronize_if_cuda()
            hook_times[f"{name}_start"] = time.perf_counter()
        return hook

    def _make_post_hook(name: str):
        """创建 post-forward hook，计算该模块耗时。"""
        def hook(module, input, output):
            synchronize_if_cuda()
            start = hook_times.get(f"{name}_start", time.perf_counter())
            hook_times[name] = (time.perf_counter() - start) * 1000.0
        return hook

    # 探测 vision tower 的正确属性名（不同版本可能不同）
    vision_module = None
    for attr in ("vision_tower", "vision_model", "vision_encoder"):
        if hasattr(model, attr):
            vision_module = getattr(model, attr)
            break

    if vision_module is not None:
        hooks.append(vision_module.register_forward_pre_hook(_make_pre_hook("vision_encode")))
        hooks.append(vision_module.register_forward_hook(_make_post_hook("vision_encode")))

    # 探测多模态投影器的正确属性名
    projector_module = None
    for attr in ("multi_modal_projector", "mm_projector", "vision_projection"):
        if hasattr(model, attr):
            projector_module = getattr(model, attr)
            break

    if projector_module is not None:
        hooks.append(projector_module.register_forward_pre_hook(_make_pre_hook("projector")))
        hooks.append(projector_module.register_forward_hook(_make_post_hook("projector")))

    # ── 步骤 5：Prefill（计时整体 TTFT）──────────────────────────────────────
    try:
        synchronize_if_cuda()
        t_prefill_start = time.perf_counter()

        with torch.no_grad():
            outputs = model(**inputs, use_cache=True)

        synchronize_if_cuda()
        ttft_ms = (time.perf_counter() - t_prefill_start) * 1000.0

        past_kv = outputs.past_key_values

        # 获取 prefill 后 KV cache 大小
        try:
            kv_pkv_prefill_mb = kv_cache_size_from_past_key_values_mb(past_kv)
        except Exception:
            kv_pkv_prefill_mb = None

        # 下一个 token
        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    except RuntimeError as e:
        # 移除所有 hook 后再返回，防止 hook 持续影响模型
        for h in hooks:
            h.remove()
        if "out of memory" in str(e).lower():
            return {**_make_oom_result("prefill", actual_prompt_len, model_load_mem_mb),
                    **_vision_extra_defaults(image_resolution=image_resolution,
                                             image_token_count=image_token_count,
                                             total_input_tokens=total_input_tokens,
                                             image_preprocess_ms=image_preprocess_ms)}
        return {**_make_oom_result("unknown", actual_prompt_len, model_load_mem_mb),
                **_vision_extra_defaults(), "message_zh": f"Prefill 异常: {e}"}
    finally:
        # hook 已完成使命，立即移除，避免影响后续 decode
        for h in hooks:
            h.remove()

    # 从 hook 记录中提取分项时间
    vision_encode_ms = hook_times.get("vision_encode")
    projector_ms = hook_times.get("projector")
    # 文本 prefill 时间 = 总 prefill - 视觉编码 - 投影器
    if vision_encode_ms is not None and projector_ms is not None:
        text_prefill_ms = ttft_ms - vision_encode_ms - projector_ms
    else:
        text_prefill_ms = ttft_ms

    # ── 步骤 6：Decode 循环 ───────────────────────────────────────────────────
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
                        image_token_count=image_token_count,
                        total_input_tokens=total_input_tokens,
                        image_preprocess_ms=image_preprocess_ms,
                        vision_encode_ms=vision_encode_ms,
                        projector_ms=projector_ms,
                        text_prefill_ms=text_prefill_ms,
                    )}

    # ── 步骤 7：汇总指标 ──────────────────────────────────────────────────────
    try:
        kv_pkv_final_mb = kv_cache_size_from_past_key_values_mb(past_kv)
    except Exception:
        kv_pkv_final_mb = None

    tpot_ms = (sum(step_times) / len(step_times)) if step_times else None
    total_latency_ms = ttft_ms + (tpot_ms or 0.0) * actual_gen_len

    # 视觉模式只估算 text decoder 部分的 KV cache（vision encoder 无 decoder KV）
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
        # ── 基础性能指标 ──
        "ttft_ms": round(ttft_ms, 3),
        "tpot_ms": round(tpot_ms, 3) if tpot_ms is not None else None,
        "total_latency_ms": round(total_latency_ms, 3),
        "tokens_s": round(tokens_s, 2) if tokens_s is not None else None,
        "model_load_mem_mb": model_load_mem_mb,
        "peak_mem_mb": round(peak_mem_mb, 1) if peak_mem_mb is not None else None,
        # ── KV Cache 指标 ──
        "kv_pkv_prefill_mb": round(kv_pkv_prefill_mb, 3) if kv_pkv_prefill_mb is not None else None,
        "kv_pkv_final_mb": round(kv_pkv_final_mb, 3) if kv_pkv_final_mb is not None else None,
        "kv_est_mb": round(kv_est_mb, 3) if kv_est_mb is not None else None,
        "kv_payload_ratio": round(kv_payload_ratio, 4) if kv_payload_ratio is not None else None,
        # ── Token 统计 ──
        "actual_prompt_len": actual_prompt_len,
        "actual_gen_len": actual_gen_len,
        # ── 输出质量 ──
        "finish_reason": finish_reason,
        "output_text": output_text,
        "output_length": len(output_text),
        "output_nonempty": len(output_text) > 0,
        "refusal_detected": _detect_refusal(output_text),
        # ── 状态 ──
        "status": "success",
        "oom_stage": "none",
        "message_zh": "成功",
        # ── 视觉专属字段 ──
        "image_count": 1,
        "image_resolution": image_resolution,
        "image_token_count": image_token_count,
        "total_input_tokens": total_input_tokens,
        "image_preprocess_ms": round(image_preprocess_ms, 3),
        "vision_encode_ms": round(vision_encode_ms, 3) if vision_encode_ms is not None else None,
        "projector_ms": round(projector_ms, 3) if projector_ms is not None else None,
        "text_prefill_ms": round(text_prefill_ms, 3) if text_prefill_ms is not None else None,
    }


def _vision_extra_defaults(
    image_resolution: int = 0,
    image_token_count: int = 0,
    total_input_tokens: int = 0,
    image_preprocess_ms: float | None = None,
    vision_encode_ms: float | None = None,
    projector_ms: float | None = None,
    text_prefill_ms: float | None = None,
) -> dict:
    """构造视觉任务专属字段的默认值字典，用于 OOM 返回。"""
    return {
        "image_count": 1,
        "image_resolution": image_resolution,
        "image_token_count": image_token_count,
        "total_input_tokens": total_input_tokens,
        "image_preprocess_ms": image_preprocess_ms,
        "vision_encode_ms": vision_encode_ms,
        "projector_ms": projector_ms,
        "text_prefill_ms": text_prefill_ms,
    }
