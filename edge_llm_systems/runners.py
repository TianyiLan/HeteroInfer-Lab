"""实验循环调度：warm-up、参数矩阵遍历、CSV 写入、进度打印。

提供四个主要函数：
- run_warmup_text：文本模型热身，不记录数据
- run_warmup_vision：视觉模型热身（含文本热身），不记录数据
- run_benchmark_text：文本性能测试主循环，实时写 CSV
- run_benchmark_vision：视觉性能测试主循环，实时写 CSV
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any

from edge_llm_systems.profiling import measure_text_single, measure_image_single
from edge_llm_systems.utils import log, append_row_to_csv, generate_run_id

# ──────────────────────────────────────────────────────────────────────────────
# 热身配置：短序列先跑几次，让 CUDA 内核和 cuDNN 优化路径预热
# ──────────────────────────────────────────────────────────────────────────────
WARMUP_CONFIGS_TEXT = [(64, 8), (256, 8), (512, 8)]    # (prompt_len, gen_len)
WARMUP_CONFIGS_VISION = [(224, 8)]                       # (image_resolution, gen_len)

# CSV 字段顺序（raw_runs 和 group_summary 共享，summary 额外加 std 字段）
RAW_FIELDNAMES = [
    "timestamp", "run_id", "group_id", "model_id", "model_hash",
    "input_mode", "test_type",
    "prompt_len", "gen_len", "image_resolution", "image_count",
    "run_index",
    "ttft_ms", "tpot_ms", "total_latency_ms", "tokens_s",
    "image_preprocess_ms", "vision_encode_ms", "projector_ms", "text_prefill_ms",
    "model_load_mem_mb", "peak_mem_mb",
    "kv_pkv_prefill_mb", "kv_pkv_final_mb", "kv_est_mb", "kv_payload_ratio",
    "actual_prompt_len", "actual_gen_len", "image_token_count", "total_input_tokens",
    "finish_reason", "output_text", "output_length", "output_nonempty", "refusal_detected",
    "status", "oom_stage", "message_zh",
]

# text_only 模式下图像相关字段填 "N/A"
_IMAGE_ONLY_FIELDS = [
    "image_resolution", "image_count",
    "image_preprocess_ms", "vision_encode_ms", "projector_ms", "text_prefill_ms",
    "image_token_count", "total_input_tokens",
]

# vision 模式下 prompt_len 对应 actual_prompt_len（文本 token 数）
_TEXT_ONLY_FIELDS = ["prompt_len"]


def _na_image_fields() -> dict:
    """返回图像字段的 N/A 填充字典，用于 text_only 模式。"""
    return {f: "N/A" for f in _IMAGE_ONLY_FIELDS}


def _build_row(
    result: dict,
    run_meta: dict,
    group_id: str,
    prompt_len: int | str,
    gen_len: int,
    image_resolution: int | str,
    run_index: int,
) -> dict:
    """将测量结果和元数据合并为一行 CSV 字典。

    Args:
        result: measure_text_single / measure_image_single 的返回值
        run_meta: 附加元数据（model_id, model_hash, input_mode, test_type 等）
        group_id: 参数组标识符（如 "prompt64_gen32"）
        prompt_len: prompt 长度（text_only）或 "N/A"（vision）
        gen_len: 生成长度
        image_resolution: 图像分辨率或 "N/A"（text_only）
        run_index: 第几次重复（1-based），0 表示均值行

    Returns:
        字段完整的 CSV 行字典
    """
    row = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "run_id": run_meta.get("run_id", generate_run_id()),
        "group_id": group_id,
        "prompt_len": prompt_len,
        "gen_len": gen_len,
        "image_resolution": image_resolution,
        "run_index": run_index,
    }
    row.update(run_meta)   # model_id, model_hash, input_mode, test_type 等
    row.update(result)     # 所有测量指标字段
    return row


def _print_progress(
    run_index: int,
    total_runs: int,
    prompt_len: int | str,
    gen_len: int,
    result: dict,
    image_resolution: int | str = "N/A",
) -> None:
    """打印单次 run 的进度行。

    格式示例：
        [1/3] prompt=64, gen=32 → TTFT=58.3ms, TPOT=49.1ms, peak=5065MB, status=success
        [1/3] img=224, gen=32 → TTFT=312.4ms, TPOT=51.2ms, peak=7230MB, status=success
        [1/3] prompt=2048, gen=128 → OOM at prefill, status=oom
    """
    status = result.get("status", "unknown")
    prefix = f"[{run_index}/{total_runs}]"

    if image_resolution != "N/A":
        # 视觉模式显示图像分辨率
        param_str = f"img={image_resolution}, gen={gen_len}"
    else:
        param_str = f"prompt={prompt_len}, gen={gen_len}"

    if status == "oom":
        oom_stage = result.get("oom_stage", "unknown")
        print(f"{prefix} {param_str} → OOM at {oom_stage}, status=oom")
    else:
        ttft = result.get("ttft_ms")
        tpot = result.get("tpot_ms")
        peak = result.get("peak_mem_mb")
        ttft_str = f"{ttft:.1f}ms" if ttft is not None else "N/A"
        tpot_str = f"{tpot:.1f}ms" if tpot is not None else "N/A"
        peak_str = f"{peak:.0f}MB" if peak is not None else "N/A"
        print(f"{prefix} {param_str} → TTFT={ttft_str}, TPOT={tpot_str}, peak={peak_str}, status={status}")


def _print_group_summary(
    prompt_len: int | str,
    gen_len: int,
    runs: list[dict],
    image_resolution: int | str = "N/A",
) -> None:
    """打印一组重复实验的均值摘要行。

    格式示例：
        ── 组均值 prompt=64, gen=32 → TTFT=58.1ms, TPOT=49.0ms ──
    """
    # 只统计成功的 run
    success_runs = [r for r in runs if r.get("status") == "success"]
    if not success_runs:
        print(f"── 组均值 prompt={prompt_len}, gen={gen_len} → 全部失败或 OOM ──")
        return

    # 计算 TTFT 和 TPOT 均值
    ttft_values = [r["ttft_ms"] for r in success_runs if r.get("ttft_ms") is not None]
    tpot_values = [r["tpot_ms"] for r in success_runs if r.get("tpot_ms") is not None]
    mean_ttft = sum(ttft_values) / len(ttft_values) if ttft_values else None
    mean_tpot = sum(tpot_values) / len(tpot_values) if tpot_values else None

    if image_resolution != "N/A":
        param_str = f"img={image_resolution}, gen={gen_len}"
    else:
        param_str = f"prompt={prompt_len}, gen={gen_len}"

    ttft_str = f"{mean_ttft:.1f}ms" if mean_ttft is not None else "N/A"
    tpot_str = f"{mean_tpot:.1f}ms" if mean_tpot is not None else "N/A"
    print(f"── 组均值 {param_str} → TTFT={ttft_str}, TPOT={tpot_str} ──")


def _compute_group_mean_row(
    runs: list[dict],
    group_id: str,
    run_meta: dict,
    prompt_len: int | str,
    gen_len: int,
    image_resolution: int | str,
) -> dict | None:
    """计算一组重复实验的均值，返回 run_index=0 的汇总行。

    只对成功 run 的数值字段求均值，其他字段取第一个成功 run 的值。
    """
    success_runs = [r for r in runs if r.get("status") == "success"]
    if not success_runs:
        return None

    # 需要求均值的数值字段
    numeric_keys = [
        "ttft_ms", "tpot_ms", "total_latency_ms", "tokens_s",
        "peak_mem_mb", "kv_pkv_prefill_mb", "kv_pkv_final_mb",
        "kv_est_mb", "kv_payload_ratio", "actual_prompt_len", "actual_gen_len",
        "output_length",
        # 视觉字段
        "image_preprocess_ms", "vision_encode_ms", "projector_ms",
        "text_prefill_ms", "image_token_count", "total_input_tokens",
    ]

    first = success_runs[0]
    mean_row = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "run_id": run_meta.get("run_id", generate_run_id()),
        "group_id": group_id,
        "prompt_len": prompt_len,
        "gen_len": gen_len,
        "image_resolution": image_resolution,
        "run_index": 0,   # 0 表示均值行
        "model_load_mem_mb": first.get("model_load_mem_mb"),
        "finish_reason": first.get("finish_reason", ""),
        "output_text": "",  # 均值行不保存文本
        "output_nonempty": first.get("output_nonempty", False),
        "refusal_detected": first.get("refusal_detected", False),
        "status": "success",
        "oom_stage": "none",
        "message_zh": f"均值 ({len(success_runs)} 次成功)",
    }
    mean_row.update(run_meta)

    # 对数值字段求均值（跳过 None 值）
    for key in numeric_keys:
        values = [r.get(key) for r in success_runs if r.get(key) is not None]
        if values:
            mean_val = sum(values) / len(values)
            # 保留与原始字段相同的精度
            if key in ("actual_prompt_len", "actual_gen_len", "output_length",
                       "image_token_count", "total_input_tokens", "image_count"):
                mean_row[key] = int(round(mean_val))
            else:
                mean_row[key] = round(mean_val, 3)
        else:
            mean_row[key] = None

    return mean_row


# ──────────────────────────────────────────────────────────────────────────────
# 热身函数
# ──────────────────────────────────────────────────────────────────────────────

def run_warmup_text(
    model: Any,
    tokenizer: Any,
    device: str,
) -> None:
    """文本模型热身：运行预设短序列，让 CUDA 内核预热。

    热身结果不写入 CSV，只打印状态。

    Args:
        model: 已加载的语言模型
        tokenizer: 对应分词器
        device: 推理设备
    """
    log("[热身] 开始文本模型热身...")
    for i, (prompt_len, gen_len) in enumerate(WARMUP_CONFIGS_TEXT):
        result = measure_text_single(
            model=model,
            tokenizer=tokenizer,
            device=device,
            prompt_len=prompt_len,
            gen_len=gen_len,
            model_load_mem_mb=0.0,  # 热身不需要记录基线
        )
        status = result.get("status", "unknown")
        ttft = result.get("ttft_ms")
        ttft_str = f"{ttft:.1f}ms" if ttft is not None else "N/A"
        log(f"  热身 [{i+1}/{len(WARMUP_CONFIGS_TEXT)}] prompt={prompt_len}, gen={gen_len} "
            f"→ TTFT={ttft_str}, status={status}")

    log("[热身] 文本模型热身完成")


def run_warmup_vision(
    model: Any,
    processor: Any,
    device: str,
    warmup_image: Any,  # PIL.Image
) -> None:
    """视觉模型热身：先跑文本热身，再跑图像热身。

    Args:
        model: 已加载的视觉-语言模型
        processor: 对应的 AutoProcessor
        device: 推理设备
        warmup_image: 用于热身的 PIL.Image
    """
    log("[热身] 开始视觉模型热身...")

    # 先做文本热身（从 processor 获取 tokenizer）
    run_warmup_text(model=model, tokenizer=processor.tokenizer, device=device)

    # 再做图像热身
    for i, (resolution, gen_len) in enumerate(WARMUP_CONFIGS_VISION):
        result = measure_image_single(
            model=model,
            processor=processor,
            device=device,
            image=warmup_image,
            image_resolution=resolution,
            gen_len=gen_len,
            model_load_mem_mb=0.0,
        )
        status = result.get("status", "unknown")
        ttft = result.get("ttft_ms")
        ttft_str = f"{ttft:.1f}ms" if ttft is not None else "N/A"
        log(f"  图像热身 [{i+1}/{len(WARMUP_CONFIGS_VISION)}] resolution={resolution}, gen={gen_len} "
            f"→ TTFT={ttft_str}, status={status}")

    log("[热身] 视觉模型热身完成")


# ──────────────────────────────────────────────────────────────────────────────
# 性能测试主循环
# ──────────────────────────────────────────────────────────────────────────────

def run_benchmark_text(
    model: Any,
    tokenizer: Any,
    device: str,
    prompt_lengths: list[int],
    gen_lengths: list[int],
    repeat: int,
    model_load_mem_mb: float,
    raw_csv_path: str,
    summary_csv_path: str,
    run_meta: dict,
) -> None:
    """文本性能测试主循环，遍历 prompt_length × gen_length 参数矩阵。

    每次 run 完成后立即 append 写入 raw CSV，
    每组 repeat 次完成后计算均值并 append 写入 summary CSV。

    Args:
        model: 已加载的语言模型
        tokenizer: 分词器
        device: 推理设备
        prompt_lengths: 要测试的 prompt 长度列表
        gen_lengths: 要测试的生成长度列表
        repeat: 每个参数组重复次数
        model_load_mem_mb: 模型基线显存
        raw_csv_path: raw_runs CSV 文件路径
        summary_csv_path: group_summary CSV 文件路径
        run_meta: 附加到每行的元数据字典（model_id, model_hash 等）
    """
    total_groups = len(prompt_lengths) * len(gen_lengths)
    group_count = 0

    for prompt_len in prompt_lengths:
        for gen_len in gen_lengths:
            group_count += 1
            group_id = f"prompt{prompt_len}_gen{gen_len}"
            log(f"\n[性能测试] 组 {group_count}/{total_groups}: {group_id}")

            group_runs: list[dict] = []

            for run_idx in range(1, repeat + 1):
                result = measure_text_single(
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    prompt_len=prompt_len,
                    gen_len=gen_len,
                    model_load_mem_mb=model_load_mem_mb,
                )

                # 打印进度
                _print_progress(run_idx, repeat, prompt_len, gen_len, result)

                # 构造并立即写入 raw CSV
                row = _build_row(
                    result=result,
                    run_meta=run_meta,
                    group_id=group_id,
                    prompt_len=prompt_len,
                    gen_len=gen_len,
                    image_resolution="N/A",
                    run_index=run_idx,
                )
                # text_only 模式：图像字段填 N/A
                row.update(_na_image_fields())
                append_row_to_csv(raw_csv_path, row, RAW_FIELDNAMES)

                group_runs.append(result)

            # 组均值行写入 summary CSV
            _print_group_summary(prompt_len, gen_len, group_runs)
            mean_row = _compute_group_mean_row(
                runs=group_runs,
                group_id=group_id,
                run_meta=run_meta,
                prompt_len=prompt_len,
                gen_len=gen_len,
                image_resolution="N/A",
            )
            if mean_row is not None:
                mean_row.update(_na_image_fields())
                append_row_to_csv(summary_csv_path, mean_row, RAW_FIELDNAMES)

    log(f"\n[性能测试] 文本测试完成，结果已写入:\n  raw: {raw_csv_path}\n  summary: {summary_csv_path}")


def run_benchmark_vision(
    model: Any,
    processor: Any,
    device: str,
    images_by_resolution: dict,   # {resolution: PIL.Image}
    gen_lengths: list[int],
    repeat: int,
    model_load_mem_mb: float,
    raw_csv_path: str,
    summary_csv_path: str,
    run_meta: dict,
) -> None:
    """视觉性能测试主循环，遍历 resolution × gen_length 参数矩阵。

    Args:
        model: 已加载的视觉-语言模型
        processor: AutoProcessor
        device: 推理设备
        images_by_resolution: {分辨率: PIL.Image} 字典，图像已预先 resize
        gen_lengths: 要测试的生成长度列表
        repeat: 每个参数组重复次数
        model_load_mem_mb: 模型基线显存
        raw_csv_path: raw_runs CSV 文件路径
        summary_csv_path: group_summary CSV 文件路径
        run_meta: 附加元数据字典
    """
    total_groups = len(images_by_resolution) * len(gen_lengths)
    group_count = 0

    for resolution, image in images_by_resolution.items():
        for gen_len in gen_lengths:
            group_count += 1
            group_id = f"img{resolution}_gen{gen_len}"
            log(f"\n[性能测试] 组 {group_count}/{total_groups}: {group_id}")

            group_runs: list[dict] = []

            for run_idx in range(1, repeat + 1):
                result = measure_image_single(
                    model=model,
                    processor=processor,
                    device=device,
                    image=image,
                    image_resolution=resolution,
                    gen_len=gen_len,
                    model_load_mem_mb=model_load_mem_mb,
                )

                _print_progress(run_idx, repeat, "N/A", gen_len, result,
                                 image_resolution=resolution)

                row = _build_row(
                    result=result,
                    run_meta=run_meta,
                    group_id=group_id,
                    prompt_len="N/A",   # 视觉模式 prompt_len 用 actual_prompt_len
                    gen_len=gen_len,
                    image_resolution=resolution,
                    run_index=run_idx,
                )
                append_row_to_csv(raw_csv_path, row, RAW_FIELDNAMES)
                group_runs.append(result)

            _print_group_summary("N/A", gen_len, group_runs, image_resolution=resolution)
            mean_row = _compute_group_mean_row(
                runs=group_runs,
                group_id=group_id,
                run_meta=run_meta,
                prompt_len="N/A",
                gen_len=gen_len,
                image_resolution=resolution,
            )
            if mean_row is not None:
                append_row_to_csv(summary_csv_path, mean_row, RAW_FIELDNAMES)

    log(f"\n[性能测试] 视觉测试完成，结果已写入:\n  raw: {raw_csv_path}\n  summary: {summary_csv_path}")
