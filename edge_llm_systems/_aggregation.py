"""内部模块：CSV 行构造、进度打印、组聚合等共享辅助。

被 memory_profiler / efficiency_profiler / runners 共用。
单独成模块以避免循环依赖。
"""

from __future__ import annotations

import datetime
from typing import Any

from edge_llm_systems.utils import generate_run_id

# ──────────────────────────────────────────────────────────────────────────────
# 数值字段集合（用于均值聚合）
# ──────────────────────────────────────────────────────────────────────────────

NUMERIC_KEYS_MEMORY = [
    "model_load_mem_mb", "peak_mem_mb",
    "kv_pkv_prefill_mb", "kv_pkv_final_mb",
    "kv_est_mb", "kv_payload_ratio",
]

NUMERIC_KEYS_EFFICIENCY = [
    "ttft_ms", "tpot_ms", "total_latency_ms", "tokens_s",
    "image_preprocess_ms", "vision_encode_ms", "projector_ms", "text_prefill_ms",
    "actual_prompt_len", "actual_gen_len",
    "image_token_count", "total_input_tokens",
    "output_length",
]

INT_KEYS = {
    "actual_prompt_len", "actual_gen_len", "output_length",
    "image_token_count", "total_input_tokens", "image_count",
}


def build_row(
    result: dict,
    run_meta: dict,
    group_id: str,
    prompt_len: int | str,
    gen_len: int,
    image_resolution: int | str,
    image_count: int | str,
    run_index: int,
) -> dict:
    """将测量结果 + 元数据合并为一行 CSV 字典。

    上层负责按字段列表过滤后再写入对应 CSV。

    Args:
        result: profiling_core.measure_*_single 返回值
        run_meta: 附加元数据（model_id / model_hash / modality / run_id 等）
        group_id: 参数组标识符（如 "prompt64_gen32" / "vision_1img_336_prompt256_gen64"）
        prompt_len, gen_len: 配置参数
        image_resolution, image_count: vision 模式专属
        run_index: 1-based 重复次数；0 = 均值行
    """
    row = {
        "timestamp":        datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "run_id":           run_meta.get("run_id", generate_run_id()),
        "group_id":         group_id,
        "prompt_len":       prompt_len,
        "gen_len":          gen_len,
        "image_resolution": image_resolution,
        "image_count":      image_count,
        "run_index":        run_index,
    }
    row.update(run_meta)
    row.update(result)
    return row


def print_progress(
    run_index: int,
    total_runs: int,
    prompt_len: int | str,
    gen_len: int,
    result: dict,
    image_resolution: int | str = "N/A",
    image_count: int | str = "N/A",
    metric_prefix: str = "",         # "[mem]" / "[eff]" / ""
) -> None:
    """单次 run 进度行。"""
    status = result.get("status", "unknown")
    prefix = f"{metric_prefix}[{run_index}/{total_runs}]" if metric_prefix else f"[{run_index}/{total_runs}]"

    if image_resolution != "N/A":
        param_str = f"img={image_resolution}×{image_count}, gen={gen_len}"
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


def print_group_summary(
    prompt_len: int | str,
    gen_len: int,
    runs: list[dict],
    image_resolution: int | str = "N/A",
    image_count: int | str = "N/A",
    metric_prefix: str = "",
) -> None:
    """组重复均值摘要行。"""
    success_runs = [r for r in runs if r.get("status") == "success"]

    if image_resolution != "N/A":
        param_str = f"img={image_resolution}×{image_count}, gen={gen_len}"
    else:
        param_str = f"prompt={prompt_len}, gen={gen_len}"

    if not success_runs:
        print(f"── {metric_prefix}组均值 {param_str} → 全部失败或 OOM ──")
        return

    ttft_values = [r["ttft_ms"] for r in success_runs if r.get("ttft_ms") is not None]
    tpot_values = [r["tpot_ms"] for r in success_runs if r.get("tpot_ms") is not None]
    mean_ttft = sum(ttft_values) / len(ttft_values) if ttft_values else None
    mean_tpot = sum(tpot_values) / len(tpot_values) if tpot_values else None
    ttft_str = f"{mean_ttft:.1f}ms" if mean_ttft is not None else "N/A"
    tpot_str = f"{mean_tpot:.1f}ms" if mean_tpot is not None else "N/A"

    print(f"── {metric_prefix}组均值 {param_str} → TTFT={ttft_str}, TPOT={tpot_str} ──")


def compute_group_mean_row(
    runs: list[dict],
    group_id: str,
    run_meta: dict,
    prompt_len: int | str,
    gen_len: int,
    image_resolution: int | str,
    image_count: int | str,
    numeric_keys: list[str],
) -> dict | None:
    """计算一组重复实验的均值，返回 run_index=0 的汇总行。

    Args:
        numeric_keys: 需要求均值的数值字段（按类别传入 NUMERIC_KEYS_MEMORY
            或 NUMERIC_KEYS_EFFICIENCY，或两者合并）
    """
    success_runs = [r for r in runs if r.get("status") == "success"]
    if not success_runs:
        return None

    first = success_runs[0]
    mean_row = {
        "timestamp":         datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "run_id":            run_meta.get("run_id", generate_run_id()),
        "group_id":          group_id,
        "prompt_len":        prompt_len,
        "gen_len":           gen_len,
        "image_resolution":  image_resolution,
        "image_count":       image_count,
        "run_index":         0,
        "model_load_mem_mb": first.get("model_load_mem_mb"),
        "finish_reason":     first.get("finish_reason", ""),
        "output_text":       "",   # 均值行不保存文本
        "output_nonempty":   first.get("output_nonempty", False),
        "refusal_detected":  first.get("refusal_detected", False),
        "status":            "success",
        "oom_stage":         "none",
        "message_zh":        f"均值 ({len(success_runs)} 次成功)",
    }
    mean_row.update(run_meta)

    for key in numeric_keys:
        values = [r.get(key) for r in success_runs if r.get(key) is not None]
        if values:
            mean_val = sum(values) / len(values)
            if key in INT_KEYS:
                mean_row[key] = int(round(mean_val))
            else:
                mean_row[key] = round(mean_val, 3)
        else:
            mean_row[key] = "N/A"   # 与 raw 行一致：缺失字段统一显示 N/A

    return mean_row


def na_fields(fields: list[str]) -> dict[str, str]:
    """返回指定字段的 N/A 填充字典。"""
    return {f: "N/A" for f in fields}


VISION_ONLY_FIELDS = [
    "image_preprocess_ms", "vision_encode_ms", "projector_ms", "text_prefill_ms",
    "image_token_count", "total_input_tokens",
]
