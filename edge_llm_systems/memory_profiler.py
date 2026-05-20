"""Memory Footprint 测量（独立可用）

测量并写入"资源开销"类指标到 `results/.../memory/` 目录：
  - 模型权重显存
  - 单次推理峰值显存
  - KV cache 大小（实测 + 理论估算）
  - KV payload 占比

底层共用 profiling_core 的推理实现；本模块只负责字段过滤与 CSV 写入。
"""

from __future__ import annotations

__version__ = "1.0.0"

from pathlib import Path
from typing import Any

from edge_llm_systems._aggregation import (
    NUMERIC_KEYS_MEMORY,
    VISION_ONLY_FIELDS,
    build_row,
    compute_group_mean_row,
    na_fields,
    print_group_summary,
    print_progress,
)
from edge_llm_systems.categories import CATEGORY_MEMORY, memory_fieldnames
from edge_llm_systems.profiling_core import (
    measure_image_single as _core_image,
    measure_text_single as _core_text,
)
from edge_llm_systems.utils import append_row_to_csv, log

# ──────────────────────────────────────────────────────────────────────────────
# 单次测量接口（不写 CSV）
# ──────────────────────────────────────────────────────────────────────────────

def measure_memory_text(
    model: Any,
    tokenizer: Any,
    device: str,
    prompt_len: int,
    gen_len: int,
    model_load_mem_mb: float,
) -> dict:
    """单次文本推理，返回 memory 字段子集（含状态字段）。

    注意：返回字典只含 memory + status 字段，不含 meta（modality / run_id 等），
    上层 suite runner 通过 build_row 补全。
    """
    full = _core_text(
        model=model, tokenizer=tokenizer, device=device,
        prompt_len=prompt_len, gen_len=gen_len,
        model_load_mem_mb=model_load_mem_mb,
    )
    return full   # 完整 dict；按字段过滤在写 CSV 时进行


def measure_memory_image(
    model: Any,
    processor: Any,
    device: str,
    image: Any,
    image_resolution: int,
    gen_len: int,
    model_load_mem_mb: float,
    image_count: int = 1,
) -> dict:
    """单次视觉推理，返回完整 telemetry（含 memory 字段）。"""
    return _core_image(
        model=model, processor=processor, device=device,
        image=image, image_resolution=image_resolution,
        gen_len=gen_len, model_load_mem_mb=model_load_mem_mb,
        image_count=image_count,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Suite 接口（带 CSV 写入）
# ──────────────────────────────────────────────────────────────────────────────

def run_memory_suite_text(
    model: Any,
    tokenizer: Any,
    device: str,
    prompt_lengths: list[int],
    gen_lengths: list[int],
    repeat: int,
    model_load_mem_mb: float,
    results_dir: Path,
    ts: str,
    run_meta: dict,
) -> None:
    """文本模式 Memory Footprint 完整测试套件。

    遍历 prompt_lengths × gen_lengths 矩阵，每组重复 repeat 次，
    raw + summary 两个 CSV 实时写入 `results_dir/memory/`。
    """
    mem_dir = Path(results_dir) / CATEGORY_MEMORY
    mem_dir.mkdir(parents=True, exist_ok=True)
    raw_csv     = mem_dir / f"memory_raw_{ts}.csv"
    summary_csv = mem_dir / f"memory_summary_{ts}.csv"
    fieldnames  = memory_fieldnames()

    total_groups = len(prompt_lengths) * len(gen_lengths)
    group_count  = 0

    for prompt_len in prompt_lengths:
        for gen_len in gen_lengths:
            group_count += 1
            group_id = f"prompt{prompt_len}_gen{gen_len}"
            log(f"\n[Memory] 组 {group_count}/{total_groups}: {group_id}")

            group_runs: list[dict] = []
            for run_idx in range(1, repeat + 1):
                result = measure_memory_text(
                    model=model, tokenizer=tokenizer, device=device,
                    prompt_len=prompt_len, gen_len=gen_len,
                    model_load_mem_mb=model_load_mem_mb,
                )
                print_progress(run_idx, repeat, prompt_len, gen_len, result,
                               metric_prefix="[mem]")

                row = build_row(
                    result=result, run_meta=run_meta, group_id=group_id,
                    prompt_len=prompt_len, gen_len=gen_len,
                    image_resolution="N/A", image_count="N/A",
                    run_index=run_idx,
                )
                # 写入前过滤到 memory 字段
                row_filtered = {k: row.get(k, "N/A") for k in fieldnames}
                append_row_to_csv(raw_csv, row_filtered, fieldnames)
                group_runs.append(result)

            print_group_summary(prompt_len, gen_len, group_runs, metric_prefix="[mem]")
            mean_row = compute_group_mean_row(
                runs=group_runs, group_id=group_id, run_meta=run_meta,
                prompt_len=prompt_len, gen_len=gen_len,
                image_resolution="N/A", image_count="N/A",
                numeric_keys=NUMERIC_KEYS_MEMORY,
            )
            if mean_row is not None:
                mean_filtered = {k: mean_row.get(k, "N/A") for k in fieldnames}
                append_row_to_csv(summary_csv, mean_filtered, fieldnames)

    log(f"\n[Memory] 完成，写入:\n  raw:     {raw_csv}\n  summary: {summary_csv}")


def run_memory_suite_vision(
    model: Any,
    processor: Any,
    device: str,
    images_by_scenario: dict,    # {(image_count, resolution): PIL.Image}
    gen_lengths: list[int],
    repeat: int,
    model_load_mem_mb: float,
    results_dir: Path,
    ts: str,
    run_meta: dict,
) -> None:
    """视觉模式 Memory Footprint 完整测试套件。

    Args:
        images_by_scenario: {(image_count, resolution): PIL.Image} 字典；
            键是 (图片数, 分辨率) 二元组，值是已 resize 的代表性图像。
    """
    mem_dir = Path(results_dir) / CATEGORY_MEMORY
    mem_dir.mkdir(parents=True, exist_ok=True)
    raw_csv     = mem_dir / f"memory_raw_{ts}.csv"
    summary_csv = mem_dir / f"memory_summary_{ts}.csv"
    fieldnames  = memory_fieldnames()

    total_groups = len(images_by_scenario) * len(gen_lengths)
    group_count  = 0

    for (image_count, resolution), image in images_by_scenario.items():
        for gen_len in gen_lengths:
            group_count += 1
            group_id = f"vision_{image_count}img_{resolution}_gen{gen_len}"
            log(f"\n[Memory] 组 {group_count}/{total_groups}: {group_id}")

            group_runs: list[dict] = []
            for run_idx in range(1, repeat + 1):
                result = measure_memory_image(
                    model=model, processor=processor, device=device,
                    image=image, image_resolution=resolution,
                    gen_len=gen_len, model_load_mem_mb=model_load_mem_mb,
                    image_count=image_count,
                )
                print_progress(run_idx, repeat, "N/A", gen_len, result,
                               image_resolution=resolution, image_count=image_count,
                               metric_prefix="[mem]")

                row = build_row(
                    result=result, run_meta=run_meta, group_id=group_id,
                    prompt_len="N/A", gen_len=gen_len,
                    image_resolution=resolution, image_count=image_count,
                    run_index=run_idx,
                )
                row_filtered = {k: row.get(k, "N/A") for k in fieldnames}
                append_row_to_csv(raw_csv, row_filtered, fieldnames)
                group_runs.append(result)

            print_group_summary("N/A", gen_len, group_runs,
                                image_resolution=resolution, image_count=image_count,
                                metric_prefix="[mem]")
            mean_row = compute_group_mean_row(
                runs=group_runs, group_id=group_id, run_meta=run_meta,
                prompt_len="N/A", gen_len=gen_len,
                image_resolution=resolution, image_count=image_count,
                numeric_keys=NUMERIC_KEYS_MEMORY,
            )
            if mean_row is not None:
                mean_filtered = {k: mean_row.get(k, "N/A") for k in fieldnames}
                append_row_to_csv(summary_csv, mean_filtered, fieldnames)

    log(f"\n[Memory] 完成，写入:\n  raw:     {raw_csv}\n  summary: {summary_csv}")
