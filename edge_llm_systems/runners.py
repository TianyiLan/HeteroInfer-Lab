"""实验循环调度：warm-up + 三分类性能测试编排。

v2.2 起按指标三分类拆分：
  - run_warmup_text / run_warmup_vision : 模型热身（不写 CSV）
  - run_profiling_suite_text   : 文本模式智能编排 Memory / Efficiency
  - run_profiling_suite_vision : 视觉模式智能编排

编排规则（避免重复推理）：
  - 只勾 Memory       → 调 memory_profiler.run_memory_suite_*
  - 只勾 Efficiency   → 调 efficiency_profiler.run_efficiency_suite_*
  - 都勾 + 参数相同   → 单次推理产 2 个 CSV（节省 50% 计算）
  - 都勾 + 参数不同   → 分别调两个 suite
  - 都不勾            → 跳过
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from edge_llm_systems._aggregation import (
    NUMERIC_KEYS_EFFICIENCY,
    NUMERIC_KEYS_MEMORY,
    build_row,
    compute_group_mean_row,
    print_group_summary,
    print_progress,
)
from edge_llm_systems.categories import (
    CATEGORY_EFFICIENCY,
    CATEGORY_MEMORY,
    efficiency_fieldnames,
    memory_fieldnames,
)
from edge_llm_systems.efficiency_profiler import run_efficiency_suite_text, run_efficiency_suite_vision
from edge_llm_systems.memory_profiler import run_memory_suite_text, run_memory_suite_vision
from edge_llm_systems.profiling_core import measure_image_single, measure_text_single
from edge_llm_systems.utils import append_row_to_csv, log

# ──────────────────────────────────────────────────────────────────────────────
# 热身配置
# ──────────────────────────────────────────────────────────────────────────────
WARMUP_CONFIGS_TEXT   = [(64, 8), (256, 8), (512, 8)]   # (prompt_len, gen_len)
WARMUP_CONFIGS_VISION = [(224, 8)]                       # (resolution, gen_len)


# ──────────────────────────────────────────────────────────────────────────────
# 热身函数（不写 CSV，仅用于 CUDA 内核 / cuDNN 优化路径预热）
# ──────────────────────────────────────────────────────────────────────────────

def run_warmup_text(model: Any, tokenizer: Any, device: str) -> None:
    """文本模型热身。"""
    log("[热身] 开始文本模型热身...")
    for i, (prompt_len, gen_len) in enumerate(WARMUP_CONFIGS_TEXT):
        result = measure_text_single(
            model=model, tokenizer=tokenizer, device=device,
            prompt_len=prompt_len, gen_len=gen_len,
            model_load_mem_mb=0.0,
        )
        status = result.get("status", "unknown")
        ttft   = result.get("ttft_ms")
        ttft_s = f"{ttft:.1f}ms" if ttft is not None else "N/A"
        log(f"  热身 [{i+1}/{len(WARMUP_CONFIGS_TEXT)}] prompt={prompt_len}, gen={gen_len} "
            f"→ TTFT={ttft_s}, status={status}")
    log("[热身] 文本模型热身完成")


def run_warmup_vision(model: Any, processor: Any, device: str, warmup_image: Any) -> None:
    """视觉模型热身（含文本热身）。"""
    log("[热身] 开始视觉模型热身...")
    run_warmup_text(model=model, tokenizer=processor.tokenizer, device=device)
    for i, (resolution, gen_len) in enumerate(WARMUP_CONFIGS_VISION):
        result = measure_image_single(
            model=model, processor=processor, device=device,
            image=warmup_image, image_resolution=resolution,
            gen_len=gen_len, model_load_mem_mb=0.0,
        )
        status = result.get("status", "unknown")
        ttft   = result.get("ttft_ms")
        ttft_s = f"{ttft:.1f}ms" if ttft is not None else "N/A"
        log(f"  图像热身 [{i+1}/{len(WARMUP_CONFIGS_VISION)}] resolution={resolution}, gen={gen_len} "
            f"→ TTFT={ttft_s}, status={status}")
    log("[热身] 视觉模型热身完成")


# ──────────────────────────────────────────────────────────────────────────────
# 内部：单次推理 → 同时写 memory + efficiency CSV（"都勾+参数相同"快路径）
# ──────────────────────────────────────────────────────────────────────────────

def _run_combined_text_loop(
    model: Any, tokenizer: Any, device: str,
    prompt_lengths: list[int], gen_lengths: list[int], repeat: int,
    model_load_mem_mb: float,
    results_dir: Path, ts: str, run_meta: dict,
) -> None:
    """文本模式：单次推理 → 同时写 memory/ + efficiency/ 两套 CSV。"""
    mem_dir = Path(results_dir) / CATEGORY_MEMORY
    eff_dir = Path(results_dir) / CATEGORY_EFFICIENCY
    mem_dir.mkdir(parents=True, exist_ok=True)
    eff_dir.mkdir(parents=True, exist_ok=True)

    mem_raw_csv = mem_dir / f"memory_raw_{ts}.csv"
    mem_sum_csv = mem_dir / f"memory_summary_{ts}.csv"
    eff_raw_csv = eff_dir / f"efficiency_raw_{ts}.csv"
    eff_sum_csv = eff_dir / f"efficiency_summary_{ts}.csv"

    mem_fields     = memory_fieldnames()
    eff_raw_fields = efficiency_fieldnames(include_output_text=True)
    eff_sum_fields = efficiency_fieldnames(include_output_text=False)

    total_groups = len(prompt_lengths) * len(gen_lengths)
    group_count  = 0

    for prompt_len in prompt_lengths:
        for gen_len in gen_lengths:
            group_count += 1
            group_id = f"prompt{prompt_len}_gen{gen_len}"
            log(f"\n[Mem+Eff] 组 {group_count}/{total_groups}: {group_id}")

            group_runs: list[dict] = []
            for run_idx in range(1, repeat + 1):
                result = measure_text_single(
                    model=model, tokenizer=tokenizer, device=device,
                    prompt_len=prompt_len, gen_len=gen_len,
                    model_load_mem_mb=model_load_mem_mb,
                )
                print_progress(run_idx, repeat, prompt_len, gen_len, result,
                               metric_prefix="[mem+eff]")

                row = build_row(
                    result=result, run_meta=run_meta, group_id=group_id,
                    prompt_len=prompt_len, gen_len=gen_len,
                    image_resolution="N/A", image_count="N/A",
                    run_index=run_idx,
                )
                # 同一行数据 → 两份 CSV 各过滤各自字段
                append_row_to_csv(mem_raw_csv,
                                  {k: row.get(k, "N/A") for k in mem_fields},
                                  mem_fields)
                append_row_to_csv(eff_raw_csv,
                                  {k: row.get(k, "N/A") for k in eff_raw_fields},
                                  eff_raw_fields)
                group_runs.append(result)

            print_group_summary(prompt_len, gen_len, group_runs, metric_prefix="[mem+eff]")

            mem_mean = compute_group_mean_row(
                runs=group_runs, group_id=group_id, run_meta=run_meta,
                prompt_len=prompt_len, gen_len=gen_len,
                image_resolution="N/A", image_count="N/A",
                numeric_keys=NUMERIC_KEYS_MEMORY,
            )
            eff_mean = compute_group_mean_row(
                runs=group_runs, group_id=group_id, run_meta=run_meta,
                prompt_len=prompt_len, gen_len=gen_len,
                image_resolution="N/A", image_count="N/A",
                numeric_keys=NUMERIC_KEYS_EFFICIENCY,
            )
            if mem_mean is not None:
                append_row_to_csv(mem_sum_csv,
                                  {k: mem_mean.get(k, "N/A") for k in mem_fields},
                                  mem_fields)
            if eff_mean is not None:
                append_row_to_csv(eff_sum_csv,
                                  {k: eff_mean.get(k, "N/A") for k in eff_sum_fields},
                                  eff_sum_fields)

    log(f"\n[Mem+Eff] 完成，写入:")
    log(f"  memory  : {mem_raw_csv} / {mem_sum_csv}")
    log(f"  efficiency: {eff_raw_csv} / {eff_sum_csv}")


def _run_combined_vision_loop(
    model: Any, processor: Any, device: str,
    images_by_scenario: dict,
    gen_lengths: list[int], repeat: int,
    model_load_mem_mb: float,
    results_dir: Path, ts: str, run_meta: dict,
) -> None:
    """视觉模式：单次推理 → 同时写 memory/ + efficiency/。"""
    mem_dir = Path(results_dir) / CATEGORY_MEMORY
    eff_dir = Path(results_dir) / CATEGORY_EFFICIENCY
    mem_dir.mkdir(parents=True, exist_ok=True)
    eff_dir.mkdir(parents=True, exist_ok=True)

    mem_raw_csv = mem_dir / f"memory_raw_{ts}.csv"
    mem_sum_csv = mem_dir / f"memory_summary_{ts}.csv"
    eff_raw_csv = eff_dir / f"efficiency_raw_{ts}.csv"
    eff_sum_csv = eff_dir / f"efficiency_summary_{ts}.csv"

    mem_fields     = memory_fieldnames()
    eff_raw_fields = efficiency_fieldnames(include_output_text=True)
    eff_sum_fields = efficiency_fieldnames(include_output_text=False)

    total_groups = len(images_by_scenario) * len(gen_lengths)
    group_count  = 0

    for (image_count, resolution), image in images_by_scenario.items():
        for gen_len in gen_lengths:
            group_count += 1
            group_id = f"vision_{image_count}img_{resolution}_gen{gen_len}"
            log(f"\n[Mem+Eff] 组 {group_count}/{total_groups}: {group_id}")

            group_runs: list[dict] = []
            for run_idx in range(1, repeat + 1):
                result = measure_image_single(
                    model=model, processor=processor, device=device,
                    image=image, image_resolution=resolution,
                    gen_len=gen_len, model_load_mem_mb=model_load_mem_mb,
                    image_count=image_count,
                )
                print_progress(run_idx, repeat, "N/A", gen_len, result,
                               image_resolution=resolution, image_count=image_count,
                               metric_prefix="[mem+eff]")

                row = build_row(
                    result=result, run_meta=run_meta, group_id=group_id,
                    prompt_len="N/A", gen_len=gen_len,
                    image_resolution=resolution, image_count=image_count,
                    run_index=run_idx,
                )
                append_row_to_csv(mem_raw_csv,
                                  {k: row.get(k, "N/A") for k in mem_fields},
                                  mem_fields)
                append_row_to_csv(eff_raw_csv,
                                  {k: row.get(k, "N/A") for k in eff_raw_fields},
                                  eff_raw_fields)
                group_runs.append(result)

            print_group_summary("N/A", gen_len, group_runs,
                                image_resolution=resolution, image_count=image_count,
                                metric_prefix="[mem+eff]")

            mem_mean = compute_group_mean_row(
                runs=group_runs, group_id=group_id, run_meta=run_meta,
                prompt_len="N/A", gen_len=gen_len,
                image_resolution=resolution, image_count=image_count,
                numeric_keys=NUMERIC_KEYS_MEMORY,
            )
            eff_mean = compute_group_mean_row(
                runs=group_runs, group_id=group_id, run_meta=run_meta,
                prompt_len="N/A", gen_len=gen_len,
                image_resolution=resolution, image_count=image_count,
                numeric_keys=NUMERIC_KEYS_EFFICIENCY,
            )
            if mem_mean is not None:
                append_row_to_csv(mem_sum_csv,
                                  {k: mem_mean.get(k, "N/A") for k in mem_fields},
                                  mem_fields)
            if eff_mean is not None:
                append_row_to_csv(eff_sum_csv,
                                  {k: eff_mean.get(k, "N/A") for k in eff_sum_fields},
                                  eff_sum_fields)

    log(f"\n[Mem+Eff] 完成，写入:")
    log(f"  memory  : {mem_raw_csv} / {mem_sum_csv}")
    log(f"  efficiency: {eff_raw_csv} / {eff_sum_csv}")


# ──────────────────────────────────────────────────────────────────────────────
# 公开编排接口
# ──────────────────────────────────────────────────────────────────────────────

def run_profiling_suite_text(
    model: Any,
    tokenizer: Any,
    device: str,
    *,
    enable_memory: bool,
    enable_efficiency: bool,
    memory_prompt_lengths:     list[int] | None = None,
    memory_gen_lengths:        list[int] | None = None,
    memory_repeat:             int = 3,
    efficiency_prompt_lengths: list[int] | None = None,
    efficiency_gen_lengths:    list[int] | None = None,
    efficiency_repeat:         int = 3,
    model_load_mem_mb: float = 0.0,
    results_dir:       Path | str = "results",
    ts:                str = "",
    run_meta:          dict | None = None,
) -> None:
    """文本模式性能测试主编排函数。

    根据 enable_memory / enable_efficiency 开关分流：
      - 都关 → 跳过
      - 都开 + 参数相同 → 单次推理写两份 CSV
      - 都开 + 参数不同 → 分别跑两个独立 suite
      - 单开 → 跑对应 suite
    """
    if not (enable_memory or enable_efficiency):
        log("[Profiling] Memory 和 Efficiency 都未启用，跳过性能测试")
        return

    run_meta    = run_meta or {}
    results_dir = Path(results_dir)

    same_params = (
        enable_memory and enable_efficiency
        and memory_prompt_lengths == efficiency_prompt_lengths
        and memory_gen_lengths    == efficiency_gen_lengths
        and memory_repeat         == efficiency_repeat
    )

    if same_params:
        log("[Profiling] Memory + Efficiency 参数相同，启用单次推理优化")
        _run_combined_text_loop(
            model=model, tokenizer=tokenizer, device=device,
            prompt_lengths=memory_prompt_lengths,
            gen_lengths=memory_gen_lengths,
            repeat=memory_repeat,
            model_load_mem_mb=model_load_mem_mb,
            results_dir=results_dir, ts=ts, run_meta=run_meta,
        )
        return

    if enable_memory:
        log("[Profiling] 单独执行 Memory suite")
        run_memory_suite_text(
            model=model, tokenizer=tokenizer, device=device,
            prompt_lengths=memory_prompt_lengths,
            gen_lengths=memory_gen_lengths,
            repeat=memory_repeat,
            model_load_mem_mb=model_load_mem_mb,
            results_dir=results_dir, ts=ts, run_meta=run_meta,
        )

    if enable_efficiency:
        log("[Profiling] 单独执行 Efficiency suite")
        run_efficiency_suite_text(
            model=model, tokenizer=tokenizer, device=device,
            prompt_lengths=efficiency_prompt_lengths,
            gen_lengths=efficiency_gen_lengths,
            repeat=efficiency_repeat,
            model_load_mem_mb=model_load_mem_mb,
            results_dir=results_dir, ts=ts, run_meta=run_meta,
        )


def run_profiling_suite_vision(
    model: Any,
    processor: Any,
    device: str,
    *,
    enable_memory: bool,
    enable_efficiency: bool,
    images_by_scenario: dict,    # {(image_count, resolution): PIL.Image}
    gen_lengths: list[int],
    memory_repeat:     int = 3,
    efficiency_repeat: int = 3,
    model_load_mem_mb: float = 0.0,
    results_dir: Path | str = "results",
    ts: str = "",
    run_meta: dict | None = None,
) -> None:
    """视觉模式性能测试主编排函数。"""
    if not (enable_memory or enable_efficiency):
        log("[Profiling] Memory 和 Efficiency 都未启用，跳过性能测试")
        return

    run_meta    = run_meta or {}
    results_dir = Path(results_dir)

    # vision 模式下，scenarios（image_count × resolution）和 gen_lengths 是共享的，
    # 参数维度只剩 repeat。当两 repeat 相同时走快路径
    same_params = (enable_memory and enable_efficiency
                   and memory_repeat == efficiency_repeat)

    if same_params:
        log("[Profiling] Memory + Efficiency 参数相同，启用单次推理优化")
        _run_combined_vision_loop(
            model=model, processor=processor, device=device,
            images_by_scenario=images_by_scenario,
            gen_lengths=gen_lengths, repeat=memory_repeat,
            model_load_mem_mb=model_load_mem_mb,
            results_dir=results_dir, ts=ts, run_meta=run_meta,
        )
        return

    if enable_memory:
        log("[Profiling] 单独执行 Memory suite")
        run_memory_suite_vision(
            model=model, processor=processor, device=device,
            images_by_scenario=images_by_scenario,
            gen_lengths=gen_lengths, repeat=memory_repeat,
            model_load_mem_mb=model_load_mem_mb,
            results_dir=results_dir, ts=ts, run_meta=run_meta,
        )

    if enable_efficiency:
        log("[Profiling] 单独执行 Efficiency suite")
        run_efficiency_suite_vision(
            model=model, processor=processor, device=device,
            images_by_scenario=images_by_scenario,
            gen_lengths=gen_lengths, repeat=efficiency_repeat,
            model_load_mem_mb=model_load_mem_mb,
            results_dir=results_dir, ts=ts, run_meta=run_meta,
        )
