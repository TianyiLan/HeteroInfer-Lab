"""质量评估套件：基于 lm-evaluation-harness 的标准基准测试封装

使用 EleutherAI lm-evaluation-harness 运行 5 个文本基准，确保评测协议与
学术社区（Open LLM Leaderboard、已发表论文）一致，结果可直接引用。

支持的基准：
  - mmlu_pro      : 大学级多选知识题，5-shot CoT（TIGER-AI-Lab 协议）
  - gsm8k         : 数学应用题，8-shot CoT
  - hellaswag     : 常识句子补全，10-shot
  - winogrande    : 代词消歧，5-shot
  - truthfulqa_mc1: 事实性多选题，0-shot

设计原则：
  - 复用已加载的 HuggingFace model/tokenizer，避免重复加载（节省显存和时间）
  - 原始 JSON 写入 temp/，提取后的干净结果写入 qual/ CSV
  - Manifest 机制：已完成的 benchmark 自动跳过
  - run_id 贯穿所有输出，与 profiling 结果对应
"""

from __future__ import annotations

__version__ = "1.0.0"

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from edge_llm_systems.utils import append_row_to_csv, log

# ──────────────────────────────────────────────────────────────────────────────
# 基准配置（lm-eval 任务名 + 标准 few-shot 数）
# ──────────────────────────────────────────────────────────────────────────────

BENCHMARK_CONFIGS: dict[str, dict] = {
    "mmlu_pro": {
        "task":            "mmlu_pro",
        "benchmark_type":  "text",
        "num_fewshot":     5,
        "limit":           36,              # 每子任务 36 条 × 14 学科 ≈ 504 总样本
        "description":     "大学级多选知识题 (A–J)，5-shot CoT",
        "metric_key":      "exact_match,custom-extract",  # lm-eval 生成式评测，CoT 后提取字母
    },
    "gsm8k": {
        "task":            "gsm8k",
        "benchmark_type":  "text",
        "num_fewshot":     8,
        "limit":           500,
        "description":     "数学应用题，8-shot CoT",
        "metric_key":      "exact_match,strict-match",
    },
    "hellaswag": {
        "task":            "hellaswag",
        "benchmark_type":  "text",
        "num_fewshot":     10,
        "limit":           500,
        "description":     "常识句子补全，10-shot",
        "metric_key":      "acc_norm,none",
    },
    "winogrande": {
        "task":            "winogrande",
        "benchmark_type":  "text",
        "num_fewshot":     5,
        "limit":           500,
        "description":     "代词消歧二选一，5-shot",
        "metric_key":      "acc,none",
    },
    "truthfulqa_mc1": {
        "task":            "truthfulqa_mc1",
        "benchmark_type":  "text",
        "num_fewshot":     0,
        "limit":           817,             # 全集 817 题
        "description":     "事实性多选题 MC1，0-shot",
        "metric_key":      "acc,none",
    },
}

# ──────────────────────────────────────────────────────────────────────────────
# Vision benchmarks 接口预留（exp002 接入 lmms-eval 时实现）
# ──────────────────────────────────────────────────────────────────────────────
# VISION_BENCHMARK_CONFIGS = {
#     "mmmu":    {"task": "mmmu",    "benchmark_type": "vision", ...},
#     "chartqa": {"task": "chartqa", "benchmark_type": "vision", ...},
#     "docvqa":  {"task": "docvqa",  "benchmark_type": "vision", ...},
# }

# CSV 字段名（汇总表）
QUALITY_SUMMARY_FIELDNAMES = [
    "run_id", "model_id", "seed", "timestamp",
    "benchmark", "benchmark_type", "num_fewshot", "limit",
    "accuracy", "stderr",
    "num_samples",
]

# 向后兼容别名（旧代码可能引用）
QUAL_SUMMARY_FIELDNAMES = QUALITY_SUMMARY_FIELDNAMES

# ──────────────────────────────────────────────────────────────────────────────
# 核心函数
# ──────────────────────────────────────────────────────────────────────────────

def _get_lm_model(model: Any, tokenizer: Any, device: str, batch_size: int = 1):
    """将已加载的 HF model/tokenizer 封装为 lm-eval HFLM 对象。

    复用调用方已加载的模型，避免 lm-eval 重新加载（节省显存 + 时间）。
    """
    try:
        from lm_eval.models.huggingface import HFLM
    except ImportError as e:
        raise ImportError(
            "lm-evaluation-harness 未安装，请运行: pip install lm-eval"
        ) from e

    return HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        device=device,
        batch_size=batch_size,
    )


def run_single_benchmark(
    benchmark_name: str,
    lm_model: Any,
    run_id: str,
    model_id: str,
    results_dir: Path,
    seed: int = 42,
) -> dict | None:
    """运行单个 benchmark，保存原始 JSON 到 temp/，提取结果到 qual/。

    Args:
        benchmark_name : BENCHMARK_CONFIGS 中的键名
        lm_model       : _get_lm_model() 返回的 HFLM 对象
        run_id         : 本次实验 ID（与 profiling 对应）
        model_id       : 模型标识符（如 meta-llama/Llama-3.2-3B-Instruct）
        results_dir    : 根结果目录（hardware/model_id/）
        seed           : 随机种子，默认 42

    Returns:
        包含 accuracy 等指标的 dict，失败时返回 None
    """
    try:
        import lm_eval
    except ImportError as e:
        raise ImportError(
            "lm-evaluation-harness 未安装，请运行: pip install lm-eval"
        ) from e

    if benchmark_name not in BENCHMARK_CONFIGS:
        raise ValueError(
            f"未知 benchmark: {benchmark_name}，"
            f"可选: {list(BENCHMARK_CONFIGS.keys())}"
        )

    cfg          = BENCHMARK_CONFIGS[benchmark_name]

    # v2.2：仅支持 text 类基准（lm-eval）；vision 基准将在 exp002 通过 lmms-eval 实现
    if cfg.get("benchmark_type", "text") != "text":
        log(f"[Quality] 跳过 {benchmark_name}（benchmark_type={cfg.get('benchmark_type')} "
            f"暂未实现，预计 exp002 通过 lmms-eval 接入）")
        return None

    temp_dir     = results_dir / "temp" / f"lm_eval_{run_id}"
    quality_dir  = results_dir / "quality"
    temp_dir.mkdir(parents=True, exist_ok=True)
    quality_dir.mkdir(parents=True, exist_ok=True)

    # ── Manifest 检查：已完成则跳过 ──────────────────────────────────────────
    # manifest 放在 temp/ 根目录（跨 run_id 共享），保持 quality/ 干净
    manifest_path = results_dir / "temp" / "quality_manifest.json"
    manifest      = _load_manifest(manifest_path)
    manifest_key  = f"{run_id}_{benchmark_name}"
    if manifest_key in manifest:
        log(f"[Quality] {benchmark_name} 已完成（manifest），跳过")
        return manifest[manifest_key]

    log(f"[Quality] 开始运行 {benchmark_name} — {cfg['description']}")
    log(f"  任务: {cfg['task']}, few-shot: {cfg['num_fewshot']}, limit: {cfg['limit']}")
    t_start = time.perf_counter()

    # ── 运行 lm-eval ──────────────────────────────────────────────────────────
    raw_results = lm_eval.simple_evaluate(
        model=lm_model,
        tasks=[cfg["task"]],
        num_fewshot=cfg["num_fewshot"],
        limit=cfg["limit"],
        random_seed=seed,
        numpy_random_seed=seed,
        torch_random_seed=seed,
        log_samples=True,               # 保存逐题结果到 temp/
        write_out=False,
    )

    elapsed_s = time.perf_counter() - t_start
    log(f"  完成，耗时 {elapsed_s/60:.1f} 分钟")

    # ── 保存原始 JSON 到 temp/ ────────────────────────────────────────────────
    raw_path = temp_dir / f"results_{benchmark_name}.json"
    with open(raw_path, "w", encoding="utf-8") as f:
        # samples 不可 JSON 序列化时做降级处理
        json.dump(raw_results, f, ensure_ascii=False, indent=2, default=str)
    log(f"  原始结果 → {raw_path}")

    # ── 提取汇总指标 ──────────────────────────────────────────────────────────
    task_results = raw_results.get("results", {}).get(cfg["task"], {})
    accuracy     = task_results.get(cfg["metric_key"], None)
    # stderr key 通用推导：在第一个逗号前插入 "_stderr"
    # 例：acc,none → acc_stderr,none；exact_match,custom-extract → exact_match_stderr,custom-extract
    _mk_parts    = cfg["metric_key"].split(",", 1)
    _stderr_key  = _mk_parts[0] + "_stderr," + _mk_parts[1] if len(_mk_parts) == 2 else cfg["metric_key"] + "_stderr"
    stderr_raw   = task_results.get(_stderr_key, None)
    num_samples  = cfg["limit"]

    if accuracy is None:
        log(f"  ⚠️  无法从结果中提取 accuracy（metric_key={cfg['metric_key']}）")
        log(f"  可用字段: {list(task_results.keys())}")
        return None

    accuracy_pct = round(accuracy * 100, 2)

    # 合法性校验：stderr 必须 < accuracy 且 < 50%；否则回退到解析估算
    import math as _math
    if (
        stderr_raw is not None
        and isinstance(stderr_raw, (int, float))
        and 0 < stderr_raw < min(accuracy, 0.5)   # stderr 必须小于 accuracy 且 < 50%
    ):
        stderr_pct = round(stderr_raw * 100, 2)
    else:
        # 回退：按二项分布公式估算 SE = sqrt(p*(1-p)/n)
        n = num_samples if num_samples and num_samples > 0 else 1
        stderr_pct = round(_math.sqrt(accuracy * (1 - accuracy) / n) * 100, 2)
        if stderr_raw is not None:
            log(f"  ⚠️  stderr 字段异常（值={stderr_raw}），已用公式估算 SE={stderr_pct}%")
    timestamp    = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    summary_row = {
        "run_id":          run_id,
        "model_id":        model_id,
        "seed":            seed,
        "timestamp":       timestamp,
        "benchmark":       benchmark_name,
        "benchmark_type":  cfg.get("benchmark_type", "text"),
        "num_fewshot":     cfg["num_fewshot"],
        "limit":           num_samples,
        "accuracy":        accuracy_pct,
        "stderr":          stderr_pct,
        "num_samples":     num_samples,
    }

    # ── 写入汇总 CSV ──────────────────────────────────────────────────────────
    summary_csv = quality_dir / f"quality_summary_{run_id}.csv"
    append_row_to_csv(summary_csv, summary_row, QUALITY_SUMMARY_FIELDNAMES)
    log(f"  汇总 → {summary_csv}")
    log(f"  结果: {benchmark_name} accuracy = {accuracy_pct}%"
        + (f" ± {stderr_pct}%" if stderr_pct else ""))

    # ── 更新 Manifest ─────────────────────────────────────────────────────────
    manifest[manifest_key] = summary_row
    _save_manifest(manifest_path, manifest)

    return summary_row


def run_quality_suite(
    benchmarks: list[str],
    model: Any,
    tokenizer: Any,
    device: str,
    run_id: str,
    model_id: str,
    results_dir: Path,
    seed: int = 42,
    batch_size: int = 1,
) -> dict[str, dict]:
    """运行多个 benchmark，返回所有结果。

    Args:
        benchmarks  : 要运行的 benchmark 列表，如 ["mmlu_pro", "gsm8k"]
        model       : 已加载的 HF AutoModelForCausalLM
        tokenizer   : 对应 tokenizer
        device      : "cuda" / "cpu"
        run_id      : 实验 ID
        model_id    : 模型标识符
        results_dir : 根结果目录（hardware/model_id/）
        seed        : 随机种子
        batch_size  : lm-eval batch size（显存紧张时设为 1）

    Returns:
        {benchmark_name: summary_row} 的字典
    """
    lm_model = _get_lm_model(model, tokenizer, device, batch_size)
    all_results: dict[str, dict] = {}

    for name in benchmarks:
        result = run_single_benchmark(
            benchmark_name=name,
            lm_model=lm_model,
            run_id=run_id,
            model_id=model_id,
            results_dir=results_dir,
            seed=seed,
        )
        if result:
            all_results[name] = result

    # ── 打印汇总表 ────────────────────────────────────────────────────────────
    log("\n" + "=" * 60)
    log(f"[Quality] 汇总 — run_id={run_id}")
    log("=" * 60)
    for name, row in all_results.items():
        stderr_str = f" ± {row['stderr']}%" if row.get("stderr") else ""
        log(f"  {name:<20} {row['accuracy']:>6.2f}%{stderr_str}")
    log("=" * 60)

    return all_results


# ──────────────────────────────────────────────────────────────────────────────
# 内部工具
# ──────────────────────────────────────────────────────────────────────────────

def _load_manifest(path: Path) -> dict:
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_manifest(path: Path, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
