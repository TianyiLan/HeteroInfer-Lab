"""质量评估套件：文本基准测试（MMLU-Pro / GSM8K / HellaSwag / WinoGrande / TruthfulQA）

Stage 1 FP16 质量基线 → Stage 3/4/5 优化实验对比基准。

支持 5 个文本基准（exp001）：
  - mmlu_pro_mini   (70 样本)：大学级多选知识题（A–J 十选一）
  - gsm8k_mini      (50 样本)：数学应用题，提取最终数字
  - hellaswag_mini  (50 样本)：常识句子补全（A–D 四选一）
  - winogrande_mini (50 样本)：代词消歧二选一（A/B）
  - truthfulqa_mc   (50 样本)：事实性多选题（MC1，A–X 多选一）

预留视觉基准接口（exp002 实现）：
  - VQAv2 / MMBench / MathVista / TextVQA / DocVQA

设计原则：
  - seed=42 固定取前 N 条，manifest.json 锁定数据集版本，保证跨运行可复现
  - per-sample 结果实时 append 写入 quality_raw/ CSV（fail-safe）
  - 汇总分数写入 quality_summary/ JSON（便于跨模型对比）
  - 使用 apply_chat_template() 保证 Instruct 模型标准输入格式
  - 贪婪解码（do_sample=False），消除随机性
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

import torch

from edge_llm_systems.utils import append_row_to_csv, save_json

# ──────────────────────────────────────────────────────────────────────────────
# 基准配置表
# ──────────────────────────────────────────────────────────────────────────────

BENCHMARK_CONFIGS: dict[str, dict] = {
    "mmlu_pro_mini": {
        "hf_id":         "TIGER-Lab/MMLU-Pro",
        "hf_split":      "test",
        "max_samples":   70,
        "max_new_tokens": 10,
        "description":   "大学级多选知识题 (A–J)",
    },
    "gsm8k_mini": {
        "hf_id":         "openai/gsm8k",
        "hf_name":       "main",
        "hf_split":      "test",
        "max_samples":   50,
        "max_new_tokens": 256,
        "description":   "数学应用题（提取最终数字）",
    },
    "hellaswag_mini": {
        "hf_id":         "Rowan/hellaswag",
        "hf_split":      "validation",
        "max_samples":   50,
        "max_new_tokens": 10,
        "description":   "常识句子补全 (A–D)",
    },
    "winogrande_mini": {
        "hf_id":         "allenai/winogrande",
        "hf_name":       "winogrande_xl",
        "hf_split":      "validation",
        "max_samples":   50,
        "max_new_tokens": 10,
        "description":   "代词消歧二选一 (A/B)",
    },
    "truthfulqa_mc": {
        "hf_id":         "truthful_qa",
        "hf_name":       "multiple_choice",
        "hf_split":      "validation",
        "max_samples":   50,
        "max_new_tokens": 10,
        "description":   "事实性多选题 MC1",
    },
}

# per-sample CSV 字段顺序
QUALITY_RAW_FIELDNAMES: list[str] = [
    "run_id", "model_id", "benchmark", "seed", "sample_id",
    "question_truncated", "correct_answer",
    "model_output_truncated", "parsed_answer", "is_correct",
    "generation_time_ms", "input_tokens", "output_tokens",
]

# ──────────────────────────────────────────────────────────────────────────────
# 视觉基准配置（预留，exp002 实现）
# ──────────────────────────────────────────────────────────────────────────────

VISION_BENCHMARK_CONFIGS: dict[str, dict] = {
    "vqav2_mini":     {"hf_id": "HuggingFaceM4/VQAv2",           "hf_split": "validation", "max_samples": 50},
    "mmbench_mini":   {"hf_id": "HuggingFaceM4/MMBench",         "hf_split": "dev",        "max_samples": 50},
    "mathvista_mini": {"hf_id": "AI4Math/MathVista",             "hf_split": "testmini",   "max_samples": 30},
    "textvqa_mini":   {"hf_id": "textvqa",                       "hf_split": "validation", "max_samples": 30},
    "docvqa_mini":    {"hf_id": "nielsr/docvqa_1200_examples",   "hf_split": "test",       "max_samples": 30},
}


# ──────────────────────────────────────────────────────────────────────────────
# 数据集加载（带本地缓存 + manifest 版本锁定）
# ──────────────────────────────────────────────────────────────────────────────

def _load_dataset_cached(
    benchmark_name: str,
    cfg: dict,
    dataset_dir: str | Path,
    seed: int = 42,
) -> list[dict]:
    """从本地缓存加载数据集，不存在则从 HuggingFace 下载并缓存。

    缓存目录结构：
        {dataset_dir}/quality_suite/text_only/{benchmark_name}/
        ├── data.json       — 固定样本列表
        └── manifest.json   — 版本元数据（sample_count, seed, hf_id）

    Args:
        benchmark_name: 基准名称
        cfg: BENCHMARK_CONFIGS 中对应的配置字典
        dataset_dir: 数据集根目录（Google Drive 上的持久路径）
        seed: 随机种子（当前实现：固定取前 N 条，seed 用于 manifest 校验）

    Returns:
        样本字典列表
    """
    cache_dir = Path(dataset_dir) / "quality_suite" / "text_only" / benchmark_name
    manifest_path = cache_dir / "manifest.json"
    data_path     = cache_dir / "data.json"

    # 缓存命中：样本数和 seed 均一致时直接返回
    if manifest_path.exists() and data_path.exists():
        with open(manifest_path, encoding="utf-8") as f:
            mf = json.load(f)
        if (mf.get("sample_count") == cfg["max_samples"]
                and mf.get("seed") == seed):
            with open(data_path, encoding="utf-8") as f:
                return json.load(f)

    # 缓存缺失或版本不匹配 → 下载
    from datasets import load_dataset
    print(f"[Quality] Downloading {benchmark_name} from {cfg['hf_id']}...")
    load_kwargs: dict = {"split": cfg["hf_split"], "trust_remote_code": True}
    if "hf_name" in cfg:
        load_kwargs["name"] = cfg["hf_name"]
    ds = load_dataset(cfg["hf_id"], **load_kwargs)

    # 固定取前 N 条（不做 shuffle，seed 记录在 manifest 供复现说明）
    n = min(cfg["max_samples"], len(ds))
    samples = [dict(ds[i]) for i in range(n)]

    # 写缓存
    cache_dir.mkdir(parents=True, exist_ok=True)
    with open(data_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)

    manifest = {
        "benchmark":    benchmark_name,
        "hf_id":        cfg["hf_id"],
        "hf_split":     cfg["hf_split"],
        "seed":         seed,
        "sample_count": n,
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"[Quality] {benchmark_name}: {n} samples cached → {cache_dir}")
    return samples


# ──────────────────────────────────────────────────────────────────────────────
# 推理工具：chat template + 贪婪解码
# ──────────────────────────────────────────────────────────────────────────────

def _generate(
    model: Any,
    tokenizer: Any,
    device: str,
    user_prompt: str,
    max_new_tokens: int,
) -> tuple[str, float, int, int]:
    """使用 chat template 格式化 prompt，贪婪解码生成答案。

    Returns:
        (output_text, generation_time_ms, input_token_count, output_token_count)
    """
    # apply_chat_template 保证 Instruct 模型接收正确的对话格式
    messages = [{"role": "user", "content": user_prompt}]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(formatted, return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[-1]

    # CUDA 同步确保计时准确
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,                        # 贪婪解码，确定性输出
            pad_token_id=tokenizer.eos_token_id,
        )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    gen_ms = (time.perf_counter() - t0) * 1000.0

    new_ids     = out[0, input_len:]
    output_text = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
    return output_text, round(gen_ms, 1), input_len, len(new_ids)


# ──────────────────────────────────────────────────────────────────────────────
# 答案解析工具
# ──────────────────────────────────────────────────────────────────────────────

def _extract_mc_letter(text: str, valid: str = "ABCD") -> str:
    """从模型输出中提取多选题字母答案。

    优先匹配 "Answer: X" / "The answer is X" 模式，
    回退到输出中第一个独立的有效字母。
    """
    text = text.strip().upper()
    valid_pat = "[" + re.escape(valid) + "]"

    for pat in [
        rf"(?:ANSWER|THE ANSWER IS)[:\s]+({valid_pat})\b",
        rf"\b({valid_pat})\b",
    ]:
        m = re.search(pat, text)
        if m:
            return m.group(1)
    return ""


def _extract_num(text: str) -> str:
    """从文本中提取最后一个数字（用于 GSM8K）。

    去除千位逗号，返回字符串形式（便于精确匹配）。
    """
    nums = re.findall(r"-?\d+(?:,\d+)*(?:\.\d+)?", text)
    return nums[-1].replace(",", "") if nums else ""


# ──────────────────────────────────────────────────────────────────────────────
# 各基准 Prompt 格式化函数
# ──────────────────────────────────────────────────────────────────────────────

def _format_mmlu_pro(sample: dict) -> tuple[str, str]:
    """MMLU-Pro：最多 10 个选项（A–J）。"""
    opts = sample.get("options", [])
    opt_str = "\n".join(f"{chr(65 + i)}. {o}" for i, o in enumerate(opts))
    prompt = (
        f"Question: {sample.get('question', '')}\n\n"
        f"Options:\n{opt_str}\n\n"
        f"Answer with a single letter ({chr(65)}–{chr(64 + len(opts))}):"
    )
    correct = str(sample.get("answer", "")).strip().upper()
    return prompt, correct


def _format_gsm8k(sample: dict) -> tuple[str, str]:
    """GSM8K：链式推理，答案在 '####' 之后。"""
    prompt = (
        f"Solve step by step. End your answer with '#### <number>'.\n\n"
        f"Problem: {sample.get('question', '')}"
    )
    answer_text = sample.get("answer", "")
    if "####" in answer_text:
        correct = answer_text.split("####")[-1].strip().replace(",", "")
    else:
        correct = _extract_num(answer_text)
    return prompt, correct


def _format_hellaswag(sample: dict) -> tuple[str, str]:
    """HellaSwag：从 4 个结尾中选最合适的一个（A–D）。"""
    endings = sample.get("endings", [])
    ctx     = sample.get("ctx", "")
    opt_str = "\n".join(f"{chr(65 + i)}. {e}" for i, e in enumerate(endings))
    prompt = (
        f"Choose the best ending for the following text.\n\n"
        f"Text: {ctx}\n\n"
        f"{opt_str}\n\n"
        f"Answer with a single letter (A–D):"
    )
    label   = int(sample.get("label", 0))
    correct = chr(65 + label)
    return prompt, correct


def _format_winogrande(sample: dict) -> tuple[str, str]:
    """WinoGrande：填空，二选一（A/B）。"""
    prompt = (
        f"Fill in the blank with the correct option.\n\n"
        f"Sentence: {sample.get('sentence', '')}\n\n"
        f"A. {sample.get('option1', '')}\n"
        f"B. {sample.get('option2', '')}\n\n"
        f"Answer with A or B:"
    )
    correct = "A" if str(sample.get("answer", "1")) == "1" else "B"
    return prompt, correct


def _format_truthfulqa(sample: dict) -> tuple[str | None, str | None]:
    """TruthfulQA MC1：从所有选项中选出唯一正确答案。

    Returns:
        (prompt, correct_letter)，若样本无选项则返回 (None, None)
    """
    mc1     = sample.get("mc1_targets", {})
    choices = mc1.get("choices", [])
    labels  = mc1.get("labels",  [])
    if not choices:
        return None, None

    correct_idx = labels.index(1) if 1 in labels else 0
    correct     = chr(65 + correct_idx)
    opt_str     = "\n".join(f"{chr(65 + i)}. {c}" for i, c in enumerate(choices))
    prompt = (
        f"Question: {sample.get('question', '')}\n\n"
        f"{opt_str}\n\n"
        f"Answer with a single letter:"
    )
    return prompt, correct


# ──────────────────────────────────────────────────────────────────────────────
# 单基准运行器
# ──────────────────────────────────────────────────────────────────────────────

def run_single_benchmark(
    benchmark_name: str,
    model: Any,
    tokenizer: Any,
    device: str,
    dataset_dir: str | Path,
    raw_csv_path: str | Path,
    run_id: str,
    model_id: str,
    seed: int = 42,
) -> dict:
    """运行单个文本基准，逐样本写 CSV，返回汇总 dict。

    Args:
        benchmark_name: BENCHMARK_CONFIGS 中的键
        model: 已加载的语言模型
        tokenizer: 对应分词器
        device: 推理设备
        dataset_dir: 数据集根目录
        raw_csv_path: per-sample CSV 输出路径
        run_id: 关联的 run_id（与性能测试共享）
        model_id: HuggingFace repo id
        seed: 数据集版本 seed

    Returns:
        {benchmark, accuracy, answer_rate, num_correct, num_samples, seed}
    """
    if benchmark_name not in BENCHMARK_CONFIGS:
        raise ValueError(
            f"Unknown benchmark: {benchmark_name!r}. "
            f"Available: {list(BENCHMARK_CONFIGS.keys())}"
        )

    cfg     = BENCHMARK_CONFIGS[benchmark_name]
    samples = _load_dataset_cached(benchmark_name, cfg, dataset_dir, seed)
    total   = len(samples)

    correct  = 0
    answered = 0
    skipped  = 0

    print(f"\n[Quality] {benchmark_name} ({cfg['description']}) — {total} samples")

    # 格式化函数分发表（避免大量 if/elif）
    _formatters = {
        "mmlu_pro_mini":   _format_mmlu_pro,
        "gsm8k_mini":      _format_gsm8k,
        "hellaswag_mini":  _format_hellaswag,
        "winogrande_mini": _format_winogrande,
        "truthfulqa_mc":   _format_truthfulqa,
    }
    formatter = _formatters[benchmark_name]

    for i, sample in enumerate(samples):
        prompt, gold = formatter(sample)

        # TruthfulQA 样本可能无选项，跳过
        if prompt is None:
            skipped += 1
            continue

        # 生成答案
        output, gen_ms, in_tok, out_tok = _generate(
            model, tokenizer, device, prompt, cfg["max_new_tokens"]
        )

        # 解析答案
        if benchmark_name == "gsm8k_mini":
            # GSM8K：提取输出中 "####" 之后的数字，否则取最后一个数字
            if "####" in output:
                parsed = output.split("####")[-1].strip().replace(",", "")
                parsed = _extract_num(parsed) or parsed
            else:
                parsed = _extract_num(output)
            is_correct = bool(parsed) and (parsed == gold)
        elif benchmark_name == "mmlu_pro_mini":
            valid  = "ABCDEFGHIJ"
            parsed = _extract_mc_letter(output, valid)
            is_correct = bool(parsed) and (parsed == gold)
        elif benchmark_name == "winogrande_mini":
            parsed = _extract_mc_letter(output, "AB")
            is_correct = bool(parsed) and (parsed == gold)
        else:
            parsed = _extract_mc_letter(output, "ABCD")
            is_correct = bool(parsed) and (parsed == gold)

        if parsed:
            answered += 1
        if is_correct:
            correct += 1

        # 立即写入 per-sample CSV（fail-safe：中途崩溃也不丢数据）
        row = {
            "run_id":                run_id,
            "model_id":              model_id,
            "benchmark":             benchmark_name,
            "seed":                  seed,
            "sample_id":             i,
            "question_truncated":    (prompt[:300] + "…") if len(prompt) > 300 else prompt,
            "correct_answer":        gold,
            "model_output_truncated":(output[:200] + "…") if len(output) > 200 else output,
            "parsed_answer":         parsed,
            "is_correct":            is_correct,
            "generation_time_ms":    gen_ms,
            "input_tokens":          in_tok,
            "output_tokens":         out_tok,
        }
        append_row_to_csv(raw_csv_path, row, QUALITY_RAW_FIELDNAMES)

        # 每 10 条打印进度
        done = i + 1 - skipped
        if done % 10 == 0 or (i + 1) == total:
            running_acc = correct / done * 100 if done > 0 else 0.0
            print(f"  [{i+1}/{total}] acc={running_acc:.1f}%  "
                  f"parsed={answered}  skipped={skipped}")

    # 汇总
    valid_total  = total - skipped
    accuracy     = correct  / valid_total * 100 if valid_total > 0 else 0.0
    answer_rate  = answered / valid_total * 100 if valid_total > 0 else 0.0

    print(f"  → {benchmark_name}: {correct}/{valid_total} = {accuracy:.1f}%  "
          f"answer_rate={answer_rate:.0f}%")
    return {
        "benchmark":    benchmark_name,
        "accuracy":     round(accuracy,    2),
        "answer_rate":  round(answer_rate, 2),
        "num_correct":  correct,
        "num_samples":  valid_total,
        "num_skipped":  skipped,
        "seed":         seed,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 文本质量评估套件主入口（exp001）
# ──────────────────────────────────────────────────────────────────────────────

def run_text_quality_suite(
    model: Any,
    tokenizer: Any,
    device: str,
    dataset_dir: str | Path,
    model_result_dir: str | Path,
    run_id: str,
    model_id: str,
    benchmarks: list[str] | None = None,
    seed: int = 42,
) -> list[dict]:
    """运行所有（或指定的）文本质量基准，保存结果。

    结果目录结构（在 model_result_dir 下）：
        quality_raw/
        ├── {run_id}_mmlu_pro_mini_raw.csv
        ├── {run_id}_gsm8k_mini_raw.csv
        ├── {run_id}_hellaswag_mini_raw.csv
        ├── {run_id}_winogrande_mini_raw.csv
        └── {run_id}_truthfulqa_mc_raw.csv
        quality_summary/
        └── {run_id}_quality_summary.json

    Args:
        model: 已加载的语言模型
        tokenizer: 对应分词器
        device: 推理设备
        dataset_dir: 数据集根目录
        model_result_dir: 模型结果根目录（如 results/exp001/Llama-3.2-1B-Instruct/）
        run_id: 与性能测试共享的 run_id
        model_id: HuggingFace repo id
        benchmarks: 要运行的基准名列表；None 表示全部 5 个
        seed: 数据集版本 seed

    Returns:
        每个基准的汇总 dict 列表
    """
    if benchmarks is None:
        benchmarks = list(BENCHMARK_CONFIGS.keys())

    model_result_dir = Path(model_result_dir)
    quality_raw_dir  = model_result_dir / "quality_raw"
    quality_sum_dir  = model_result_dir / "quality_summary"
    quality_raw_dir.mkdir(parents=True, exist_ok=True)
    quality_sum_dir.mkdir(parents=True, exist_ok=True)

    all_results: list[dict] = []

    for bm_name in benchmarks:
        if bm_name not in BENCHMARK_CONFIGS:
            print(f"[Quality] ⚠️  Unknown benchmark: {bm_name!r}, skipping")
            continue

        raw_csv = quality_raw_dir / f"{run_id}_{bm_name}_raw.csv"
        result  = run_single_benchmark(
            benchmark_name=bm_name,
            model=model, tokenizer=tokenizer, device=device,
            dataset_dir=dataset_dir,
            raw_csv_path=str(raw_csv),
            run_id=run_id, model_id=model_id, seed=seed,
        )
        all_results.append(result)

    # 汇总 JSON
    import datetime
    summary = {
        "run_id":        run_id,
        "model_id":      model_id,
        "seed":          seed,
        "timestamp":     datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "benchmarks":    all_results,
        "mean_accuracy": round(
            sum(r["accuracy"] for r in all_results) / len(all_results), 2
        ) if all_results else 0.0,
    }
    summary_path = quality_sum_dir / f"{run_id}_quality_summary.json"
    save_json(summary_path, summary)

    print(f"\n[Quality] ✅ 文本基准完成 — 平均准确率 {summary['mean_accuracy']:.1f}%")
    print(f"[Quality] Summary → {summary_path.name}")
    for r in all_results:
        print(f"  {r['benchmark']:20s}  {r['accuracy']:5.1f}%  "
              f"({r['num_correct']}/{r['num_samples']})")

    return all_results


# ──────────────────────────────────────────────────────────────────────────────
# 视觉质量评估接口（预留，exp002 实现）
# ──────────────────────────────────────────────────────────────────────────────

def run_vision_quality_suite(
    model: Any,
    processor: Any,
    device: str,
    dataset_dir: str | Path,
    model_result_dir: str | Path,
    run_id: str,
    model_id: str,
    benchmarks: list[str] | None = None,
    seed: int = 42,
) -> list[dict]:
    """[预留接口] 视觉质量评估套件，exp002 中实现。

    接口签名与 run_text_quality_suite 对齐，exp002 notebook 可直接调用。
    支持 benchmarks：VQAv2 / MMBench / MathVista / TextVQA / DocVQA。

    Raises:
        NotImplementedError: exp001 不实现此功能
    """
    raise NotImplementedError(
        "视觉质量评估在 exp002 中实现。"
        f"\n可用基准: {list(VISION_BENCHMARK_CONFIGS.keys())}"
    )
