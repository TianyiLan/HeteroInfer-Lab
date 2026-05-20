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

# ── 版本标识 ──────────────────────────────────────────────────────────────────
__version__      = "1.2.2"
__version_info__ = (
    "quality suite v1.0 | "
    "eval: logprob_content(HellaSwag/WinoGrande/TruthfulQA) "
    "+ generate(MMLU-Pro 5-shot / GSM8K 8-shot CoT) | "
    "2026-05-17"
)
# ─────────────────────────────────────────────────────────────────────────────

import json
import re
import time
from pathlib import Path
from typing import Any

import torch

from edge_llm_systems.utils import append_row_to_csv, save_json, build_timestamp_filename, log

# ──────────────────────────────────────────────────────────────────────────────
# 基准配置表
# ──────────────────────────────────────────────────────────────────────────────

BENCHMARK_CONFIGS: dict[str, dict] = {
    "mmlu_pro": {
        "hf_id":             "TIGER-Lab/MMLU-Pro",
        "hf_split":          "test",
        "few_shot_hf_split": "validation",   # 5-shot 示例取自 val，测试取自 test
        "max_samples":       500,            # full test=12K; 500 → ±4.4% CI@95%
        "max_new_tokens":    2048,           # CoT 推理链需要足够 token（官方脚本默认 2048）
        "standard_shots":    5,
        "use_cot":           True,           # 5-shot CoT（官方 TIGER-AI-Lab 协议）
        "eval_method":       "generate",     # Instruct 模型用 generate+字母提取（logprob 受 A/I 词频偏置影响）
        "description":       "大学级多选知识题 (A–J)，5-shot，generate",
    },
    "gsm8k": {
        "hf_id":             "openai/gsm8k",
        "hf_name":           "main",
        "hf_split":          "test",
        "few_shot_hf_split": "train",        # 8-shot CoT 示例取自 train
        "max_samples":       500,            # full test=1319；500 是常用子集
        "max_new_tokens":    512,            # CoT 推理链需要更多 token
        "standard_shots":    8,
        "use_cot":           True,           # 8-shot Chain-of-Thought
        "eval_method":       "generate",     # 数学推理需要生成 CoT 链
        "description":       "数学应用题，8-shot CoT",
    },
    "hellaswag": {
        "hf_id":             "Rowan/hellaswag",
        "hf_split":          "validation",
        "few_shot_hf_split": None,           # content scoring 不需要 few-shot
        "max_samples":       500,            # full val=10K；500 → ±4.4% CI@95%
        "max_new_tokens":    10,
        "standard_shots":    0,
        "use_cot":           False,
        "eval_method":       "logprob_content",  # 对每个 ending 文本打 log-prob 分
        "description":       "常识句子补全，续写文本 log-prob scoring，0-shot",
    },
    "winogrande": {
        "hf_id":             "allenai/winogrande",
        "hf_name":           "winogrande_xl",
        "hf_split":          "validation",
        "few_shot_hf_split": None,           # content scoring 不需要 few-shot
        "max_samples":       500,            # full val=1267
        "max_new_tokens":    10,
        "standard_shots":    0,
        "use_cot":           False,
        "eval_method":       "logprob_content",  # 填空句子 log-prob scoring
        "description":       "代词消歧填空，填空文本 log-prob scoring，0-shot",
    },
    "truthfulqa_mc": {
        "hf_id":             "truthful_qa",
        "hf_name":           "multiple_choice",
        "hf_split":          "validation",
        "few_shot_hf_split": None,           # 0-shot：原论文设计
        "max_samples":       817,            # full val=817，直接跑全集
        "max_new_tokens":    10,
        "standard_shots":    0,
        "use_cot":           False,
        "eval_method":       "logprob_content",  # 对每个选项文本打 log-prob 分
        "description":       "事实性多选题 MC1，选项文本 log-prob scoring，0-shot",
    },
}

# per-sample CSV 字段顺序（qual_raw_{benchmark}_{ts}.csv）
QUALITY_RAW_FIELDNAMES: list[str] = [
    "run_id", "model_id", "benchmark", "seed", "sample_id",
    "question_truncated", "correct_answer",
    "model_output_truncated", "parsed_answer", "is_correct",
    "generation_time_ms", "input_tokens", "output_tokens",
]

# 汇总 CSV 字段顺序（qual_summary_{ts}.csv，一行 = 一个基准）
QUALITY_SUMMARY_FIELDNAMES: list[str] = [
    "run_id", "model_id", "seed", "timestamp",
    "benchmark", "accuracy", "answer_rate",
    "num_correct", "num_samples", "num_skipped",
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
    load_kwargs: dict = {"split": cfg["hf_split"]}
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


def _load_few_shot_examples(
    benchmark_name: str,
    cfg: dict,
    dataset_dir: str | Path,
) -> list[dict]:
    """从 train/dev split 加载 few-shot 示例，本地缓存避免重复下载。

    few-shot 示例与测试集严格分离：
      - 测试集：cfg["hf_split"]（test / validation）
      - few-shot 示例：cfg["few_shot_hf_split"]（train / validation，与测试不重叠）

    缓存路径：{dataset_dir}/quality_suite/few_shot/{benchmark_name}/shots_{n}.json

    Args:
        benchmark_name: BENCHMARK_CONFIGS 中的键
        cfg: 对应配置字典
        dataset_dir: 数据集根目录

    Returns:
        few-shot 样本字典列表；0-shot 时返回空列表
    """
    n_shots        = cfg.get("standard_shots", 0)
    few_shot_split = cfg.get("few_shot_hf_split")

    if n_shots == 0 or not few_shot_split:
        return []

    cache_dir  = Path(dataset_dir) / "quality_suite" / "few_shot" / benchmark_name
    cache_path = cache_dir / f"shots_{n_shots}.json"

    if cache_path.exists():
        with open(cache_path, encoding="utf-8") as f:
            return json.load(f)

    from datasets import load_dataset
    print(f"[Quality] Downloading few-shot examples: {benchmark_name} ({few_shot_split}) ...")
    load_kwargs: dict = {"split": few_shot_split}
    if "hf_name" in cfg:
        load_kwargs["name"] = cfg["hf_name"]
    ds       = load_dataset(cfg["hf_id"], **load_kwargs)
    examples = [dict(ds[i]) for i in range(min(n_shots, len(ds)))]

    cache_dir.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(examples, f, ensure_ascii=False, indent=2)

    print(f"[Quality] {benchmark_name}: {n_shots}-shot examples cached")
    return examples


# ──────────────────────────────────────────────────────────────────────────────
# 推理工具：chat template + 贪婪解码
# ──────────────────────────────────────────────────────────────────────────────

_MCQ_SYSTEM_PROMPT = (
    "You are a multiple choice exam assistant. "
    "Always respond with only the single letter of the correct answer (e.g. A, B, C). "
    "Do not explain, do not repeat the question, do not add any other text."
)

def _generate(
    model: Any,
    tokenizer: Any,
    device: str,
    user_prompt: str,
    max_new_tokens: int,
    system_prompt: str | None = None,
) -> tuple[str, float, int, int]:
    """使用 chat template 格式化 prompt，贪婪解码生成答案。

    Args:
        system_prompt: 可选系统提示，用于约束 Instruct 模型的输出格式。
            对选择题基准传入 _MCQ_SYSTEM_PROMPT 可显著提高单字母合规率。

    Returns:
        (output_text, generation_time_ms, input_token_count, output_token_count)
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
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


def _generate_from_messages(
    model: Any,
    tokenizer: Any,
    device: str,
    messages: list[dict],
    max_new_tokens: int,
    stop_sequences: list[str] | None = None,
) -> tuple[str, float, int, int]:
    """多轮对话格式的 generate，直接接受完整 messages 列表。

    与 _generate() 的区别：接受 system/user/assistant 多轮消息列表，
    适用于 few-shot 示例作为独立对话轮次的场景（MMLU-Pro 5-shot / GSM8K 8-shot）。
    模型能清晰区分"演示例子"和"需要回答的测试题"，解决单消息格式下的误解问题。

    stop_sequences: 遇到这些字符串时截断输出（如 ["Question:"] 防止模型自生成下一题）。

    Returns:
        (output_text, generation_time_ms, input_token_count, output_token_count)
    """
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs    = tokenizer(formatted, return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[-1]

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    gen_ms = (time.perf_counter() - t0) * 1000.0

    new_ids     = out[0, input_len:]
    output_text = tokenizer.decode(new_ids, skip_special_tokens=True).strip()

    # 按 stop_sequences 截断（generate() 原生 stopping_criteria 较复杂，后处理更简单）
    if stop_sequences:
        for stop in stop_sequences:
            idx = output_text.find(stop)
            if idx != -1:
                output_text = output_text[:idx].strip()

    return output_text, round(gen_ms, 1), input_len, len(new_ids)


def _score_mc(
    model: Any,
    tokenizer: Any,
    device: str,
    prompt: str,
    option_letters: list[str],
) -> tuple[str, str, float, int]:
    """Log-probability scoring for MCQ benchmarks (academic standard).

    Single forward pass; picks the option letter with highest log-prob at the
    last token position. Eliminates format-compliance and A-bias issues.

    Each letter is looked up as " A" first (space-prefix, common in SentencePiece
    tokenizers like LLaMA), then as bare "A". Only single-token encodings count;
    multi-token encodings fall back to -inf so they are never chosen.

    Returns:
        (best_letter, display_str, score_time_ms, input_token_count)
        display_str example: "B [A:-2.31, B:-1.09, C:-3.42, D:-2.78]"
    """
    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs    = tokenizer(formatted, return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[-1]

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    with torch.no_grad():
        outputs = model(**inputs)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    score_ms = (time.perf_counter() - t0) * 1000.0

    logits    = outputs.logits[0, -1, :]
    log_probs = torch.log_softmax(logits.float(), dim=-1)

    scores: dict[str, float] = {}
    for letter in option_letters:
        found = False
        for variant in (f" {letter}", letter):
            ids = tokenizer.encode(variant, add_special_tokens=False)
            if len(ids) == 1:
                scores[letter] = log_probs[ids[0]].item()
                found = True
                break
        if not found:
            scores[letter] = float("-inf")

    best        = max(scores, key=scores.__getitem__)
    score_str   = ", ".join(f"{l}:{scores[l]:.2f}" for l in option_letters)
    display_str = f"{best} [{score_str}]"
    return best, display_str, round(score_ms, 1), input_len


def _score_continuation(
    model: Any,
    tokenizer: Any,
    device: str,
    context: str,
    continuation: str,
) -> tuple[float, float, int]:
    """Log-probability scoring of a continuation given context (academic standard).

    Scores how naturally the continuation follows the context by computing the
    mean per-token log-prob of the continuation tokens. Uses raw tokenization
    (no chat template) — comparable to lm-eval-harness base-model evaluation.

    Used for HellaSwag (continuation of activity), WinoGrande (cloze fill-in),
    and TruthfulQA MC1 (factual answer continuation).

    Tokenization:
        context   → tokenized WITH special tokens (adds BOS for LLaMA)
        continuation → tokenized WITHOUT special tokens (no extra BOS)
        full sequence = context_enc + continuation_enc

    Args:
        context:      The conditioning text (question, sentence prefix, etc.)
        continuation: The candidate text to score (ending, option, answer text)

    Returns:
        (mean_logprob_per_token, score_time_ms, n_continuation_tokens)
        Higher mean_logprob = more likely continuation.
    """
    ctx_enc  = tokenizer.encode(context,      add_special_tokens=True)
    cont_enc = tokenizer.encode(continuation, add_special_tokens=False)

    if not cont_enc:
        return float("-inf"), 0.0, 0

    n_ctx  = len(ctx_enc)
    n_cont = len(cont_enc)
    full_ids = torch.tensor([ctx_enc + cont_enc]).to(device)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    with torch.no_grad():
        outputs = model(input_ids=full_ids)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    score_ms = (time.perf_counter() - t0) * 1000.0

    # logits[i] predicts token at position i+1 in the full sequence
    logits    = outputs.logits[0, :-1, :]            # shape: [seq_len-1, vocab]
    log_probs = torch.log_softmax(logits.float(), dim=-1)

    cont_ids = torch.tensor(cont_enc).to(device)
    # Continuation token at position n_ctx + i is predicted by logits[n_ctx - 1 + i]
    total = sum(
        log_probs[n_ctx - 1 + i, cont_ids[i]].item()
        for i in range(n_cont)
    )
    mean_lp = total / n_cont
    return mean_lp, round(score_ms, 1), n_cont


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


def _extract_cot_answer(text: str, valid: str = "ABCDEFGHIJ") -> str:
    """从 CoT 输出中提取答案，优先匹配官方格式 'the answer is (X)'。

    优先级：
      1. "the answer is (X)" / "the answer is X"（官方 MMLU-Pro CoT 格式）
      2. 末尾独立的有效字母（回退）
    """
    text_up   = text.strip().upper()
    valid_pat = "[" + re.escape(valid.upper()) + "]"

    # 官方格式：the answer is (X) 或 the answer is X
    m = re.search(rf"THE ANSWER IS\s*\(?\s*({valid_pat})\s*\)?", text_up)
    if m:
        return m.group(1)

    # 回退：取文本末尾部分最后一个独立有效字母（避免把选项序号误判）
    tail = text_up[-200:]
    matches = re.findall(rf"\b({valid_pat})\b", tail)
    return matches[-1] if matches else ""


def _extract_num(text: str) -> str:
    """从文本中提取最后一个数字（用于 GSM8K）。

    去除千位逗号，返回字符串形式（便于精确匹配）。
    """
    nums = re.findall(r"-?\d+(?:,\d+)*(?:\.\d+)?", text)
    return nums[-1].replace(",", "") if nums else ""


# ──────────────────────────────────────────────────────────────────────────────
# Prompt 构建函数（few-shot + CoT）
#
# 每个函数接收 few_shot_examples（0-shot 时为空列表）和 test_sample，
# 返回完整的 user message 字符串（传入 apply_chat_template 的 content）。
# ──────────────────────────────────────────────────────────────────────────────

def _build_mmlu_pro_prompt(
    few_shot_examples: list[dict],
    test_sample: dict,
) -> tuple[str, str]:
    """MMLU-Pro 5-shot：大学级知识多选题（A–J）。

    示例格式（每个 few-shot example）：
        Question: ...
        Options:
        A. ...  B. ...
        Answer: C

    测试格式（末尾不给答案，与 few-shot 结尾一致）：
        Question: ...
        Options:
        A. ...
        Answer:
    """
    lines = ["The following are multiple choice questions (with answers). "
             "Reply with only the answer letter.\n"]

    for ex in few_shot_examples:
        opts    = ex.get("options", [])
        opt_str = "\n".join(f"{chr(65 + i)}. {o}" for i, o in enumerate(opts))
        ans     = str(ex.get("answer", "")).strip().upper()
        lines.append(
            f"Question: {ex.get('question', '')}\n\n"
            f"Options:\n{opt_str}\n\n"
            f"Answer: {ans}\n"
        )

    opts    = test_sample.get("options", [])
    opt_str = "\n".join(f"{chr(65 + i)}. {o}" for i, o in enumerate(opts))
    lines.append(
        f"Question: {test_sample.get('question', '')}\n\n"
        f"Options:\n{opt_str}\n\n"
        f"Answer:"
    )
    correct = str(test_sample.get("answer", "")).strip().upper()
    return "\n".join(lines), correct


def _build_mmlu_pro_messages(
    few_shot_examples: list[dict],
    test_sample: dict,
) -> tuple[list[dict], str]:
    """MMLU-Pro 5-shot CoT 多轮对话格式（官方 TIGER-AI-Lab 协议）。

    系统提示使用官方模板，要求逐步推理并以 'the answer is (X)' 结尾。
    每个 few-shot 示例的 assistant 回复使用数据集自带的 cot_content 字段
    （预写推理链），确保示范风格和官方一致。

    格式：
        [system]: The following are multiple choice questions ... Think step by step
                  and finish with 'the answer is (X)'.
        [user]:   Question: 例1 ... Answer:
        [assistant]: Let's think step by step. ... the answer is (C).
        ...
        [user]:   Question: 测试题 ... Answer:   ← 模型在此生成 CoT
    """
    category = test_sample.get("category", "")
    subject  = category.replace("_", " ") if category else "the following topic"

    system_content = (
        f"The following are multiple choice questions (with answers) about {subject}. "
        f"Think step by step and then finish your answer with "
        f"\"the answer is (X)\" where X is the correct letter choice."
    )
    messages: list[dict] = [{"role": "system", "content": system_content}]

    for ex in few_shot_examples:
        opts    = ex.get("options", [])
        opt_str = "\n".join(f"{chr(65 + i)}. {o}" for i, o in enumerate(opts))
        # cot_content 是数据集自带的推理链，原始格式为
        # "X: Let's think step by step. ... the answer is (X)."
        # 去掉开头的 "X: " 前缀，避免模型学到"字母优先输出"模式（导致 A-bias）
        cot = str(ex.get("cot_content", "")).strip()
        cot = re.sub(r"^[A-J]:\s*", "", cot)   # 去掉 "X: " 前缀
        if not cot:
            # 极少数样本无 cot_content，回退到单字母
            cot = str(ex.get("answer", "")).strip().upper()
        messages.append({
            "role": "user",
            "content": (
                f"Question: {ex.get('question', '')}\n\n"
                f"Options:\n{opt_str}\n\n"
                f"Answer:"
            ),
        })
        messages.append({"role": "assistant", "content": cot})

    opts    = test_sample.get("options", [])
    opt_str = "\n".join(f"{chr(65 + i)}. {o}" for i, o in enumerate(opts))
    messages.append({
        "role": "user",
        "content": (
            f"Question: {test_sample.get('question', '')}\n\n"
            f"Options:\n{opt_str}\n\n"
            f"Answer: Let's think step by step."  # 引导模型直接进入 CoT，不输出字母前缀
        ),
    })

    correct = str(test_sample.get("answer", "")).strip().upper()
    return messages, correct


def _build_gsm8k_cot_prompt(
    few_shot_examples: list[dict],
    test_sample: dict,
) -> tuple[str, str]:
    """GSM8K 8-shot Chain-of-Thought：数学推理，答案以 '#### <number>' 结尾。

    示例格式（来自 train split，answer 字段已包含 CoT + #### 格式）：
        Problem: Tom has 5 apples...
        Solution: Tom starts with 5... 5 + 3 = 8. #### 8

    测试格式：
        Problem: ...
        Solution:       ← 让模型续写 CoT
    """
    lines = [
        "Solve each math problem step by step. "
        "Show your reasoning and end your answer with '#### <number>'.\n"
    ]

    for ex in few_shot_examples:
        answer_text = ex.get("answer", "")
        lines.append(
            f"Problem: {ex.get('question', '')}\n"
            f"Solution: {answer_text}\n"
        )

    lines.append(
        f"Problem: {test_sample.get('question', '')}\n"
        f"Solution:"
    )

    # 从 answer 字段提取数字作为 gold label
    answer_text = test_sample.get("answer", "")
    if "####" in answer_text:
        correct = answer_text.split("####")[-1].strip().replace(",", "")
    else:
        correct = _extract_num(answer_text)
    return "\n".join(lines), correct


def _build_gsm8k_messages(
    few_shot_examples: list[dict],
    test_sample: dict,
) -> tuple[list[dict], str]:
    """GSM8K 8-shot 多轮对话格式（Instruct 模型标准做法）。

    每个 few-shot 示例作为独立 user/assistant 轮次，测试题作为最后一个 user 消息。
    解决单消息格式下模型误以为有 8 道题需要全解的问题。

    格式：
        [system]: 逐步解题，结尾 #### <数字>
        [user]:   Problem: 例1
        [assistant]: 解题过程... #### 72
        ...
        [user]:   Problem: 测试题   ← 模型在此生成
    """
    messages: list[dict] = [
        {
            "role": "system",
            "content": (
                "Solve math problems step by step. "
                "Show your reasoning and end your answer with '#### <number>'."
            ),
        }
    ]

    for ex in few_shot_examples:
        messages.append({
            "role": "user",
            "content": f"Problem: {ex.get('question', '')}",
        })
        messages.append({
            "role": "assistant",
            "content": ex.get("answer", ""),
        })

    messages.append({
        "role": "user",
        "content": f"Problem: {test_sample.get('question', '')}",
    })

    answer_text = test_sample.get("answer", "")
    if "####" in answer_text:
        correct = answer_text.split("####")[-1].strip().replace(",", "")
    else:
        correct = _extract_num(answer_text)
    return messages, correct


def _build_hellaswag_prompt(
    few_shot_examples: list[dict],
    test_sample: dict,
) -> tuple[str, str]:
    """HellaSwag 10-shot：常识句子补全（A–D）。

    示例格式：
        Text: [context]
        A. [ending0]  B. [ending1]  C. [ending2]  D. [ending3]
        Answer: B

    测试格式：
        Text: [context]
        A. ...
        Answer:
    """
    lines = ["Choose the best ending for each text. Reply with only the letter A, B, C, or D.\n"]

    for ex in few_shot_examples:
        endings = ex.get("endings", [])
        opt_str = "\n".join(f"{chr(65 + i)}. {e}" for i, e in enumerate(endings))
        gold    = chr(65 + int(ex.get("label", 0)))
        lines.append(
            f"Text: {ex.get('ctx', '')}\n"
            f"{opt_str}\n"
            f"Answer: {gold}\n"
        )

    endings = test_sample.get("endings", [])
    opt_str = "\n".join(f"{chr(65 + i)}. {e}" for i, e in enumerate(endings))
    lines.append(
        f"Text: {test_sample.get('ctx', '')}\n"
        f"{opt_str}\n"
        f"Answer:"
    )
    correct = chr(65 + int(test_sample.get("label", 0)))
    return "\n".join(lines), correct


def _build_winogrande_prompt(
    few_shot_examples: list[dict],
    test_sample: dict,
) -> tuple[str, str]:
    """WinoGrande 5-shot：代词消歧填空（A/B）。

    示例格式：
        Sentence: [sentence with _]
        A. [option1]  B. [option2]
        Answer: A

    测试格式：
        Sentence: ...
        A. ...  B. ...
        Answer:
    """
    lines = ["Choose A or B to fill in the blank. Reply with only the letter A or B.\n"]

    for ex in few_shot_examples:
        gold = "A" if str(ex.get("answer", "1")) == "1" else "B"
        lines.append(
            f"Sentence: {ex.get('sentence', '')}\n"
            f"A. {ex.get('option1', '')}\n"
            f"B. {ex.get('option2', '')}\n"
            f"Answer: {gold}\n"
        )

    correct = "A" if str(test_sample.get("answer", "1")) == "1" else "B"
    lines.append(
        f"Sentence: {test_sample.get('sentence', '')}\n"
        f"A. {test_sample.get('option1', '')}\n"
        f"B. {test_sample.get('option2', '')}\n"
        f"Answer:"
    )
    return "\n".join(lines), correct


def _build_truthfulqa_prompt(
    few_shot_examples: list[dict],
    test_sample: dict,
) -> tuple[str | None, str | None]:
    """TruthfulQA MC1：0-shot（原论文设计，加示例反而干扰）。

    Returns:
        (prompt, correct_letter)，若样本无选项则返回 (None, None)
    """
    mc1     = test_sample.get("mc1_targets", {})
    choices = mc1.get("choices", [])
    labels  = mc1.get("labels",  [])
    if not choices:
        return None, None

    correct_idx = labels.index(1) if 1 in labels else 0
    correct     = chr(65 + correct_idx)
    opt_str     = "\n".join(f"{chr(65 + i)}. {c}" for i, c in enumerate(choices))
    prompt = (
        f"Question: {test_sample.get('question', '')}\n\n"
        f"{opt_str}\n\n"
        f"Answer:"
    )
    return prompt, correct


# ──────────────────────────────────────────────────────────────────────────────
# Content scoring 构建函数（logprob_content 路径）
#
# 返回 (context, [continuation_0, ...], correct_letter)
# context 和 continuation 均为原始文本（不经过 chat template）。
# ──────────────────────────────────────────────────────────────────────────────

def _hellaswag_preprocess(text: str) -> str:
    """lm-eval-harness 兼容的 HellaSwag 文本预处理。

    去除 WikiHow 数据集中的 [title] / [header] / [substeps] 等括号标记，
    与 Open LLM Leaderboard 评估标准保持一致。
    """
    text = text.strip()
    text = text.replace(" [title]", ". ")
    text = re.sub(r"\[.*?\]", "", text)
    text = text.replace("  ", " ")
    return text


def _build_hellaswag_content(
    sample: dict,
) -> tuple[str, list[str], str]:
    """HellaSwag：给定活动描述上文，对每个 ending 文本计算 log-prob 并选最高。

    lm-eval-harness 标准：
        context       = preprocess(activity_label + ": " + ctx)
        continuations = [" " + preprocess(ending), ...]   # space-prefix

    activity_label 是关键上下文（如 "How to bake a cake"），缺少时模型无法
    判断活动类型，discriminability 大幅下降。space-prefix 保证 SentencePiece
    分词器在 word boundary 处正确加 ▁ 前缀。
    """
    activity = sample.get("activity_label", "")
    ctx      = sample.get("ctx", "")
    endings  = sample.get("endings", [])
    correct  = chr(65 + int(sample.get("label", 0)))

    context       = _hellaswag_preprocess(f"{activity}: {ctx}")
    continuations = [" " + _hellaswag_preprocess(e) for e in endings]
    return context, continuations, correct


def _build_winogrande_content(
    sample: dict,
) -> tuple[str, list[str], str]:
    """WinoGrande：cloze 填空，对两个选项各填入空白后打 log-prob。

    context     = 句子中 '_' 之前的部分（包含尾部空格）
    continuations = [option1 + 句子后半段, option2 + 句子后半段]
    """
    sentence = sample.get("sentence", "")
    opt1     = sample.get("option1", "")
    opt2     = sample.get("option2", "")
    correct  = "A" if str(sample.get("answer", "1")) == "1" else "B"

    parts = sentence.split("_", 1)
    if len(parts) == 2:
        ctx  = parts[0]               # text before blank (may end with space)
        rest = parts[1]               # text after blank
        continuations = [opt1 + rest, opt2 + rest]
    else:
        # fallback: no blank found, compare full sentence with each option appended
        ctx           = sentence
        continuations = [opt1, opt2]

    return ctx, continuations, correct


def _build_truthfulqa_content(
    sample: dict,
) -> tuple[str | None, list[str] | None, str | None]:
    """TruthfulQA MC1：对每个选项文本计算 log-prob（lm-eval-harness 标准）。

    context     = "Q: {question}\\nA:"
    continuations = [" {choice0}", " {choice1}", ...]  # space-prefix 与训练分布一致
    """
    mc1     = sample.get("mc1_targets", {})
    choices = mc1.get("choices", [])
    labels  = mc1.get("labels",  [])
    if not choices:
        return None, None, None

    correct_idx   = labels.index(1) if 1 in labels else 0
    correct       = chr(65 + correct_idx)
    context       = f"Q: {sample.get('question', '')}\nA:"
    continuations = [" " + c for c in choices]   # space-prefix
    return context, continuations, correct


# ── 分发表 ──────────────────────────────────────────────────────────────────

# generate：多轮对话 messages builders（返回 list[dict]）
_PROMPT_BUILDERS = {
    "mmlu_pro": _build_mmlu_pro_messages,   # 5-shot 多轮，字母提取
    "gsm8k":    _build_gsm8k_messages,      # 8-shot 多轮，数字提取
}

# logprob_content：raw-text content builders
_CONTENT_BUILDERS = {
    "hellaswag":     _build_hellaswag_content,
    "winogrande":    _build_winogrande_content,
    "truthfulqa_mc": _build_truthfulqa_content,
}


def _get_option_letters(benchmark_name: str, sample: dict) -> list[str]:
    """Return valid option letters for logprob (letter-scoring) benchmarks.

    Only called for eval_method == "logprob" (currently only MMLU-Pro).
    """
    if benchmark_name == "mmlu_pro":
        n = len(sample.get("options", []))
        return [chr(65 + i) for i in range(n)]
    return ["A", "B", "C", "D"]


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

    # few-shot 示例：循环前一次性加载，所有样本共用同一套示例
    few_shot_examples = _load_few_shot_examples(benchmark_name, cfg, dataset_dir)
    n_shots     = len(few_shot_examples)
    cot_tag     = " + CoT" if cfg.get("use_cot") else ""
    eval_method = cfg.get("eval_method", "generate")
    print(f"\n[Quality] {benchmark_name} — {total} samples, "
          f"{n_shots}-shot{cot_tag}, eval={eval_method}  ({cfg['description']})")

    correct  = 0
    answered = 0
    skipped  = 0

    for i, sample in enumerate(samples):

        if eval_method == "logprob_content":
            # ── 续写文本 log-prob scoring（HellaSwag / WinoGrande / TruthfulQA）
            ctx, continuations, gold = _CONTENT_BUILDERS[benchmark_name](sample)

            if ctx is None:          # 无效样本（TruthfulQA 无选项）
                skipped += 1
                continue

            prompt     = ctx         # 用于 question_truncated 字段
            best_score = float("-inf")
            best_idx   = 0
            score_parts: list[str] = []
            total_ms   = 0.0
            total_cont_tokens = 0

            for j, cont in enumerate(continuations):
                mean_lp, ms, n_tok = _score_continuation(
                    model, tokenizer, device, ctx, cont
                )
                score_parts.append(f"{chr(65 + j)}:{mean_lp:.2f}")
                total_ms          += ms
                total_cont_tokens += n_tok
                if mean_lp > best_score:
                    best_score = mean_lp
                    best_idx   = j

            parsed     = chr(65 + best_idx)
            is_correct = (parsed == gold)
            answered  += 1                   # content scoring 永远有答案
            output     = f"{parsed} [{', '.join(score_parts)}]"
            gen_ms     = round(total_ms, 1)
            in_tok     = 0                   # 多次前向传播，不单独统计输入 token
            out_tok    = total_cont_tokens   # 各选项 continuation token 总数

        elif eval_method == "logprob":
            # ── 字母 log-prob scoring（MMLU-Pro）
            prompt, gold = _PROMPT_BUILDERS[benchmark_name](few_shot_examples, sample)
            if prompt is None:
                skipped += 1
                continue
            option_letters = _get_option_letters(benchmark_name, sample)
            parsed, output, gen_ms, in_tok = _score_mc(
                model, tokenizer, device, prompt, option_letters
            )
            out_tok    = 1
            is_correct = (parsed == gold)
            answered  += 1

        else:
            # ── generate 路径：MMLU-Pro（5-shot 多轮 + 字母提取）+ GSM8K（8-shot 多轮 + 数字提取）
            messages, gold = _PROMPT_BUILDERS[benchmark_name](few_shot_examples, sample)
            if messages is None:
                skipped += 1
                continue
            # 取最后一条 user 消息作为日志/CSV 的 prompt 字段
            prompt = messages[-1]["content"] if messages else ""

            # MMLU-Pro：stop at "Question:" 防止模型自生成下一道题
            stop_seqs = ["Question:"] if benchmark_name == "mmlu_pro" else None
            output, gen_ms, in_tok, out_tok = _generate_from_messages(
                model, tokenizer, device, messages, cfg["max_new_tokens"],
                stop_sequences=stop_seqs,
            )

            if benchmark_name == "mmlu_pro":
                # MMLU-Pro CoT：提取 "the answer is (X)" 格式答案
                n_opts = len(sample.get("options", []))
                valid  = "".join(chr(65 + i) for i in range(n_opts))
                parsed = _extract_cot_answer(output, valid=valid)
                is_correct = bool(parsed) and (parsed == gold)
                if parsed:
                    answered += 1

            else:
                # GSM8K：提取 #### 后的数字
                if "####" in output:
                    parsed = output.split("####")[-1].strip().replace(",", "")
                    parsed = _extract_num(parsed) or parsed
                else:
                    parsed = _extract_num(output)
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

    结果直接平铺在 model_result_dir 下（与性能结果同级）：
        qual_raw_{benchmark}_{ts}.csv   — per-sample 原始数据（每个基准一个文件）
        qual_summary_{ts}.json          — 汇总分数（所有基准）
    通过文件名前缀 qual_ 与性能文件（perf_ 前缀）区分，run_id 字段关联两者。

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
    model_result_dir.mkdir(parents=True, exist_ok=True)

    all_results: list[dict] = []

    for bm_name in benchmarks:
        if bm_name not in BENCHMARK_CONFIGS:
            print(f"[Quality] ⚠️  Unknown benchmark: {bm_name!r}, skipping")
            continue

        # qual_raw_{benchmark}_{YYYYMMDD_HHMMSS}.csv — 平铺在模型结果目录
        raw_csv = model_result_dir / build_timestamp_filename(f"qual_raw_{bm_name}", "csv")
        result  = run_single_benchmark(
            benchmark_name=bm_name,
            model=model, tokenizer=tokenizer, device=device,
            dataset_dir=dataset_dir,
            raw_csv_path=str(raw_csv),
            run_id=run_id, model_id=model_id, seed=seed,
        )
        all_results.append(result)

    # 汇总 CSV：qual_summary_{ts}.csv — 一行 = 一个基准，与 perf_summary 格式对齐
    import datetime
    ts_str       = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    summary_path = model_result_dir / build_timestamp_filename("qual_summary", "csv")

    for r in all_results:
        row = {
            "run_id":       run_id,
            "model_id":     model_id,
            "seed":         seed,
            "timestamp":    ts_str,
            "benchmark":    r["benchmark"],
            "accuracy":     r["accuracy"],
            "answer_rate":  r["answer_rate"],
            "num_correct":  r["num_correct"],
            "num_samples":  r["num_samples"],
            "num_skipped":  r["num_skipped"],
        }
        append_row_to_csv(summary_path, row, QUALITY_SUMMARY_FIELDNAMES)

    mean_acc = (
        round(sum(r["accuracy"] for r in all_results) / len(all_results), 2)
        if all_results else 0.0
    )
    log(f"[Quality] ✅ 文本基准完成 — 平均准确率 {mean_acc:.1f}%")
    log(f"[Quality] Summary → {summary_path.name}")
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
