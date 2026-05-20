"""工具函数：环境检测、CSV 读写、日志、哈希。

提供实验基础设施所需的通用工具，与模型/推理逻辑解耦，
便于在不同运行环境（Colab / Kaggle / 本地）中复用。
"""

from __future__ import annotations

import csv
import datetime
import hashlib
import os
from pathlib import Path
from typing import Any


# ──────────────────────────────────────────────────────────────────────────────
# 环境检测
# ──────────────────────────────────────────────────────────────────────────────

def get_hw_slug() -> str:
    """返回当前 GPU 的文件系统安全短名，用于结果目录命名。

    从 torch.cuda.get_device_name(0) 读取 GPU 名称，去除 "NVIDIA "、
    "Tesla " 等厂商前缀，并截断内存/变体后缀（如 "-SXM4-40GB"）。

    示例：
        "Tesla T4"              → "T4"
        "NVIDIA L4"             → "L4"
        "NVIDIA A100-SXM4-40GB" → "A100"
        "Tesla V100-SXM2-16GB"  → "V100"

    无 GPU 时返回 "cpu"。
    """
    import torch
    if not torch.cuda.is_available():
        return "cpu"

    name = torch.cuda.get_device_name(0)
    for prefix in ("NVIDIA ", "Tesla ", "GeForce ", "Quadro "):
        name = name.replace(prefix, "")
    name = name.strip()
    # 将每个空格分段的 token 截断到连字符之前，去掉显存/型号后缀
    parts = [p.split("-")[0] for p in name.split()]
    return "_".join(parts)


def detect_environment() -> str:
    """检测当前运行环境。

    通过特征路径和环境变量区分不同平台：
    - Colab：存在 /content 目录
    - Kaggle：存在 KAGGLE_KERNEL_RUN_TYPE 环境变量
    - 本地：其他情况

    Returns:
        "colab" / "kaggle" / "local"
    """
    # Kaggle 通过环境变量标识
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
        return "kaggle"
    # Colab 通过 /content 路径标识（Colab 默认工作目录）
    if Path("/content").exists():
        return "colab"
    return "local"


def check_drive_mounted(drive_root: str) -> bool:
    """检查 Google Drive 是否已挂载，即目标路径是否存在。

    Args:
        drive_root: Drive 根目录路径，如 "/content/drive/MyDrive/EdgeLLM-Systems"

    Returns:
        True 表示路径存在（已挂载），False 表示不存在
    """
    return Path(drive_root).exists()


# ──────────────────────────────────────────────────────────────────────────────
# CSV 读写
# ──────────────────────────────────────────────────────────────────────────────

def read_csv_utf8sig(path: str | Path) -> list[dict]:
    """以 UTF-8-sig 编码读取 CSV 文件（兼容 Excel 生成的 BOM 头）。

    Args:
        path: CSV 文件路径

    Returns:
        每行为一个 dict 的列表
    """
    path = Path(path)
    if not path.exists():
        return []

    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def append_row_to_csv(
    path: str | Path,
    row: dict,
    fieldnames: list[str],
) -> None:
    """以 UTF-8-sig append 模式写入一行 CSV。

    文件不存在时自动写入 header；文件已存在则直接追加数据行。
    每次 run 完成后立即调用，确保数据不因中途崩溃而丢失。

    Args:
        path: CSV 文件路径
        row: 要写入的数据字典
        fieldnames: 字段名列表，决定列顺序
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # 判断文件是否需要写 header（文件不存在或为空）
    write_header = not path.exists() or path.stat().st_size == 0

    with open(path, mode="a", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# ──────────────────────────────────────────────────────────────────────────────
# 日志打印
# ──────────────────────────────────────────────────────────────────────────────

def log(msg: str) -> None:
    """带时间戳的日志打印，格式：[HH:MM:SS] msg

    Args:
        msg: 日志消息
    """
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")


# ──────────────────────────────────────────────────────────────────────────────
# 哈希与 ID 生成
# ──────────────────────────────────────────────────────────────────────────────

def model_hash(model_id: str) -> str:
    """生成模型 ID 的短哈希（sha256 前 8 位），用于唯一标识实验配置。

    Args:
        model_id: 模型标识符，如 "meta-llama/Llama-3.2-1B-Instruct"

    Returns:
        8 位十六进制字符串
    """
    return hashlib.sha256(model_id.encode()).hexdigest()[:8]


def generate_run_id() -> str:
    """生成时间戳格式的 run ID，用于唯一标识一次完整实验运行。

    格式：run_YYYYMMDD_HHMMSS

    Returns:
        如 "run_20260514_143022"
    """
    return datetime.datetime.now().strftime("run_%Y%m%d_%H%M%S")


def build_timestamp_filename(prefix: str, ext: str) -> str:
    """生成带时间戳的文件名，防止重复运行覆盖历史数据。

    Args:
        prefix: 文件名前缀，如 "raw_runs"
        ext: 文件扩展名，如 "csv"（不含点）

    Returns:
        如 "raw_runs_20260514_143022.csv"
    """
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{ts}.{ext}"


# ──────────────────────────────────────────────────────────────────────────────
# Exp Info：环境、模型、运行配置 JSON 收集与保存
# ──────────────────────────────────────────────────────────────────────────────

def save_json(path: str | Path, data: dict) -> None:
    """将 dict 以格式化 JSON 保存到指定路径（UTF-8 编码）。

    目录不存在时自动创建。

    Args:
        path: 目标文件路径
        data: 要保存的字典
    """
    import json
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def collect_environment_info(
    run_id: str,
    model_id: str,
    model_key: str,
    raw_csv_path: str,
    summary_csv_path: str,
) -> dict:
    """收集运行时环境信息，用于实验复现和跨平台对比。

    包含：Python 版本、平台、PyTorch/transformers 版本、GPU 型号、
    CUDA 版本、显存大小、本次实验的输出文件路径。

    Args:
        run_id: 本次运行的唯一标识符
        model_id: HuggingFace repo id
        model_key: MODEL_REGISTRY 中的键
        raw_csv_path: raw_runs CSV 路径
        summary_csv_path: group_summary CSV 路径

    Returns:
        环境信息字典
    """
    import platform as _platform
    import torch
    import transformers
    import pandas as pd

    info: dict = {
        "run_id": run_id,
        "model_id": model_id,
        "model_key": model_key,
        "python": _platform.python_version(),
        "platform": _platform.platform(),
        "pytorch": torch.__version__,
        "transformers": transformers.__version__,
        "pandas": pd.__version__,
        "cuda_available": torch.cuda.is_available(),
    }

    if torch.cuda.is_available():
        info["cuda_runtime"] = torch.version.cuda
        info["gpu"] = torch.cuda.get_device_name(0)
        info["cuda_capability"] = list(torch.cuda.get_device_capability(0))
        props = torch.cuda.get_device_properties(0)
        info["gpu_memory_gb"] = round(props.total_memory / 1024 ** 3, 2)

    info["raw_csv_path"] = str(raw_csv_path)
    info["summary_csv_path"] = str(summary_csv_path)
    return info


def collect_model_info(
    model: Any,
    model_id: str,
    model_key: str,
    local_model_path: str | Path,
) -> dict:
    """收集模型架构元数据，用于记录实验配置和估算 KV Cache。

    包含：层数、注意力头数、KV 头数、hidden size、head dim、
    参数量、FP16 显存占用、每千 token 的 KV Cache 大小。

    Args:
        model: 已加载的模型对象
        model_id: HuggingFace repo id
        model_key: MODEL_REGISTRY 中的键
        local_model_path: 本地模型目录路径

    Returns:
        模型信息字典
    """
    cfg = model.config

    num_layers    = getattr(cfg, "num_hidden_layers", None)
    num_attn      = getattr(cfg, "num_attention_heads", None)
    num_kv_heads  = getattr(cfg, "num_key_value_heads", num_attn)
    hidden_size   = getattr(cfg, "hidden_size", None)
    head_dim      = getattr(cfg, "head_dim",
                            (hidden_size // num_attn
                             if hidden_size and num_attn else None))

    # 参数量统计（仅计算可训练参数，忽略 buffer）
    total_params   = sum(p.numel() for p in model.parameters())
    param_size_mb  = round(total_params * 2 / 1024 ** 2, 1)   # FP16 = 2 bytes

    # 每千 token 的 KV Cache 开销：2（K+V）× 层数 × kv_heads × head_dim × 1000 × 2（FP16）
    kv_mb_per_1k: float | None = None
    if all(v is not None for v in [num_layers, num_kv_heads, head_dim]):
        kv_bytes     = 2 * num_layers * num_kv_heads * head_dim * 1000 * 2
        kv_mb_per_1k = round(kv_bytes / 1024 ** 2, 2)

    return {
        "model_id":            model_id,
        "model_key":           model_key,
        "model_path":          str(local_model_path),
        "model_type":          getattr(cfg, "model_type", "unknown"),
        "layers":              num_layers,
        "hidden_size":         hidden_size,
        "attention_heads":     num_attn,
        "kv_heads":            num_kv_heads,
        "head_dim":            head_dim,
        "torch_dtype":         "fp16",
        "parameter_count_b":   round(total_params / 1e9, 3),
        "parameter_size_mb":   param_size_mb,
        "kv_mb_per_1k_tokens": kv_mb_per_1k,
    }


def collect_run_config(
    run_id: str,
    model_id: str,
    model_key: str,
    config: dict,
    raw_csv_path: str,
    summary_csv_path: str,
) -> dict:
    """收集本次实验的完整运行配置，便于复现和回溯。

    包含：实验名称、模型信息、输入模式、参数矩阵、输出路径。

    Args:
        run_id: 本次运行的唯一标识符
        model_id: HuggingFace repo id
        model_key: MODEL_REGISTRY 中的键
        config: notebook Section 3 确认的配置字典
        raw_csv_path: raw_runs CSV 路径
        summary_csv_path: group_summary CSV 路径

    Returns:
        运行配置字典
    """
    # v2.2: modality 取代旧的 input_mode；兼容旧 key
    modality = config.get("modality", config.get("input_mode", "text"))
    # 兼容旧值：text_only → text
    if modality == "text_only":
        modality = "text"
    elif modality in ("single_image", "multi_images"):
        modality = "vision"

    info: dict = {
        "run_id":          run_id,
        "experiment_name": "exp001_llama32_stage1_v2.2",
        "model_id":        model_id,
        "model_key":       model_key,
        "modality":        modality,
        "torch_dtype":     "float16",
        "enable_memory":     config.get("enable_memory"),
        "enable_efficiency": config.get("enable_efficiency"),
        "enable_quality":    config.get("enable_quality"),
        "raw_csv_path":    str(raw_csv_path),
        "summary_csv_path": str(summary_csv_path),
    }

    if modality == "text":
        info["memory_prompt_lengths"]     = config.get("memory_prompt_lengths", [])
        info["memory_gen_lengths"]        = config.get("memory_gen_lengths", [])
        info["memory_repeat"]             = config.get("memory_repeat")
        info["efficiency_prompt_lengths"] = config.get("efficiency_prompt_lengths", [])
        info["efficiency_gen_lengths"]    = config.get("efficiency_gen_lengths", [])
        info["efficiency_repeat"]         = config.get("efficiency_repeat")
    else:
        info["image_counts"]      = config.get("image_counts", [])
        info["image_resolutions"] = config.get("image_resolutions", [])
        info["gen_lengths"]       = config.get("gen_lengths", [])

    return info


# ──────────────────────────────────────────────────────────────────────────────
# Thinking Mode 预留接口
# ──────────────────────────────────────────────────────────────────────────────

def disable_thinking_mode(generation_config: dict) -> dict:
    """关闭 thinking mode，避免思考 token 污染 gen_len 统计。

    LLaMA 3.2：无需处理，原样返回。
    Qwen3.5：generation_config.update({"enable_thinking": False})

    Args:
        generation_config: 生成配置字典

    Returns:
        处理后的生成配置字典
    """
    # LLaMA 3.2 不使用 thinking mode，直接原样返回
    # 未来支持 Qwen3.5 时在此处添加：
    # if model_family == "qwen3.5":
    #     generation_config.update({"enable_thinking": False})
    return generation_config
