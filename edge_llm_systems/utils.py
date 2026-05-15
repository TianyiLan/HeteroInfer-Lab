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
