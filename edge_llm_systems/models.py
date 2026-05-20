"""模型加载、完整性检查与缓存管理。

支持 LLaMA 3.2 1B / 3B / 11B-Vision 三个变体，统一通过 MODEL_REGISTRY 映射
HuggingFace repo id，并在 Google Drive 缓存目录中持久化模型权重。
"""

from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import snapshot_download
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    MllamaForConditionalGeneration,
)

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# 模型注册表：model_key → HuggingFace repo id
# ──────────────────────────────────────────────────────────────────────────────
MODEL_REGISTRY: dict[str, str] = {
    "LLaMA-3.2-1B":         "meta-llama/Llama-3.2-1B-Instruct",
    "LLaMA-3.2-3B":         "meta-llama/Llama-3.2-3B-Instruct",
    "LLaMA-3.2-11B-Vision": "meta-llama/Llama-3.2-11B-Vision-Instruct",
}


def check_model_integrity(local_path: Path) -> bool:
    """检查本地模型目录的文件完整性。

    验证必需文件是否全部存在：
    - config.json：模型架构配置
    - tokenizer_config.json：分词器配置
    - *.safetensors：至少一个权重分片

    Args:
        local_path: 本地模型目录路径

    Returns:
        True 表示完整，False 表示不完整或不存在
    """
    if not local_path.is_dir():
        return False

    # 检查必需的配置文件
    required_files = ["config.json", "tokenizer_config.json"]
    for fname in required_files:
        if not (local_path / fname).exists():
            logger.info("缺少必需文件: %s", fname)
            return False

    # 检查至少存在一个 safetensors 权重文件
    safetensors_files = list(local_path.glob("*.safetensors"))
    if not safetensors_files:
        logger.info("未找到任何 .safetensors 权重文件")
        return False

    return True


def download_model_if_needed(
    model_key: str,
    cache_dir: str,
    hf_token: str,
) -> Path:
    """仅在本地文件不完整时从 HuggingFace 下载模型。

    下载完成后模型权重持久化于 cache_dir/{model_name}/，
    后续运行跳过下载直接加载。

    Args:
        model_key: MODEL_REGISTRY 中的键，如 "LLaMA-3.2-1B"
        cache_dir: 本地缓存根目录（如 Google Drive 路径）
        hf_token: HuggingFace 访问令牌（访问 gated 模型所需）

    Returns:
        本地模型目录的 Path 对象
    """
    if model_key not in MODEL_REGISTRY:
        raise ValueError(
            f"未知模型键: {model_key!r}，可用键: {list(MODEL_REGISTRY.keys())}"
        )

    repo_id = MODEL_REGISTRY[model_key]
    # 使用 HuggingFace repo 名称作为本地子目录名
    model_name = repo_id.split("/")[-1]
    local_path = Path(cache_dir) / model_name

    if check_model_integrity(local_path):
        print(f"[模型缓存] {model_key} 已存在，跳过下载: {local_path}")
        return local_path

    print(f"[模型下载] {model_key} → {repo_id}")
    print(f"  目标路径: {local_path}")
    local_path.mkdir(parents=True, exist_ok=True)

    # snapshot_download 会打印下载进度，并在文件已存在时跳过
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(local_path),
        token=hf_token,
        ignore_patterns=["*.bin"],  # 优先使用 safetensors，忽略旧格式
    )

    print(f"[模型下载] 完成: {local_path}")
    return local_path


def load_text_model(
    local_path: Path,
    hf_token: str,
) -> tuple[Any, Any]:
    """加载纯文本 causal LM（FP16，自动设备映射）。

    Args:
        local_path: 本地模型目录
        hf_token: HuggingFace 访问令牌

    Returns:
        (model, tokenizer) 元组
    """
    print(f"[模型加载] 加载文本模型: {local_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        str(local_path),
        token=hf_token,
    )

    model = AutoModelForCausalLM.from_pretrained(
        str(local_path),
        torch_dtype=torch.float16,  # FP16 减少显存占用
        device_map="auto",          # Accelerate 自动分配到可用 GPU
        token=hf_token,
    )
    model.eval()
    print(f"[模型加载] 文本模型加载完成，设备: {model.device}")
    return model, tokenizer


def load_vision_model(
    local_path: Path,
    hf_token: str,
) -> tuple[Any, Any]:
    """加载多模态视觉-语言模型（FP16，自动设备映射）。

    使用 AutoProcessor 统一处理文本和图像输入。
    LLaMA 3.2 Vision 使用 MllamaForConditionalGeneration 架构。

    Args:
        local_path: 本地模型目录
        hf_token: HuggingFace 访问令牌

    Returns:
        (model, processor) 元组
    """
    print(f"[模型加载] 加载视觉模型: {local_path}")

    processor = AutoProcessor.from_pretrained(
        str(local_path),
        token=hf_token,
    )

    # LLaMA 3.2 Vision 使用专用的条件生成模型类
    model = MllamaForConditionalGeneration.from_pretrained(
        str(local_path),
        torch_dtype=torch.float16,  # FP16 减少显存占用
        device_map="auto",          # Accelerate 自动分配到可用 GPU
        token=hf_token,
    )
    model.eval()
    print(f"[模型加载] 视觉模型加载完成，设备: {model.device}")
    return model, processor


def unload_model(*objects: Any) -> None:
    """卸载模型，释放 GPU 显存。

    依次删除所有传入对象的引用，触发垃圾回收，
    并清空 PyTorch CUDA 显存缓存。

    Args:
        *objects: 需要卸载的模型/分词器/processor 对象
    """
    # 删除调用方传入的对象引用
    for obj in objects:
        del obj

    # 强制运行垃圾收集器，释放可能循环引用的 Python 对象
    gc.collect()

    # 将 PyTorch 显存缓存块归还给 CUDA 驱动
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("[卸载] 模型已卸载，显存缓存已清理")


def get_model_load_mem_mb(device: str = "cuda") -> float:
    """记录模型加载后的基线显存占用（MB）。

    应在模型加载完成后立即调用，在任何推理之前，
    得到的值代表模型权重本身的显存占用基线。

    Args:
        device: 设备字符串，目前仅支持 "cuda"

    Returns:
        当前已分配的 CUDA 显存（MB）
    """
    if not torch.cuda.is_available():
        return 0.0

    # 同步确保所有 CUDA 操作完成，然后读取当前分配量
    torch.cuda.synchronize()
    mem_mb = torch.cuda.memory_allocated() / 1024**2
    print(f"[基线显存] 模型加载后显存占用: {mem_mb:.1f} MB")
    return mem_mb
