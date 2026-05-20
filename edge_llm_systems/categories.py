"""三大类指标分类 (Three-category metric taxonomy)

按学术惯例（MLPerf Inference / MobileLLM / LLM-in-a-Flash 等边缘部署论文）
将所有指标分为 3 类，直接对应 EdgeLLM 部署优化的三大研究目标：

  Memory Footprint     (资源开销) → Deployability    可部署性
  Inference Efficiency (推理效率) → Speed            加速
  Model Quality        (模型质量) → Accuracy         质量保持

每一类独立测量、独立写 CSV、独立目录，互不耦合（实现上 Memory 与
Efficiency 共享同一次推理以避免浪费，但 CSV 输出完全独立）。
"""

from __future__ import annotations

__version__ = "1.0.0"

# ──────────────────────────────────────────────────────────────────────────────
# 类别常量（用于目录名、模块名、CONFIG key 等）
# ──────────────────────────────────────────────────────────────────────────────

CATEGORY_MEMORY:     str = "memory"
CATEGORY_EFFICIENCY: str = "efficiency"
CATEGORY_QUALITY:    str = "quality"

ALL_CATEGORIES: tuple[str, ...] = (
    CATEGORY_MEMORY,
    CATEGORY_EFFICIENCY,
    CATEGORY_QUALITY,
)

# ──────────────────────────────────────────────────────────────────────────────
# 字段归属
#
# Memory 与 Efficiency CSV 都需要的字段（meta + status），单独抽出共享。
# ──────────────────────────────────────────────────────────────────────────────

SHARED_META_FIELDS: list[str] = [
    "timestamp",
    "run_id",
    "group_id",
    "model_id",
    "model_hash",
    "modality",                  # text | vision  （取代原 input_mode）
    "prompt_len",
    "gen_len",
    "image_count",               # vision 模式下非 N/A
    "image_resolution",          # vision 模式下非 N/A
    "run_index",                 # 0 = 聚合行，1+ = 单次运行
]

SHARED_STATUS_FIELDS: list[str] = [
    "status",                    # success / oom / error
    "oom_stage",                 # prefill / decode / image_encode / none
    "message_zh",
    "output_nonempty",
    "refusal_detected",
]

# ──────────────────────────────────────────────────────────────────────────────
# Memory Footprint 专属字段
# ──────────────────────────────────────────────────────────────────────────────

MEMORY_FIELDS: list[str] = [
    "model_load_mem_mb",         # 模型权重加载后显存占用
    "peak_mem_mb",               # 单次推理峰值显存
    "kv_pkv_prefill_mb",         # prefill 结束时 KV cache 大小
    "kv_pkv_final_mb",           # 生成完所有 token 后 KV cache 大小
    "kv_est_mb",                 # 理论估算值
    "kv_payload_ratio",          # KV cache / total peak memory
]

# ──────────────────────────────────────────────────────────────────────────────
# Inference Efficiency 专属字段
# ──────────────────────────────────────────────────────────────────────────────

EFFICIENCY_FIELDS: list[str] = [
    # 核心延迟 / 吞吐
    "ttft_ms",                   # Time To First Token
    "tpot_ms",                   # Time Per Output Token (decode 平均)
    "total_latency_ms",          # 端到端
    "tokens_s",                  # tokens / second

    # Vision 阶段分解（exp002 用，exp001 留 N/A）
    "image_preprocess_ms",
    "vision_encode_ms",
    "projector_ms",
    "text_prefill_ms",

    # Token 计数（与吞吐相关）
    "actual_prompt_len",
    "actual_gen_len",
    "image_token_count",
    "total_input_tokens",

    # 生成行为
    "finish_reason",

    # 输出文本（仅 raw CSV 保留用于检视，summary 不写）
    "output_text",
    "output_length",
]

# ──────────────────────────────────────────────────────────────────────────────
# 派生：拼装最终 CSV 字段顺序
# ──────────────────────────────────────────────────────────────────────────────

def memory_fieldnames() -> list[str]:
    """Memory CSV 完整字段列表（meta + memory + status）。"""
    return SHARED_META_FIELDS + MEMORY_FIELDS + SHARED_STATUS_FIELDS


def efficiency_fieldnames(include_output_text: bool = True) -> list[str]:
    """Efficiency CSV 完整字段列表。

    Args:
        include_output_text: 是否保留 output_text / output_length。
            raw CSV 通常需要（便于人工检视生成质量），summary CSV 不需要。
    """
    fields = SHARED_META_FIELDS + EFFICIENCY_FIELDS + SHARED_STATUS_FIELDS
    if not include_output_text:
        fields = [f for f in fields if f not in ("output_text", "output_length")]
    return fields


def all_telemetry_fieldnames() -> list[str]:
    """完整 telemetry 字段列表（供 profiling_core 内部使用）。

    包含 memory + efficiency 全部字段，不去重共享字段。
    """
    seen: set[str] = set()
    result: list[str] = []
    for f in SHARED_META_FIELDS + MEMORY_FIELDS + EFFICIENCY_FIELDS + SHARED_STATUS_FIELDS:
        if f not in seen:
            seen.add(f)
            result.append(f)
    return result
