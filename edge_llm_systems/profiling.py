"""DEPRECATED — 本模块已拆分为 memory_profiler + efficiency_profiler。

v2.2 起，性能测量按指标三分类体系拆分：
  - edge_llm_systems.memory_profiler     : Memory Footprint
  - edge_llm_systems.efficiency_profiler : Inference Efficiency
  - edge_llm_systems.profiling_core      : 内部共享推理实现

本模块仅作向后兼容 shim，重新导出旧 API（measure_text_single /
measure_image_single）。新代码请直接 import 新模块。
"""

from __future__ import annotations

import warnings

# 重新导出底层测量函数，与旧版 API 完全兼容
from edge_llm_systems.profiling_core import (  # noqa: F401
    OOM_STAGES,
    measure_image_single,
    measure_text_single,
)

warnings.warn(
    "edge_llm_systems.profiling 已废弃，请改用 memory_profiler / efficiency_profiler。"
    "底层共享推理在 profiling_core，但通常不应直接调用。",
    DeprecationWarning,
    stacklevel=2,
)
