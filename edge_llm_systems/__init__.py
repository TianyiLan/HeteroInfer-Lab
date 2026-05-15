"""EdgeLLM-Systems — Stage 1 v2.1（LLaMA 3.2）

各子模块按需显式导入，不在此处集中 re-export，避免循环依赖和跨分支兼容问题。

主要模块：
  edge_llm_systems.models    — 模型加载 / 下载 / 卸载
  edge_llm_systems.profiling — 单次推理性能测量（TTFT / TPOT / KV Cache）
  edge_llm_systems.runners   — 实验循环调度（warm-up / 基准测试 / CSV 写入）
  edge_llm_systems.quality   — 文本质量评估套件（MMLU-Pro / GSM8K CoT / …）
  edge_llm_systems.utils     — 通用工具（日志 / CSV / JSON / 环境检测）
  edge_llm_systems.kv_cache  — KV Cache 大小测量与理论估算
  edge_llm_systems.memory    — GPU 显存监控
  edge_llm_systems.metrics   — 通用指标计算
  edge_llm_systems.cuda_utils — CUDA 同步 / 显存清理工具
"""
