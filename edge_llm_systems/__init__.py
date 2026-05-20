"""EdgeLLM-Systems — Stage 1 v2.2（LLaMA 3.2）

═══════════════════════════════════════════════════════════════════════════════
指标三分类框架 (Three-category metric taxonomy)
═══════════════════════════════════════════════════════════════════════════════

对应论文的三大研究目标，详见 edge_llm_systems.categories：

  1. Memory Footprint     (资源开销) → 可部署性
     字段：model_load_mem_mb, peak_mem_mb, kv_pkv_*_mb, kv_est_mb, kv_payload_ratio
     输出：results/.../memory/ 目录
     模块：edge_llm_systems.memory_profiler

  2. Inference Efficiency (推理效率) → 加速
     字段：ttft_ms, tpot_ms, tokens_s, total_latency_ms, image_*/projector_ms, ...
     输出：results/.../efficiency/ 目录
     模块：edge_llm_systems.efficiency_profiler

  3. Model Quality        (模型质量) → 质量保持
     字段：accuracy, stderr, num_samples
     输出：results/.../quality/ 目录
     模块：edge_llm_systems.lm_eval_runner

每类独立测量、独立写 CSV、独立目录。Memory 与 Efficiency 在底层共享同一次
推理（profiling_core），CSV 输出按字段子集过滤后完全独立。

═══════════════════════════════════════════════════════════════════════════════
模态分类 (Modality)
═══════════════════════════════════════════════════════════════════════════════

顶层模态简化为 text / vision 二选一（取代旧的 input_mode 三值枚举）。
图片数、分辨率作为正交维度展开进 group_id，例如：
  vision_4img_336_prompt256_gen64

═══════════════════════════════════════════════════════════════════════════════
模块清单
═══════════════════════════════════════════════════════════════════════════════

公开 API：
  categories            — 三分类常量与字段映射
  memory_profiler       — Memory Footprint 测量
  efficiency_profiler   — Inference Efficiency 测量
  lm_eval_runner        — Model Quality 评估（封装 lm-evaluation-harness）
  runners               — 三分类编排（warmup + run_profiling_suite_*）
  models                — 模型加载 / 下载 / 卸载
  utils                 — 通用工具（日志 / CSV / JSON / 环境检测）

内部 / 工具：
  profiling_core        — 共享推理实现（被两个 profiler 复用，不直接调用）
  _aggregation          — 行构造 / 进度打印 / 组聚合
  kv_cache              — KV Cache 大小测量与理论估算
  memory                — GPU 显存监控
  metrics               — 通用指标计算
  cuda_utils            — CUDA 同步 / 显存清理工具

废弃 / 归档：
  profiling             — DEPRECATED：v2.2 拆为 memory_profiler + efficiency_profiler
  quality (top-level)   — 旧自研质量评估，已被 lm_eval_runner 替代
  legacy.quality_v1.2.2 — 自研质量评估的归档版本
"""

__version__ = "2.2.0"
