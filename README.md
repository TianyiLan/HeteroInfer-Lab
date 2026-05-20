# EdgeLLM-Systems

**边缘大模型推理系统**

A research-oriented system project for memory-constrained edge LLM inference, profiling, optimization, and heterogeneous acceleration.

EdgeLLM-Systems 是一个面向资源受限边缘环境的大模型推理系统研究项目，聚焦部署边界、性能瓶颈、软件优化与异构硬件加速。

---

## Research Scope

项目关注 LLM/VLM 在两类边缘平台上的真实系统行为：

1. **Host-centric Edge Platforms（主机式边缘平台）**  
   x86 / ARM 主机 + 离散 GPU / FPGA 加速卡，包括个人 PC、小型工作站、边缘服务器。

2. **SoC-integrated Edge Platforms（片上集成式边缘平台）**  
   CPU / GPU / NPU 集成于同一片上系统，包括手机、机器人、Jetson / Orin 类嵌入式 AI 设备。

核心问题：

> 在有限显存、有限带宽、低 batch 和低延迟约束下，如何让大模型稳定部署，并进一步提升推理效率？

整体研究路径：

> **Profiling → Bottleneck Analysis → Software Optimization → Heterogeneous Hardware Acceleration → FPGA Compiler & Dataflow Mapping**

---

## Current Status

**Stage 1 — Performance Characterization（进行中）**

- **exp001（已完成）**：LLaMA-3.2-3B-Instruct，Text-only，Google Colab L4 GPU，FP16，batch=1
  - 完成三类指标全量采集：Memory Footprint、Inference Efficiency、Model Quality（5项基准）
  - 建立 FP16 text-only baseline，作为后续所有优化实验的参照

- **exp002（规划中）**：LLaMA-3.2-11B-Vision，Vision-Language，L4 GPU

---

## Three-Category Metric Taxonomy

本项目采用与学术界（MLPerf Inference、MobileLLM）对齐的三分类测量框架：

| 类别 | 研究目标 | 核心指标 | 输出目录 |
|------|---------|---------|---------|
| **Memory Footprint** | 可部署性 | `model_load_mem_mb`, `peak_mem_mb`, `kv_*_mb` | `results/.../memory/` |
| **Inference Efficiency** | 推理速度 | `ttft_ms`, `tpot_ms`, `tokens_s` | `results/.../efficiency/` |
| **Model Quality** | 精度保持 | accuracy, stderr（5项 text benchmark） | `results/.../quality/` |

三类独立测量、独立写 CSV、独立目录，可按需单独运行。

---

## exp001 Key Results（LLaMA-3.2-3B-Instruct, L4, FP16）

### Memory Footprint

- 模型权重：**6,128 MB**（与 FP16 理论值 3.21B × 2 bytes 完全吻合）
- KV Cache 占峰值显存比例：**< 3.5%**（prompt=2048 时最大）
- 最大峰值显存：**6,881 MB**（prompt=2048, gen=128），在 L4 22 GB 限制内充裕

### Inference Efficiency

- TPOT：**~34 ms/token**，高度稳定，与 prompt_len / gen_len 无关
- 吞吐：**~29.4 tokens/s**（batch=1 内存带宽极限，利用率约 58%）
- TTFT：**38 ms**（prompt=64）→ **294 ms**（prompt=2048），线性增长

### Model Quality（lm-evaluation-harness，与 Open LLM Leaderboard 协议一致）

| Benchmark | Accuracy | ±stderr | Protocol |
|-----------|----------|---------|---------|
| HellaSwag | 67.2% | ±2.1% | 10-shot, acc_norm |
| WinoGrande | 73.2% | ±2.0% | 5-shot, acc |
| TruthfulQA MC1 | 33.54% | ±1.65% | 0-shot, acc |
| MMLU-Pro | 33.33% | ±2.1% | 5-shot CoT, exact_match |
| GSM8K | 67.4% | ±2.1% | 8-shot CoT, strict-match |

---

## Project Structure

```text
EdgeLLM-Systems/
│
├── README.md
│
├── edge_llm_systems/          # 核心测量框架（Python package）
│   ├── __init__.py
│   ├── categories.py          # 三分类常量与字段映射
│   ├── profiling_core.py      # 共享内部推理实现（被两个 profiler 复用）
│   ├── memory_profiler.py     # Memory Footprint 测量公开 API
│   ├── efficiency_profiler.py # Inference Efficiency 测量公开 API
│   ├── runners.py             # 三分类编排（含单次推理优化）
│   ├── lm_eval_runner.py      # Model Quality 评估（封装 lm-evaluation-harness）
│   ├── models.py              # 模型加载 / 下载 / 卸载
│   ├── utils.py               # 通用工具（日志 / CSV / JSON / 环境检测）
│   ├── kv_cache.py            # KV Cache 大小测量与理论估算
│   ├── memory.py              # GPU 显存监控
│   ├── metrics.py             # 通用指标计算
│   ├── cuda_utils.py          # CUDA 同步 / 显存清理工具
│   ├── _aggregation.py        # 行构造 / 进度打印 / 组聚合（内部）
│   └── profiling.py           # DEPRECATED：已拆为 memory_profiler + efficiency_profiler
│
├── notebooks/
│   └── Stage 1/
│       └── exp001/
│           └── exp001_llama32_stage1_v2_2_Colab.ipynb
│
├── docs/
│   ├── roadmap.md             # 技术路线图
│   └── experiment_log.md      # 实验日志与阶段性结论
│
├── benchmarks/
│   └── README.md
│
├── results/                   # 实验数据（不纳入版本控制）
│
└── requirements.txt
```

---

## Research Questions

1. **部署边界**：权重、KV Cache、context length 和 batch size 如何共同决定模型能否运行？
2. **性能瓶颈**：Prefill 与 Decode 阶段的主要瓶颈来自计算、显存容量、访存带宽，还是 kernel 实现？
3. **KV Cache 影响**：KV Cache 在延迟、显存和带宽压力中分别占什么位置？
4. **软件优化收益**：量化、FlashAttention、PagedAttention 等方法能带来多少实际收益？
5. **FPGA 异构加速价值**：FPGA 能否有效缓解 Decode 阶段的 memory-bound bottleneck？
6. **NPU / SoC 异构推理价值**：CPU / GPU / NPU 如何协同，才能提升端侧部署能力？
