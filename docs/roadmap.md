# EdgeLLM-Systems 技术路线图

## 一、项目目标

EdgeLLM-Systems 是一个面向资源受限边缘环境的大模型推理系统研究项目，聚焦部署边界、性能瓶颈、软件优化与异构硬件加速。

核心问题：

> 在有限显存、有限带宽、低 batch 和低延迟约束下，如何让大模型稳定部署，并进一步提升推理效率？

整体技术路线：

> **Profiling → Bottleneck Analysis → Software Optimization → Heterogeneous Hardware Acceleration → FPGA Compiler & Dataflow Mapping**

---

## 二、测量框架：三分类指标体系

本项目采用与学术界（MLPerf Inference、MobileLLM、LLM-in-a-Flash）对齐的三分类测量框架，对应三大研究目标：

| 类别 | 研究目标 | 核心指标 |
|------|---------|---------|
| **Memory Footprint** | 可部署性 | `model_load_mem_mb`, `peak_mem_mb`, `kv_pkv_*_mb`, `kv_est_mb`, `kv_payload_ratio` |
| **Inference Efficiency** | 推理速度 | `ttft_ms`, `tpot_ms`, `tokens_s`, `total_latency_ms` |
| **Model Quality** | 精度保持 | accuracy, stderr（标准 NLP benchmark） |

三类独立测量、独立写 CSV、独立目录，每次实验按需勾选。

---

## 三、系统模块结构

### Module A：Measurement Layer（测量层）

负责系统基础性能测量与运行行为采集：三类指标的自动化测量、KV Cache 实测与理论验证、可复现实验配置。

核心问题：
> 系统当前的真实运行状态是什么？

### Module B：Analysis Layer（分析层）

负责系统瓶颈分析与性能解释：部署边界分析、内存带宽利用率、Roofline 分析、KV Cache 增长趋势。

核心问题：
> 为什么系统会在特定阶段出现容量压力、带宽压力或延迟瓶颈？

### Module C：Optimization Layer（优化层）

负责面向瓶颈的软件优化设计与实现：CUDA kernel 优化、量化、KV Cache 压缩、运行时调度。

核心问题：
> 哪些优化路径具有实际收益？

### Module D：Compiler & Mapping Layer（编译器与映射层）

负责将单点硬件加速经验提升为可复用方法：FPGA Dataflow Mapping、HLS 调度、MLIR 代码生成。

核心问题：
> 如何将单个 FPGA 加速案例推广为可复用的 dataflow mapping 与 compiler optimization 框架？

---

## 四、阶段推进路线

### Stage 1：Performance Characterization（性能表征）

**目标**：在未做任何优化的 FP16 原始模型上，完整采集三类指标，建立后续所有阶段的参照基准。

| 实验 | 内容 | 状态 |
|------|------|------|
| **exp001** | LLaMA-3.2-3B-Instruct，Text-only，L4 GPU，FP16 | ✅ 完成 |
| **exp002** | LLaMA-3.2-11B-Vision，Vision-Language，L4 GPU，FP16 | 🔲 规划中 |

exp001 完成内容：
- Memory Footprint：18 场景 × 3 次重复，prompt 64–2048，gen 32–128
- Inference Efficiency：18 场景 × 3 次重复，含均值汇总行
- Model Quality：5 项 text 基准（HellaSwag / WinoGrande / TruthfulQA MC1 / MMLU-Pro / GSM8K）

Stage 1 在 exp002（multimodal baseline）完成后关闭。

---

### Stage 2：Bottleneck Analysis（瓶颈分析）

**目标**：不做优化，只做诊断。基于 Stage 1 的三类指标数据，系统性定位制约边缘部署的主要瓶颈，为 Stage 3 / 4 的优化方向提供实验依据。

核心分析维度：
- 内存带宽利用率分析（Memory-bound Decode 特征）
- 部署边界分析（模型规模 × 设备内存 × 上下文长度）
- KV Cache 增长趋势与占比分析
- 质量-效率帕累托边界

核心问题：
> Decode 阶段的核心 bottleneck 首先体现为 capacity problem、bandwidth problem 还是 kernel implementation problem？

**输出**：明确瓶颈分类，建立 Stage 3 / 4 的优化优先级。

---

### Stage 3：Software Optimization（软件优化）

**目标**：在保持部署环境不变的前提下，通过软件手段提升效率、降低内存，并使用 Stage 1 baseline 计算 `retention_rate` 量化质量损失。

| 方向 | 代表技术 |
|------|---------|
| 模型量化 | INT8 / INT4、GPTQ、AWQ、SmoothQuant |
| 算子优化 | FlashAttention、算子融合、KV Cache 量化 |
| 推理框架 | vLLM、llama.cpp、MLC-LLM |

核心问题：
> 在不改变硬件的前提下，软件优化最多能解决多少问题？

**输出**：完成第一轮可验证的软件优化闭环，明确 GPU 软件优化的收益上限与残余瓶颈。

---

### Stage 4：Heterogeneous Hardware Acceleration（异构硬件加速）

Stage 4 拆分为两个相对独立的 hardware track：

#### Track 4A：FPGA-based Host-centric Edge Acceleration

**目标**：验证 FPGA 能否在 Host-centric Edge Platforms 上作为 GPU 的异构加速器，缓解 Decode 阶段的 memory-bound bottleneck。

典型平台：x86 / ARM 主机 + NVIDIA GPU + FPGA 加速卡（PCIe 互联）

核心任务：
- FPGA Decode Path 探索
- GPU + FPGA 协同执行分析
- HLS kernel 原型实现
- PCIe 数据搬移开销分析

核心问题：
> FPGA 是否能有效缓解 Decode 阶段的访存压力与延迟瓶颈？

#### Track 4B：NPU-based SoC-integrated Edge Inference

**目标**：分析 SoC-integrated Edge Platforms 上 CPU / GPU / NPU 的异构协同机制，评估 NPU 对边缘侧 LLM 部署能力的影响。

典型平台：Mobile SoC / Jetson / Orin，ARM CPU + GPU + NPU，统一内存架构

核心任务：
- CPU / GPU / NPU 算子分配
- NPU Tensor Shape 灵敏度分析
- Unified Memory 行为分析

核心问题：
> CPU / GPU / NPU 如何协同执行 LLM 推理，才能提升端侧部署能力和效率？

---

### Stage 5：FPGA Compiler & Dataflow Mapping Framework

**目标**：在 Track 4A FPGA 加速验证的基础上，从单个加速案例走向系统性可复用的方法论。

核心任务：
- Decode Kernel HLS 实现
- FPGA Dataflow Mapping
- MLIR-based 优化与 Hardware Code Generation
- Loop Transformation / Memory Hierarchy Mapping / Pipeline Balancing
- Design Space Exploration

核心问题：
> 如何系统性地将 decode kernel 映射到 FPGA，并自动或半自动生成高效 dataflow execution？

**输出**：完整的异构推理优化闭环与可复用 FPGA dataflow mapping 框架。

---

## 五、执行原则

1. **先测量，再优化**：在完成 profiling 与测量协议校准前，不进入复杂优化阶段。
2. **问题驱动，工具服务问题**：CUDA、FPGA、NPU、MLIR 均服务于具体系统问题，所有优化建立在真实 bottleneck 之上。
3. **区分平台假设**：Host-centric 与 SoC-integrated 是两类不同边缘平台，评价逻辑不混用。
4. **保持单阶段推进**：当前只推进一个主阶段，其他方向保留规划与接口。
5. **以实验结果驱动研究**：核心资产是数据、图表、分析结论与可复现实验流程。

---

## 六、近期目标

**当前：完成 Stage 1**

- [x] exp001 三类指标全量完成（text-only，L4，LLaMA-3.2-3B-Instruct）
- [ ] exp002 multimodal baseline（LLaMA-3.2-11B-Vision，L4）
- [ ] 确认 Stage 1 关闭条件，正式进入 Stage 2 瓶颈分析
