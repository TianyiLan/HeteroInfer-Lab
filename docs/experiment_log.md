# Experiment Log

## Stage 1: Performance Characterization

### 阶段目标

建立 FP16 原始模型的三类指标（Memory Footprint / Inference Efficiency / Model Quality）完整 baseline，作为后续所有优化实验的参照基准。

### 当前状态

| 实验 | 模型 | 平台 | 状态 |
|------|------|------|------|
| **exp001** | LLaMA-3.2-3B-Instruct | Google Colab L4, FP16, batch=1 | ✅ 完成 |
| **exp002** | LLaMA-3.2-11B-Vision | Google Colab L4, FP16, batch=1 | 🔲 规划中 |

---

## exp001：LLaMA-3.2-3B-Instruct，Text-only，L4 GPU

### 实验配置

| 项目 | 值 |
|------|---|
| 模型 | `meta-llama/Llama-3.2-3B-Instruct` |
| 平台 | Google Colab，NVIDIA L4（22 GB HBM） |
| 精度 | FP16 |
| Batch size | 1 |
| 框架版本 | edge_llm_systems v2.2 |
| Notebook | `notebooks/Stage 1/exp001/exp001_llama32_stage1_v2_2_Colab.ipynb` |

**Profiling 矩阵**（Memory + Efficiency 共用同一次推理）：
- Prompt lengths：64 / 128 / 256 / 512 / 1024 / 2048 tokens
- Generation lengths：32 / 64 / 128 tokens
- 每组重复：3 次（取均值行写入 summary CSV）

**Quality 评测工具**：EleutherAI lm-evaluation-harness，协议与 Open LLM Leaderboard 一致

### 结果文件

```text
results/exp001/L4/Llama-3.2-3B-Instruct/
├── memory/
│   ├── memory_raw_20260518_085525.csv      # 18 场景 × 3 次原始行
│   └── memory_summary_20260518_085525.csv  # 18 场景均值行
├── efficiency/
│   ├── efficiency_raw_20260518_111702.csv
│   └── efficiency_summary_20260518_111702.csv
├── quality/
│   ├── quality_summary_run_20260518_112726.csv  # hellaswag, winogrande
│   └── quality_summary_run_20260518_135629.csv  # truthfulqa_mc1, mmlu_pro, gsm8k
└── temp/
    ├── quality_manifest.json
    └── lm_eval_run_*/                           # 各 benchmark 原始 JSON
```

---

### ① Memory Footprint 结果

**模型权重验证**：`model_load_mem_mb = 6,127.835 MB`

理论值：$3.21 \times 10^9\ \text{params} \times 2\ \text{bytes} = 6,420\ \text{MB}$（含 embedding buffer 等实际与理论完全吻合）✅

**KV Cache 随序列长度的增长**（均值行）：

| prompt_len | gen_len | peak_mem_mb | kv_pkv_final_mb | kv_payload_ratio |
|-----------|---------|-------------|-----------------|-----------------|
| 64 | 32 | 6,163 | 10.5 | 0.17% |
| 256 | 128 | 6,242 | 42.0 | 0.67% |
| 512 | 128 | 6,344 | 70.0 | 1.10% |
| 1024 | 128 | 6,517 | 126.0 | 1.93% |
| 2048 | 128 | 6,881 | 238.0 | 3.46% |

**核心结论**：
- KV Cache 占峰值显存比例极低（< 3.5%），3B 模型的主要内存压力来自**权重本身**
- `kv_est_mb == kv_pkv_final_mb`（完全一致），理论公式准确，测量方法可靠
- 即使 prompt=2048 gen=128，总显存 6.88 GB，在 L4（22 GB）下充裕

---

### ② Inference Efficiency 结果

**TTFT 随 prompt_len 的变化**（gen=64 均值行）：

| prompt_len | ttft_ms | tpot_ms | tokens_s |
|-----------|---------|---------|---------|
| 64 | 38.6 | 33.7 | 29.7 |
| 128 | 39.3 | 33.7 | 29.7 |
| 256 | 43.8 | 34.1 | 29.4 |
| 512 | 70.3 | 33.9 | 29.5 |
| 1024 | 154.9 | 33.7 | 29.6 |
| 2048 | 293.5 | 34.1 | 29.3 |

**核心结论**：
- **TPOT 高度稳定**：33.7–34.6 ms，与 prompt_len / gen_len 无关（Memory-bound Decode 特征）
- **TTFT 线性增长**：prompt 32× → TTFT 约 7.6×（斜率 ≈ 0.13 ms/token）
- **内存带宽利用率估算**：权重 6 GB × 29 tokens/s ≈ 174 GB/s，L4 HBM 峰值 300 GB/s → 利用率约 **58%**（batch=1 decode 的理论区间，正常）
- `finish_reason` 全为 `max_tokens`，模型未提前结束，测试矩阵完整

---

### ③ Model Quality 结果

所有 benchmark 使用 EleutherAI lm-evaluation-harness，协议与 Open LLM Leaderboard v1 完全对齐。

| Benchmark | Accuracy | ±stderr | n | Few-shot | 指标 | 社区同级参考 | 可接受？ |
|-----------|----------|---------|---|---------|------|------------|---------|
| **HellaSwag** | 67.2% | ±2.1% | 500 | 10-shot | acc_norm | ~70%（3B base） | ✅ |
| **WinoGrande** | 73.2% | ±2.0% | 500 | 5-shot | acc | ~74%（3B base） | ✅ |
| **TruthfulQA MC1** | 33.54% | ±1.65% | 817（全集） | 0-shot | acc | ~30–35%（3B） | ✅ |
| **MMLU-Pro** | 33.33% | ±2.1% | 504（36/科） | 5-shot CoT | exact_match,custom-extract | ~35–40%（3B） | ✅ |
| **GSM8K** | 67.4% | ±2.1% | 500 | 8-shot CoT | exact_match,strict-match | ~65–75%（3B） | ✅ |

GSM8K 参考数据：flexible-extract 为 72.6%（JSON 原始数据），strict-match 67.4% 为正式记录值。

**MMLU-Pro 各子学科细分**（36 题/科，误差 ±7–8%）：

| 学科 | 准确率 | | 学科 | 准确率 |
|------|--------|-|------|--------|
| Math | 47.2% | | Physics | 30.6% |
| Biology | 47.2% | | Health | 30.6% |
| Economics | 41.7% | | Business | 27.8% |
| Psychology | 38.9% | | Law | 27.8% |
| Other | 36.1% | | Philosophy | 27.8% |
| Computer Sci | 33.3% | | History | 25.0% |
| Chemistry | 30.6% | | Engineering | 22.2% |

**核心结论**：LLaMA-3.2-3B-Instruct 数学/语言能力均衡，各项指标在 3B Instruct 模型预期区间内，全部可接受。五项结果构成后续优化实验的 **FP16 Quality Baseline**。

---

### 技术问题记录

| 问题 | 原因 | 修复 |
|------|------|------|
| MMLU-Pro metric_key 写为 `acc,none` | lm-eval mmlu_pro 任务实际用生成式评测 | 改为 `exact_match,custom-extract`（v2.2 已修复） |
| GSM8K stderr 写入 CSV 为 67.4（应为 2.1） | 旧 stderr 提取逻辑仅处理 `,none` 后缀，对 `strict-match` 失效 | 通用化 stderr key 推导逻辑（v2.2 已修复） |

---

## Historical Experiments（已归档）

以下实验基于旧版框架（单文件 `profiling.py`，二分类 perf/qual），已归档，不作为当前研究的 baseline。

| 实验 | 模型 | 平台 | 说明 |
|------|------|------|------|
| Experiment 001A / PKV Modular | Gemma 2 2B IT | Tesla T4 | Stage 1A：PKV 测量协议校准 |
| Experiment 001B v2.1 | Gemma 2 2B/9B IT | Tesla T4 | Stage 1B：模型规模压力测试，9B OOM |

Gemma/T4 实验的核心结论（供参考）：
- Gemma 2 2B IT FP16 在 T4 可稳定运行，peak memory 约 5.1–7.2 GB，KV payload 最高约 221 MB
- Gemma 2 9B IT FP16 在 T4 模型加载阶段 CUDA OOM，为 T4 FP16 部署边界直接证据
- `past_key_values` payload 统计方法经校准，与理论 KV cache 公式高度吻合

---

## Deferred Items

- LLaMA-3.2-1B-Instruct baseline（待定，3B 结果已满足当前研究需求）
- exp002 multimodal baseline（LLaMA-3.2-11B-Vision，待 exp001 commit 后启动）
- INT8 / INT4 量化对比（Stage 3）
- Gemma 4 T4 probing（历史遗留，优先级低）
