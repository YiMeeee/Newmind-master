# RNewMind

轻量级大语言模型，从零实现完整的 Transformer 架构及训练流程。
基于 MiniMind 重构升级，模块化 `model/core/` 子包，新增 Pro 系列大参数配置、NTK RoPE 长文本外推、统一训练引擎与推理/评估模块。

---

## 📋 目录

- [RNewMind](#rnewmind)
  - [📋 目录](#-目录)
  - [模型架构](#模型架构)
    - [Lite 系列（轻量，适合训练实验）](#lite-系列轻量适合训练实验)
    - [Pro 系列（大参数，适合高质量任务）](#pro-系列大参数适合高质量任务)
    - [核心技术栈](#核心技术栈)
  - [技术特性](#技术特性)
    - [架构特性](#架构特性)
    - [训练技术](#训练技术)
    - [推理能力](#推理能力)
  - [项目结构](#项目结构)
  - [环境要求](#环境要求)
    - [硬件要求](#硬件要求)
    - [软件要求](#软件要求)
  - [快速开始](#快速开始)
    - [1. 安装依赖](#1-安装依赖)
    - [2. 验证安装](#2-验证安装)
    - [3. 加载已有权重推理](#3-加载已有权重推理)
  - [训练流程（详解）](#训练流程详解)
    - [阶段 0：准备工作](#阶段-0准备工作)
    - [阶段 1：预训练（Pretrain）](#阶段-1预训练pretrain)
    - [阶段 2：监督微调（Full SFT）](#阶段-2监督微调full-sft)
    - [阶段 3：DPO 偏好对齐](#阶段-3dpo-偏好对齐)
    - [阶段 4：LoRA 微调（可选）](#阶段-4lora-微调可选)
    - [阶段 5：知识蒸馏（可选）](#阶段-5知识蒸馏可选)
    - [阶段 6：推理蒸馏 CoT（可选）](#阶段-6推理蒸馏-cot可选)
    - [显存不足解决方案](#显存不足解决方案)
  - [模型推理](#模型推理)
    - [命令行交互](#命令行交互)
    - [Python 代码调用](#python-代码调用)
    - [API 服务](#api-服务)
    - [Web 界面](#web-界面)
  - [评估](#评估)
    - [Perplexity](#perplexity)
    - [生成质量基准](#生成质量基准)
  - [向后兼容说明](#向后兼容说明)
  - [依赖安装](#依赖安装)

---

## 模型架构

RNewMind 采用 Decoder-Only Transformer 架构，提供 **Lite** 和 **Pro** 两个系列。

### Lite 系列（轻量，适合训练实验）

| 参数 | Small (26M) | Base (104M) | MoE (145M) |
|------|:-----------:|:-----------:|:----------:|
| `hidden_size` | 512 | 768 | 640 |
| `num_hidden_layers` | 8 | 16 | 8 |
| `num_attention_heads` | 8 | 12 | 10 |
| `num_key_value_heads` | 2 | 3 | 2 |
| `vocab_size` | 6400 | 6400 | 6400 |
| `max_position_embeddings` | 32768 | 32768 | 32768 |
| `use_moe` | ❌ | ❌ | ✅ |

### Pro 系列（大参数，适合高质量任务）

| 参数 | Pro-1B | Pro-3B |
|------|:------:|:------:|
| `hidden_size` | 1536 | 2560 |
| `num_hidden_layers` | 24 | 32 |
| `num_attention_heads` | 24 | 32 |
| `num_key_value_heads` | 6 | 8 |
| `intermediate_size` | 4096 | 8192 |
| `max_position_embeddings` | 8192 | 4096 |

### 核心技术栈

- **注意力机制**: GQA (Grouped Query Attention) + RoPE
- **归一化**: RMSNorm (Pre-Norm)
- **激活函数**: SwiGLU (Swish 门控)
- **位置编码**: RoPE 支持 32K 上下文；NTK Scaling 版本可按比例外推更长上下文
- **注意力优化**: Flash Attention (PyTorch 2.0+)
- **MoE**: 稀疏混合专家，含共享专家 + 辅助负载均衡损失

---

## 技术特性

### 架构特性

- ✅ **模块化 core 子包**：Attention / FeedForward / MoE / RoPE / Norm 各自独立，便于替换与扩展
- ✅ **GQA**：减少 KV Cache 显存，提升推理吞吐
- ✅ **RoPE + NTK Scaling**：`precompute_freqs_cis_ntk()` 按比例调整 theta，无需微调即可外推上下文
- ✅ **RMSNorm**：收敛更稳定
- ✅ **SwiGLU**：增强非线性表达
- ✅ **MoE**：稀疏激活扩大模型容量，计算成本可控
- ✅ **Pro 配置预设**：`model/pro/` 直接 `build_pro_1b()` / `build_pro_3b()` 即可实例化

### 训练技术

- ✅ **预训练 / SFT / LoRA / DPO / 知识蒸馏 / 推理蒸馏** 完整六阶段流程
- ✅ **TrainerEngine**（`trainer/engine.py`）：统一封装 forward / loss / backward / checkpoint 逻辑
- ✅ **梯度累积**：小显存下等效大 batch 训练
- ✅ **DDP 多卡并行**：`torchrun` 一键启动
- ✅ **AMP 混合精度**：bfloat16 / float16 自动切换
- ✅ **WandB 监控**：`--use_wandb` 开启实时曲线

### 推理能力

- ✅ **KV Cache**：`inference/kv_cache.py` 管理 past_key_values，加速自回归生成
- ✅ **Top-k / Top-p / Temperature 采样**：`inference/sampler.py`
- ✅ **流式生成**：`inference/generate.py` 支持 `stream=True` 逐 token 输出
- ✅ **OpenAI 兼容 API**：`scripts/serve_openai_api.py`

---

## 项目结构

```
RNewMind/
├── model/
│   ├── core/                       # 核心原子组件（可独立复用）
│   │   ├── __init__.py
│   │   ├── norm.py                 # RMSNorm
│   │   ├── rope.py                 # RoPE + NTK Scaling
│   │   ├── attention.py            # GQA + Flash Attention + KV Cache
│   │   ├── feedforward.py          # SwiGLU FFN
│   │   ├── moe.py                  # MoEGate + MOEFeedForward
│   │   └── embeddings.py           # rope 辅助函数汇总导出
│   ├── lite/
│   │   ├── __init__.py
│   │   └── rnewmind_base.py        # RNewMindConfig / Block / Model / ForCausalLM
│   ├── pro/
│   │   ├── __init__.py
│   │   ├── rnewmind_pro_1b.py      # Pro-1B 配置 + build_pro_1b()
│   │   └── rnewmind_pro_3b.py      # Pro-3B 配置 + build_pro_3b()
│   ├── config/
│   │   ├── __init__.py
│   │   ├── config_lite.py          # Lite 预设字典
│   │   └── config_pro.py           # Pro 预设字典
│   ├── model_minimind.py           # 向后兼容 shim（MiniMind* 别名）
│   └── model_lora.py               # LoRA 适配器
├── dataset/
│   └── lm_dataset.py               # Pretrain / SFT / DPO / RLAIF 数据集
├── trainer/
│   ├── engine.py                   # 统一 TrainerEngine（新增）
│   ├── train_pretrain.py           # 阶段 1：预训练
│   ├── train_full_sft.py           # 阶段 2：全量 SFT
│   ├── train_lora.py               # 阶段 4：LoRA 微调
│   ├── train_dpo.py                # 阶段 3：DPO 偏好对齐
│   ├── train_distillation.py       # 阶段 5：知识蒸馏
│   └── train_distill_reason.py     # 阶段 6：推理蒸馏（CoT）
├── inference/
│   ├── __init__.py
│   ├── generate.py                 # 自回归生成（含流式）
│   ├── kv_cache.py                 # KV Cache 管理
│   └── sampler.py                  # Top-k / Top-p / Temperature
├── evaluation/
│   ├── __init__.py
│   ├── perplexity.py               # Perplexity 计算
│   └── benchmark.py                # 生成质量批量测试
├── scripts/
│   ├── web_demo.py                 # Streamlit Web 界面
│   ├── serve_openai_api.py         # OpenAI 兼容 API 服务
│   ├── chat_openai_api.py          # API 调用示例
│   ├── convert_model.py            # torch ↔ transformers 格式转换
│   └── train_tokenizer.py          # 分词器训练
├── eval_model.py                   # 命令行推理 / 评估入口
└── requirements.txt
```

---

## 环境要求

### 硬件要求

| 任务 | 最低显存 | 推荐显存 |
|------|:--------:|:--------:|
| Lite Small (26M) 训练 | 6 GB | 12 GB |
| Lite Base (104M) 训练 | 12 GB | 24 GB |
| Pro-1B 训练 | 24 GB | 40 GB+ |
| Pro-3B 训练 | 48 GB | 80 GB+（多卡）|
| 推理（任意 Lite） | 2 GB | 4 GB |

### 软件要求

- Python 3.8+
- PyTorch 2.0+（推荐 2.2+，原生支持 Flash Attention）
- CUDA 11.8+
- Transformers 4.48.0+

---

## 快速开始

### 1. 安装依赖

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 2. 验证安装

```bash
python -c "from model.lite.rnewmind_base import RNewMindForCausalLM, RNewMindConfig; \
           RNewMindForCausalLM(RNewMindConfig()); print('OK')"
```

### 3. 加载已有权重推理

```bash
python eval_model.py --model_mode 1 --hidden_size 512 --num_hidden_layers 8
```

---

## 训练流程（详解）

> 完整训练分为六个阶段，**按顺序执行**，每个阶段的输出权重是下一阶段的起点。

```
[阶段1] 预训练
    ↓  out/pretrain_{hidden_size}.pth
[阶段2] 全量 SFT
    ↓  out/full_sft_{hidden_size}.pth
[阶段3] DPO 偏好对齐
    ↓  out/rlhf_{hidden_size}.pth
[阶段4] LoRA 微调（可选，领域适配）
    ↓  out/lora/lora_{name}_{hidden_size}.pth
[阶段5] 知识蒸馏（可选，大→小压缩）
    ↓  out/full_dist_{hidden_size}.pth
[阶段6] 推理蒸馏（可选，CoT 能力迁移）
    ↓  out/reason_{hidden_size}.pth
```

所有训练脚本均在 `trainer/` 目录下，**需要先 `cd trainer`** 再执行。

---

### 阶段 0：准备工作

**创建目录**

```bash
mkdir -p out dataset
```

**数据文件**

将对应数据文件放入 `dataset/` 目录：

| 文件名 | 用途 |
|--------|------|
| `pretrain_hq.jsonl` | 预训练语料（每行一条文本） |
| `sft_512.jsonl` | SFT 对话数据（512 token 长度） |
| `dpo.jsonl` | DPO 偏好对（含 chosen / rejected） |
| `lora_medical.jsonl` | LoRA 领域数据（可替换为其他领域） |
| `sft_xxx.jsonl` | 知识蒸馏训练数据 |
| `r1_mix_1024.jsonl` | 推理蒸馏数据（含 `<think>` 标签） |

**模型规格选择建议**

| 目标 | `--hidden_size` | `--num_hidden_layers` | 参数量 | 适合场景 |
|------|-----------------|-----------------------|--------|----------|
| 快速验证流程 | 512 | 8 | ~26M | 本地 GPU 实验 |
| 均衡性能 | 768 | 16 | ~104M | 生产可用 |
| 稀疏扩展 | 640 | 8 | ~145M（激活~45M）| MoE 实验 |

---

### 阶段 1：预训练（Pretrain）

从随机初始化开始，在大规模语料上学习语言建模能力（预测下一个 token）。这是所有后续阶段的基础。

```bash
cd trainer

python train_pretrain.py \
    --epochs 1 \
    --batch_size 32 \
    --accumulation_steps 8 \
    --learning_rate 5e-4 \
    --hidden_size 512 \
    --num_hidden_layers 8 \
    --max_seq_len 512 \
    --data_path "../dataset/pretrain_hq.jsonl" \
    --out_dir "../out" \
    --dtype bfloat16 \
    --log_interval 50 \
    --save_interval 500
```

**多卡 DDP 训练**

```bash
torchrun --nproc_per_node=2 train_pretrain.py \
    --ddp \
    --epochs 1 \
    --batch_size 16 \
    --accumulation_steps 4 \
    --hidden_size 768 \
    --num_hidden_layers 16 \
    --learning_rate 5e-4
```

**启用 WandB 监控**

```bash
python train_pretrain.py --use_wandb --wandb_project "RNewMind-Pretrain" [其他参数]
```

**参数详解**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--epochs` | 1 | 数据量充足时 1 轮已足够；可训练 2～3 轮进一步收敛 |
| `--batch_size` | 32 | 单卡 batch 大小；显存不足时减半并翻倍 `accumulation_steps` |
| `--accumulation_steps` | 8 | 梯度累积步数；等效 batch = batch_size × accumulation_steps |
| `--learning_rate` | 5e-4 | 预训练学习率，采用余弦衰减至 lr/10 |
| `--max_seq_len` | 512 | 单条训练序列长度；越长显存需求越高（显存 ∝ seq_len²） |
| `--grad_clip` | 1.0 | 梯度裁剪阈值，防止梯度爆炸 |
| `--dtype` | bfloat16 | 混合精度类型；A100/H100 用 bfloat16，其余用 float16 |
| `--hidden_size` | 512 | 隐藏层维度，决定模型宽度 |
| `--num_hidden_layers` | 8 | Transformer 层数，决定模型深度 |
| `--use_moe` | False | 是否启用 MoE 架构 |

**输出**: `out/pretrain_512.pth`（或 `pretrain_512_moe.pth`）

---

### 阶段 2：监督微调（Full SFT）

加载预训练权重，在**指令—回复**对数据上全参数微调，赋予模型对话与指令遵循能力。

```bash
python train_full_sft.py \
    --epochs 3 \
    --batch_size 16 \
    --accumulation_steps 1 \
    --learning_rate 5e-7 \
    --hidden_size 512 \
    --num_hidden_layers 8 \
    --max_seq_len 512 \
    --data_path "../dataset/sft_512.jsonl" \
    --out_dir "../out" \
    --dtype bfloat16 \
    --save_interval 200
```

**参数详解**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--epochs` | 2 | SFT 建议 2～5 轮；过多轮次会导致过拟合 |
| `--learning_rate` | 5e-7 | **必须远低于预训练**（推荐范围：1e-7 ～ 5e-6），防止灾难性遗忘 |
| `--max_seq_len` | 512 | 与 SFT 数据集对话长度匹配 |

> ⚠️ **关键注意**：SFT 学习率若设置过大（如 1e-4），模型会迅速遗忘预训练的语言知识，变成乱码输出。

**输出**: `out/full_sft_512.pth`

---

### 阶段 3：DPO 偏好对齐

加载 SFT 权重，用**正样本（chosen）和负样本（rejected）**对的对比学习让模型偏向人类偏好，是 RLHF 的稳定替代方案。

```bash
python train_dpo.py \
    --epochs 2 \
    --batch_size 8 \
    --learning_rate 1e-8 \
    --hidden_size 512 \
    --num_hidden_layers 8 \
    --max_seq_len 1024 \
    --data_path "../dataset/dpo.jsonl" \
    --out_dir "../out"
```

**参数详解**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--learning_rate` | 1e-8 | **极小学习率**，建议 ≤ 1e-8；过大会破坏 SFT 对话能力 |
| `--max_seq_len` | 1024 | DPO 数据通常含更长的正负样本 |
| `--batch_size` | 8 | DPO 每条数据含 chosen + rejected 两路，实际显存消耗是普通训练的 2 倍 |
| `--epochs` | 2 | DPO 不宜训练过多轮，容易过拟合偏好数据而失去通用性 |

> ⚠️ **关键注意**：DPO 阶段内部会同时维护一个冻结的参考模型（reference model），因此显存消耗约为 SFT 的 2 倍，请相应降低 `--batch_size`。

**输出**: `out/rlhf_512.pth`

---

### 阶段 4：LoRA 微调（可选）

仅训练少量低秩适配参数（约占总参数 0.5%～2%），适合在特定领域（医疗、法律、金融等）快速适配，**不破坏通用能力**。

```bash
python train_lora.py \
    --epochs 10 \
    --batch_size 32 \
    --learning_rate 1e-4 \
    --hidden_size 512 \
    --num_hidden_layers 8 \
    --max_seq_len 512 \
    --data_path "../dataset/lora_medical.jsonl" \
    --lora_name "lora_medical" \
    --out_dir "../out"
```

**参数详解**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--lora_name` | lora_medical | LoRA 权重文件名前缀，建议与领域名称对应 |
| `--learning_rate` | 1e-4 | LoRA 学习率可高于全量 SFT，因只更新少量参数 |
| `--epochs` | 10 | 领域数据量少时可适当多轮 |

**推理时加载 LoRA**

```bash
python eval_model.py --model_mode 1 --lora_name lora_medical
```

**输出**: `out/lora/lora_medical_512.pth`（仅保存 LoRA 增量权重，体积极小）

---

### 阶段 5：知识蒸馏（可选）

用**大模型（教师）**的 logits 分布指导**小模型（学生）**训练，在保持性能的同时压缩体积。损失 = α × CE loss + (1-α) × KL 蒸馏 loss。

```bash
python train_distillation.py \
    --epochs 6 \
    --batch_size 32 \
    --learning_rate 5e-6 \
    --max_seq_len 512 \
    --data_path "../dataset/sft_xxx.jsonl" \
    --out_dir "../out"
```

> 教师/学生模型的规格在脚本内的 `lm_config_student` / `lm_config_teacher` 中直接修改：
> - 默认学生：`hidden_size=512, num_hidden_layers=8`（~26M）
> - 默认教师：`hidden_size=768, num_hidden_layers=16`（~104M）

**输出**: `out/full_dist_512.pth`

---

### 阶段 6：推理蒸馏 CoT（可选）

从包含思维链标签（`<think>...</think><answer>...</answer>`）的数据中学习推理能力，特殊 token 位置使用 10× 惩罚权重加强学习。

```bash
python train_distill_reason.py \
    --epochs 1 \
    --batch_size 8 \
    --learning_rate 1e-6 \
    --hidden_size 512 \
    --num_hidden_layers 8 \
    --max_seq_len 1024 \
    --data_path "../dataset/r1_mix_1024.jsonl" \
    --out_dir "../out"
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--learning_rate` | 1e-6 | 推理蒸馏学习率要小，防止破坏已有对话能力 |
| `--max_seq_len` | 1024 | 思维链数据通常更长 |

**输出**: `out/reason_512.pth`

---

### 显存不足解决方案

| 问题 | 解决方法 |
|------|----------|
| 单步 OOM | 降低 `--batch_size`，按比例升高 `--accumulation_steps` 保持等效 batch 不变 |
| 序列太长 OOM | 降低 `--max_seq_len`（512 → 256） |
| 模型太大 OOM | 选更小的 `--hidden_size` 和 `--num_hidden_layers` |
| 显存仍不足 | 多卡 DDP：`torchrun --nproc_per_node=N` + `--ddp` |
| DPO 阶段 OOM | DPO 有 2 个模型，`batch_size` 应为 SFT 阶段的一半 |

**梯度累积等效示例**

```
目标等效 batch = 256
显存只能放 batch_size = 8
→ accumulation_steps = 256 / 8 = 32
```

---

## 模型推理

### 命令行交互

```bash
# SFT 对话模型
python eval_model.py --model_mode 1

# RLHF/DPO 对齐模型
python eval_model.py --model_mode 2

# 推理/CoT 模型
python eval_model.py --model_mode 3

# LoRA 领域适配
python eval_model.py --model_mode 1 --lora_name lora_medical
```

**推理参数说明**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model_mode` | 1 | 0: 预训练续写，1: SFT，2: RLHF，3: 推理，4: RLAIF |
| `--hidden_size` | 512 | 与训练时一致 |
| `--num_hidden_layers` | 8 | 与训练时一致 |
| `--temperature` | 0.85 | 越高输出越随机，越低越确定 |
| `--top_p` | 0.85 | 核采样阈值 |
| `--max_seq_len` | 8192 | 最大生成长度 |
| `--history_cnt` | 0 | 携带历史对话轮数（须为偶数） |
| `--lora_name` | None | 指定 LoRA 权重名称 |
| `--load` | 0 | 0: PyTorch `.pth`，1: Transformers 格式 |

### Python 代码调用

```python
import torch
from model.lite.rnewmind_base import RNewMindConfig, RNewMindForCausalLM
from transformers import AutoTokenizer
from inference.generate import generate

config = RNewMindConfig(hidden_size=512, num_hidden_layers=8)
model = RNewMindForCausalLM(config)
model.load_state_dict(torch.load("out/full_sft_512.pth", map_location="cpu"))
model.eval()

tokenizer = AutoTokenizer.from_pretrained("./model/")
input_ids = tokenizer("请介绍一下自己", return_tensors="pt").input_ids

# 普通生成
output = generate(model, input_ids, max_new_tokens=200, temperature=0.7, top_p=0.9)
print(tokenizer.decode(output[0].tolist(), skip_special_tokens=True))

# 流式生成
for token in generate(model, input_ids, max_new_tokens=200, stream=True):
    print(tokenizer.decode(token[0].tolist(), skip_special_tokens=True), end="", flush=True)
```

### API 服务

```bash
python scripts/serve_openai_api.py --hidden_size 512 --num_hidden_layers 8
```

```python
from openai import OpenAI
client = OpenAI(api_key="none", base_url="http://localhost:8000/v1")
response = client.chat.completions.create(
    model="rnewmind",
    messages=[{"role": "user", "content": "你好，请介绍一下自己"}]
)
print(response.choices[0].message.content)
```

### Web 界面

```bash
streamlit run scripts/web_demo.py
```

---

## 评估

### Perplexity

```python
from evaluation.perplexity import calc_perplexity
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("./model/")
ppl = calc_perplexity(model, tokenizer, ["这是一段测试文本"], device="cuda")
print(f"Perplexity: {ppl:.2f}")
```

### 生成质量基准

```python
from evaluation.benchmark import run_benchmark

results = run_benchmark(
    model, tokenizer,
    prompts=["解释一下量子纠缠", "写一首关于春天的诗"],
    device="cuda", max_new_tokens=200
)
for r in results:
    print(f"[Q] {r['prompt']}\n[A] {r['generated']}\n")
```

---

## 向后兼容说明

原 MiniMind 代码**无需任何修改**即可继续使用，旧类名通过 `model/model_minimind.py` shim 重定向到新实现：

```python
# 旧写法（仍然有效）
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM

# 新写法（推荐）
from model.lite.rnewmind_base import RNewMindConfig, RNewMindForCausalLM
```

| 旧类名 | 新类名 |
|--------|--------|
| `MiniMindConfig` | `RNewMindConfig` |
| `MiniMindForCausalLM` | `RNewMindForCausalLM` |
| `MiniMindModel` | `RNewMindModel` |
| `MiniMindBlock` | `RNewMindBlock` |
| `model_type = "minimind"` | `model_type = "rnewmind"` |

---

## 依赖安装

```bash
# PyTorch（CUDA 11.8）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 其余依赖（清华镜像加速）
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

核心依赖版本：

```
torch>=2.0.0
transformers>=4.48.0
datasets>=2.21.0
trl>=0.13.0
peft>=0.7.1
tiktoken>=0.5.1
streamlit>=1.30.0
fastapi
uvicorn
```

---

*最后更新：2026 年 3 月 | RNewMind v2.0*
