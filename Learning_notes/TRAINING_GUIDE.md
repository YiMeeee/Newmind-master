# RNewMind 完整训练指南

> 从零开始训练属于你的大语言模型

---

## 📋 目录

- [概述](#概述)
- [环境准备](#环境准备)
- [数据准备](#数据准备)
- [模型规格选择](#模型规格选择)
- [阶段 1：预训练](#阶段 1 预训练)
- [阶段 2：监督微调](#阶段 2 监督微调)
- [阶段 3：DPO 偏好对齐](#阶段 3dpo 偏好对齐)
- [阶段 4：LoRA 微调](#阶段 4lora 微调)
- [阶段 5：知识蒸馏](#阶段 5 知识蒸馏)
- [阶段 6：推理蒸馏](#阶段 6 推理蒸馏)
- [模型推理](#模型推理)
- [常见问题](#常见问题)

---

## 概述

### 训练流程总览

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  阶段 1      │────▶│  阶段 2      │────▶│  阶段 3      │
│  预训练      │     │  SFT        │     │  DPO        │
│  (语言基础)  │     │  (对话能力)  │     │  (偏好对齐)  │
└─────────────┘     └─────────────┘     └─────────────┘
                           │
                           ▼
                    ┌─────────────┐
                    │  阶段 4      │
                    │  LoRA       │
                    │  (领域适配)  │
                    └─────────────┘
```

### 训练时间与成本参考

| 模型规格 | 预训练 | SFT | DPO | 总成本 (3090 GPU) |
|---------|--------|-----|-----|------------------|
| 26M (Small) | ~30 分钟 | ~1 小时 | ~30 分钟 | ~3 RMB |
| 104M (Base) | ~2 小时 | ~4 小时 | ~2 小时 | ~15 RMB |
| 145M (MoE) | ~3 小时 | ~6 小时 | ~3 小时 | ~25 RMB |

---

## 环境准备

### 硬件要求

| 任务 | 最低显存 | 推荐显存 |
|------|:--------:|:--------:|
| 26M 模型训练 | 6 GB | 12 GB |
| 104M 模型训练 | 12 GB | 24 GB |
| MoE 模型训练 | 16 GB | 32 GB |
| 模型推理 | 2 GB | 4 GB |

### 软件安装

```bash
# 1. 安装 PyTorch (CUDA 11.8)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 2. 安装其他依赖
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 验证安装

```bash
python -c "from model.lite.rnewmind_base import RNewMindForCausalLM, RNewMindConfig; RNewMindForCausalLM(RNewMindConfig()); print('✅ 环境正常')"
```

### 创建输出目录

```bash
cd /Users/renjiapin/Documents/GitHub/Newmind-master
mkdir -p out dataset
```

---

## 数据准备

### 数据集概览
[[Dataset]]

| 文件名 | 用途 | 格式 | 必需 |
|--------|------|------|------|
| `pretrain_hq.jsonl` | 预训练 | `{"text": "..."}` | ✅ |
| `sft_512.jsonl` | SFT | `{"conversations": [...]}` | ✅ |
| `dpo.jsonl` | DPO | `{"chosen": [...], "rejected": [...]}` | ✅ |
| `lora_xxx.jsonl` | LoRA | `{"conversations": [...]}` | ⭕ |
| `r1_mix_1024.jsonl` | 推理蒸馏 | `{"conversations": [...]}` | ⭕ |

### 数据格式详解

#### 1. 预训练数据 (`pretrain_hq.jsonl`)

```json
{"text": "Python 是一种高级编程语言，由 Guido van Rossum 于 1989 年发明..."}
{"text": "量子力学是物理学的一个分支，主要描述微观粒子的行为..."}
{"text": "How to overcome procrastination? Here are some tips..."}
```

**要求**：
- 每行一条 JSON 数据
- 必须包含 `text` 字段
- 支持中英文混合

#### 2. SFT 数据 (`sft_512.jsonl`)

```json
{"conversations": [{"content": "你好"}, {"content": "你好！有什么可以帮助你的？"}]}
{"conversations": [{"content": "Python 是什么？"}, {"content": "Python 是一种高级编程语言..."}]}
{"conversations": [{"content": "Hello"}, {"content": "Hello! How can I help you?"}]}
```

**要求**：
- `conversations` 列表必须包含偶数个元素
- 奇数位置为用户消息，偶数位置为助手回复

#### 3. DPO 数据 (`dpo.jsonl`)

```json
{
  "chosen": [
    {"role": "user", "content": "如何学习编程？"},
    {"role": "assistant", "content": "学习编程建议：1. 选择一门语言 2. 系统学习基础 3. 多做项目练习..."}
  ],
  "rejected": [
    {"role": "user", "content": "如何学习编程？"},
    {"role": "assistant", "content": "随便学学就行了"}
  ]
}
```

**要求**：
- `chosen` 为优质回复，`rejected` 为较差回复
- 两者必须对应同一个用户问题

### 快速下载数据集

```bash
cd /Users/renjiapin/Documents/GitHub/Newmind-master/dataset

# 从 HuggingFace 数据集下载（推荐）
wget https://huggingface.co/datasets/jingyaogong/minimind_dataset/resolve/main/pretrain_hq.jsonl
wget https://huggingface.co/datasets/jingyaogong/minimind_dataset/resolve/main/sft_mini_512.jsonl -O sft_512.jsonl
wget https://huggingface.co/datasets/jingyaogong/minimind_dataset/resolve/main/dpo.jsonl
```

### 数据验证

```bash
# 检查数据行数
wc -l dataset/*.jsonl

# 预览数据
head -n 3 dataset/sft_512.jsonl
```

---

## 模型规格选择

### Lite 系列（推荐入门）

| 参数 | Small | Base | MoE |
|------|:-----:|:----:|:---:|
| `--hidden_size` | 512 | 768 | 640 |
| `--num_hidden_layers` | 8 | 16 | 8 |
| 参数量 | 26M | 104M | 145M |
| 适用场景 | 快速验证 | 生产可用 | MoE 实验 |

### Pro 系列（大模型）

| 参数 | Pro-1B | Pro-3B |
|------|:------:|:------:|
| `--hidden_size` | 1536 | 2560 |
| `--num_hidden_layers` | 24 | 32 |
| 参数量 | ~1B | ~3B |
| 适用场景 | 高质量任务 | 多卡训练 |

---

## 阶段 1：预训练

### 目标

从随机初始化开始，学习语言基础（预测下一个 token）。

### 训练命令

```bash
cd /Users/renjiapin/Documents/GitHub/Newmind-master/trainer

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

### 参数说明

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `--epochs` | 1 | 数据量大时 1 轮足够 |
| `--batch_size` | 32 | 单卡 batch 大小 |
| `--accumulation_steps` | 8 | 梯度累积，等效 batch = 32×8=256 |
| `--learning_rate` | 5e-4 | 预训练学习率，余弦衰减 |
| `--max_seq_len` | 512 | 序列长度，越长显存占用越高 |
| `--dtype` | bfloat16 | A100/H100 用 bfloat16，否则用 float16 |

### 多卡 DDP 训练

```bash
torchrun --nproc_per_node=2 train_pretrain.py \
    --ddp \
    --epochs 1 \
    --batch_size 16 \
    --hidden_size 768 \
    --num_hidden_layers 16 \
    --learning_rate 5e-4
```

### 启用 WandB 监控

```bash
python train_pretrain.py \
    --use_wandb \
    --wandb_project "RNewMind-Pretrain" \
    --wandb_run_name "small_26m" \
    [其他参数]
```

### 输出

- **文件**: `../out/pretrain_512.pth`
- **大小**: 约 100MB (26M 模型)
- **能力**: 能续写文本，但不会对话

### 预期损失曲线

```
Step 0:     loss=10.5
Step 500:   loss=3.2
Step 1000:  loss=2.1
Step 2000:  loss=1.8
```

---

## 阶段 2：监督微调 (SFT)

### 目标

在指令 - 回复数据上微调，赋予模型对话与指令遵循能力。

### 训练命令

```bash
cd /Users/renjiapin/Documents/GitHub/Newmind-master/trainer

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

### 参数说明

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `--epochs` | 3 | SFT 建议 2-5 轮 |
| `--learning_rate` | 5e-7 | **必须远低于预训练**，防止灾难性遗忘 |
| `--max_seq_len` | 512 | 与 SFT 数据集对话长度匹配 |

### ⚠️ 关键注意

SFT 学习率若设置过大（如 1e-4），模型会迅速遗忘预训练的语言知识，变成乱码输出。

### 输出

- **文件**: `../out/full_sft_512.pth`
- **能力**: 能进行对话、回答问题

### 测试模型

```bash
cd /Users/renjiapin/Documents/GitHub/Newmind-master

python eval_model.py \
    --model_mode 1 \
    --hidden_size 512 \
    --num_hidden_layers 8
```

---

## 阶段 3：DPO 偏好对齐

### 目标

用正负样本对的对比学习让模型偏向人类偏好，是 RLHF 的稳定替代方案。

### 训练命令

```bash
cd /Users/renjiapin/Documents/GitHub/Newmind-master/trainer

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

### 参数说明

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `--learning_rate` | 1e-8 | **极小学习率**，过大会破坏 SFT 对话能力 |
| `--batch_size` | 8 | DPO 有 chosen+rejected 两路，显存消耗是普通训练的 2 倍 |
| `--max_seq_len` | 1024 | DPO 数据通常含更长的正负样本 |

### ⚠️ 关键注意

DPO 阶段内部会同时维护一个冻结的参考模型，因此显存消耗约为 SFT 的 2 倍，请相应降低 `--batch_size`。

### 输出

- **文件**: `../out/rlhf_512.pth`
- **能力**: 输出质量更高，更符合人类偏好

---

## 阶段 4：LoRA 微调（可选）

### 目标

仅训练少量低秩适配参数（约占总参数 0.5%～2%），适合在特定领域快速适配。

### 训练命令

```bash
cd /Users/renjiapin/Documents/GitHub/Newmind-master/trainer

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

### 参数说明

| 参数 | 说明 |
|------|------|
| `--lora_name` | LoRA 权重文件名前缀，建议与领域名称对应 |
| `--learning_rate` | LoRA 学习率可高于全量 SFT，因只更新少量参数 |
| `--epochs` | 领域数据量少时可适当多轮 |

### 输出

- **文件**: `../out/lora/lora_medical_512.pth`
- **大小**: 仅几 MB（增量权重）

### 推理时加载 LoRA

```bash
python eval_model.py \
    --model_mode 1 \
    --lora_name lora_medical
```

---

## 阶段 5：知识蒸馏（可选）

### 目标

用大模型（教师）的 logits 分布指导小模型（学生）训练，在保持性能的同时压缩体积。

### 训练命令

```bash
cd /Users/renjiapin/Documents/GitHub/Newmind-master/trainer

python train_distillation.py \
    --epochs 6 \
    --batch_size 32 \
    --learning_rate 5e-6 \
    --max_seq_len 512 \
    --data_path "../dataset/sft_xxx.jsonl" \
    --out_dir "../out"
```

### 配置教师/学生模型

在 `train_distillation.py` 中修改：

```python
# 学生模型（默认 26M）
lm_config_student = RNewMindConfig(hidden_size=512, num_hidden_layers=8)

# 教师模型（默认 104M）
lm_config_teacher = RNewMindConfig(hidden_size=768, num_hidden_layers=16)
```

### 输出

- **文件**: `../out/full_dist_512.pth`
- **能力**: 小体积，接近教师模型的性能

---

## 阶段 6：推理蒸馏（可选）

### 目标

从包含思维链标签的数据中学习推理能力。

### 数据格式

```json
{
  "conversations": [
    {"content": "鸡兔同笼，共有 35 个头，94 只脚，问鸡兔各多少？"},
    {"content": "<think>设鸡有 x 只，兔有 y 只。根据题意：x + y = 35，2x + 4y = 94。解方程组...</think><answer>鸡有 23 只，兔有 12 只</answer>"}
  ]
}
```

### 训练命令

```bash
cd /Users/renjiapin/Documents/GitHub/Newmind-master/trainer

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

### 输出

- **文件**: `../out/reason_512.pth`
- **能力**: 能进行思维链推理

---

## 模型推理

### 命令行交互

```bash
cd /Users/renjiapin/Documents/GitHub/Newmind-master

# SFT 对话模型
python eval_model.py --model_mode 1 --hidden_size 512 --num_hidden_layers 8

# RLHF/DPO 对齐模型
python eval_model.py --model_mode 2

# 推理/CoT 模型
python eval_model.py --model_mode 3

# LoRA 领域适配
python eval_model.py --model_mode 1 --lora_name lora_medical
```

### 推理参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model_mode` | 1 | 0: 预训练续写，1: SFT，2: RLHF，3: 推理，4: RLAIF |
| `--temperature` | 0.85 | 越高输出越随机，越低越确定 |
| `--top_p` | 0.85 | 核采样阈值 |
| `--max_seq_len` | 8192 | 最大生成长度 |
| `--history_cnt` | 0 | 携带历史对话轮数（须为偶数） |

### Python 代码调用

```python
import torch
from model.lite.rnewmind_base import RNewMindConfig, RNewMindForCausalLM
from transformers import AutoTokenizer
from inference.generate import generate

# 加载模型
config = RNewMindConfig(hidden_size=512, num_hidden_layers=8)
model = RNewMindForCausalLM(config)
model.load_state_dict(torch.load("out/full_sft_512.pth", map_location="cpu"))
model.eval()

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained("./model/")

# 生成
input_ids = tokenizer("请介绍一下自己", return_tensors="pt").input_ids
output = generate(model, input_ids, max_new_tokens=200, temperature=0.7, top_p=0.9)
print(tokenizer.decode(output[0].tolist(), skip_special_tokens=True))
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

## 常见问题

### 显存不足怎么办？

| 问题 | 解决方法 |
|------|----------|
| 单步 OOM | 降低 `--batch_size`，按比例升高 `--accumulation_steps` |
| 序列太长 OOM | 降低 `--max_seq_len`（512 → 256） |
| 模型太大 OOM | 选更小的 `--hidden_size` 和 `--num_hidden_layers` |
| 显存仍不足 | 多卡 DDP：`torchrun --nproc_per_node=N` + `--ddp` |

### 梯度累积等效示例

```
目标等效 batch = 256
显存只能放 batch_size = 8
→ accumulation_steps = 256 / 8 = 32
```

### 训练损失不下降怎么办？

1. 检查数据格式是否正确
2. 降低学习率（特别是 SFT 阶段）
3. 增加 warmup 步数
4. 检查数据质量（是否有噪声）

### 模型输出乱码怎么办？

1. SFT 学习率过高 → 降低到 1e-7 ~ 5e-7
2. 预训练不充分 → 增加预训练轮数
3. 数据格式错误 → 检查 JSONL 格式

### 支持中英文双语吗？

✅ 支持。当前 Tokenizer 基于 BPE，天然支持多语言。只需在训练数据中混合中英文语料即可。

---

## 训练输出汇总

| 阶段 | 输出文件 | 大小 | 能力 |
|------|----------|------|------|
| 预训练 | `pretrain_512.pth` | ~100MB | 文本续写 |
| SFT | `full_sft_512.pth` | ~100MB | 对话、指令遵循 |
| DPO | `rlhf_512.pth` | ~100MB | 人类偏好对齐 |
| LoRA | `lora/lora_xxx_512.pth` | ~几 MB | 领域专家 |
| 蒸馏 | `full_dist_512.pth` | ~100MB | 小体积高性能 |
| 推理 | `reason_512.pth` | ~100MB | CoT 推理能力 |

---

## 快速开始脚本

```bash
#!/bin/bash
# 一键训练脚本（26M 模型）

set -e

cd /Users/renjiapin/Documents/GitHub/Newmind-master

# 1. 准备目录
mkdir -p out dataset

# 2. 下载数据集（如果还没有）
cd dataset
if [ ! -f pretrain_hq.jsonl ]; then
    wget https://huggingface.co/datasets/jingyaogong/minimind_dataset/resolve/main/pretrain_hq.jsonl
fi
if [ ! -f sft_512.jsonl ]; then
    wget https://huggingface.co/datasets/jingyaogong/minimind_dataset/resolve/main/sft_mini_512.jsonl -O sft_512.jsonl
fi
if [ ! -f dpo.jsonl ]; then
    wget https://huggingface.co/datasets/jingyaogong/minimind_dataset/resolve/main/dpo.jsonl
fi
cd ..

# 3. 预训练
cd trainer
python train_pretrain.py \
    --epochs 1 \
    --batch_size 32 \
    --accumulation_steps 8 \
    --learning_rate 5e-4 \
    --hidden_size 512 \
    --num_hidden_layers 8 \
    --data_path "../dataset/pretrain_hq.jsonl" \
    --out_dir "../out"

# 4. SFT
python train_full_sft.py \
    --epochs 3 \
    --batch_size 16 \
    --learning_rate 5e-7 \
    --hidden_size 512 \
    --num_hidden_layers 8 \
    --data_path "../dataset/sft_512.jsonl" \
    --out_dir "../out"

# 5. DPO
python train_dpo.py \
    --epochs 2 \
    --batch_size 8 \
    --learning_rate 1e-8 \
    --hidden_size 512 \
    --num_hidden_layers 8 \
    --data_path "../dataset/dpo.jsonl" \
    --out_dir "../out"

echo "✅ 训练完成！模型保存在 ../out/ 目录"
```

---

*最后更新：2026 年 3 月 | RNewMind v2.0*
