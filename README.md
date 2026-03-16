# RNewiMind

轻量级大语言模型，从零实现完整的 Transformer 架构及训练流程。

---

## 📋 目录

- [模型架构](#模型架构)
- [技术特性](#技术特性)
- [项目结构](#项目结构)
- [环境要求](#环境要求)
- [快速开始](#快速开始)
- [训练流程](#训练流程)
- [模型推理](#模型推理)
- [依赖安装](#依赖安装)

---

## 模型架构

RNewMind 采用 Decoder-Only Transformer 架构，核心配置如下：

| 参数 | Small (26M) | Standard (104M) | MoE (145M) |
|------|-------------|-----------------|------------|
| `hidden_size` | 512 | 768 | 640 |
| `num_hidden_layers` | 8 | 16 | 8 |
| `num_attention_heads` | 8 | 12 | 10 |
| `num_key_value_heads` | 2 | 3 | 2 |
| `vocab_size` | 6400 | 6400 | 6400 |
| `max_position_embeddings` | 32768 | 32768 | 32768 |
| `use_moe` | ❌ | ❌ | ✅ |

### 核心技术栈

- **注意力机制**: GQA (Grouped Query Attention) + RoPE (Rotary Positional Embedding)
- **归一化**: RMSNorm (Pre-Norm 架构)
- **激活函数**: SiLU (Swish)
- **位置编码**: RoPE，支持最长 32K 上下文
- **注意力优化**: Flash Attention 2.0 (PyTorch 2.0+)
- **MoE 架构**: 稀疏混合专家网络，支持可配置专家数量和路由策略

---

## 技术特性

### 架构特性

- ✅ **GQA (分组查询注意力)**: 减少 KV Cache 显存占用，提升推理速度
- ✅ **RoPE (旋转位置编码)**: 支持长度外推，改善长文本理解
- ✅ **RMSNorm**: 更稳定的归一化方法，加速收敛
- ✅ **SwiGLU**: Swish 门控线性单元，增强非线性表达能力
- ✅ **MoE (混合专家)**: 稀疏激活，扩大模型容量同时控制计算成本

### 训练技术

- ✅ **全量预训练**: 从无标注语料学习语言表示
- ✅ **监督微调 (SFT)**: 指令遵循能力训练
- ✅ **LoRA 微调**: 参数高效微调，仅训练低秩适配器
- ✅ **DPO (直接偏好优化)**: 替代传统 RLHF，更稳定的对齐方法
- ✅ **知识蒸馏**: 大模型指导小模型，压缩模型体积
- ✅ **推理蒸馏**: 思维链 (CoT) 能力迁移

### 工程优化

- ✅ **梯度累积**: 支持小显存设备大 batch 训练
- ✅ **DDP 分布式训练**: 多卡并行加速
- ✅ **混合精度训练**: AMP 自动混合精度
- ✅ **KV Cache**: 推理时复用历史键值对，加速生成
- ✅ **流式输出**: TextStreamer 实时显示生成内容

---

## 项目结构

```
minimind/
├── model/                      # 模型定义
│   ├── lite/                   # 核心模型实现
│   │   └── rnewmind_base.py    # RNewMind 架构定义
│   ├── core/                   # 基础组件
│   │   ├── attention.py        # 注意力机制
│   │   ├── norm.py             # RMSNorm 归一化
│   │   ├── rope.py             # 旋转位置编码
│   │   ├── feedforward.py      # 前馈网络
│   │   └── moe.py              # MoE 前馈网络
│   ├── config/                 # 模型配置
│   ├── model_minimind.py       # 兼容性导出
│   ├── model_lora.py           # LoRA 适配器实现
│   ├── tokenizer.json          # 分词器配置 (6400 tokens)
│   └── tokenizer_config.json   # 分词器配置
├── dataset/                    # 数据处理
│   ├── __init__.py
│   └── lm_dataset.py           # Pretrain/SFT/DPO/RLAIF 数据集
├── trainer/                    # 训练脚本
│   ├── engine.py               # 训练引擎
│   ├── train_pretrain.py       # 预训练
│   ├── train_full_sft.py       # 全量 SFT
│   ├── train_lora.py           # LoRA 微调
│   ├── train_dpo.py            # DPO 偏好对齐
│   ├── train_distillation.py   # 知识蒸馏
│   └── train_distill_reason.py # 推理蒸馏
├── scripts/                    # 工具脚本
│   ├── web_demo.py             # Streamlit Web 界面
│   ├── serve_openai_api.py     # OpenAI 兼容 API 服务
│   ├── chat_openai_api.py      # OpenAI API 调用示例
│   ├── convert_model.py        # 模型格式转换
│   └── train_tokenizer.py      # 分词器训练
├── evaluation/                 # 评估脚本
├── inference/                  # 推理脚本
├── eval_model.py               # 模型推理/评估入口
├── requirements.txt            # 依赖列表
├── 训练手册.md                  # 详细训练文档
├── 所使用的模型.md              # 模型架构详解
└── README.md                   # 项目说明
```

---

## 环境要求

### 硬件要求

| 配置 | 最低要求 | 推荐配置 |
|------|----------|----------|
| GPU | NVIDIA 6GB | NVIDIA 24GB+ |
| 内存 | 16 GB | 64 GB+ |
| 存储 | 50 GB | 200 GB SSD |

### 软件要求

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+
- Transformers 4.48.0+

---

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 下载预训练模型

从 HuggingFace 或 ModelScope 下载权重文件至 `./out/` 目录。

### 3. 推理测试

```bash
# 测试 SFT 对话模型
python eval_model.py --model_mode 1

# 测试 RLHF 对齐模型
python eval_model.py --model_mode 2

# 测试推理模型 (CoT)
python eval_model.py --model_mode 3

# 测试 RLAIF 模型
python eval_model.py --model_mode 4

# 使用 LoRA 领域适配
python eval_model.py --lora_name lora_medical --model_mode 2
```

### 推理参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model_mode` | 1 | 0: 预训练，1: SFT-Chat，2: RLHF-Chat，3: Reason，4: RLAIF-Chat |
| `--hidden_size` | 512 | 隐藏层维度 (512=Small, 768=Standard, 640=MoE) |
| `--num_hidden_layers` | 8 | Transformer 层数 |
| `--use_moe` | False | 是否启用 MoE 架构 |
| `--max_seq_len` | 8192 | 最大生成长度 |
| `--history_cnt` | 0 | 携带历史对话条数（需为偶数） |
| `--temperature` | 0.85 | 采样温度 |
| `--top_p` | 0.85 | Top-P 采样阈值 |
| `--lora_name` | None | LoRA 适配器名称 |
| `--load` | 0 | 0: 原生 PyTorch 权重，1: Transformers 格式 |

---

## 训练流程

### 完整训练流程

```
预训练 → 监督微调 → 偏好对齐
   ↓          ↓          ↓
基础能力  指令遵循    人类对齐
```

### 步骤 1: 预训练

```bash
cd trainer
python train_pretrain.py \
    --epochs 1 \
    --batch_size 32 \
    --learning_rate 5e-4 \
    --hidden_size 512 \
    --num_hidden_layers 8 \
    --max_seq_len 512 \
    --data_path "../dataset/pretrain_hq.jsonl"
```

**输出**: `./out/pretrain_512.pth`

### 步骤 2: 监督微调 (SFT)

```bash
python train_full_sft.py \
    --epochs 2 \
    --batch_size 16 \
    --learning_rate 5e-7 \
    --hidden_size 512 \
    --max_seq_len 512 \
    --data_path "../dataset/sft_mini_512.jsonl"
```

**输出**: `./out/full_sft_512.pth`

### 步骤 3: DPO 偏好对齐

```bash
python train_dpo.py \
    --epochs 2 \
    --batch_size 8 \
    --learning_rate 1e-8 \
    --max_seq_len 1024 \
    --data_path "../dataset/dpo.jsonl"
```

**输出**: `./out/rlhf_512.pth`

### LoRA 微调 (可选)

```bash
python train_lora.py \
    --epochs 10 \
    --batch_size 32 \
    --learning_rate 1e-4 \
    --data_path "../dataset/lora_medical.jsonl"
```

**输出**: `./out/lora/lora_medical_512.pth`

---

## 模型推理

### 命令行交互

```bash
python eval_model.py --model_mode 1
```

支持两种模式：
- `[0] 自动测试`: 使用预设问题批量测试
- `[1] 手动输入`: 交互式对话

### API 服务

启动 OpenAI 兼容 API：

```bash
python scripts/serve_openai_api.py
```

调用示例：

```python
from openai import OpenAI

client = OpenAI(api_key="none", base_url="http://localhost:8000/v1")

response = client.chat.completions.create(
    model="minimind",
    messages=[{"role": "user", "content": "你好，请介绍一下自己"}]
)
print(response.choices[0].message.content)
```

### Web 界面

```bash
pip install streamlit
streamlit run scripts/web_demo.py
```

---

## 依赖安装

### 完整依赖列表

完整依赖请参见 `requirements.txt`，核心依赖如下：

```txt
torch==2.2.2
torchvision==0.17.2
transformers==4.48.0
datasets==2.21.0
trl==0.13.0
peft==0.7.1
tiktoken==0.5.1
streamlit==1.30.0
openai==1.59.6
```

### 快速安装

```bash
# 安装 PyTorch (CUDA 11.8)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 安装其他依赖（使用清华镜像源）
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

---

## 相关文档

- **[训练手册.md](训练手册.md)**: 详细训练流程和各脚本使用说明
- **[所使用的模型.md](所使用的模型.md)**: 模型架构、数学原理和参数详解

---

*最后更新：2026 年 3 月*
