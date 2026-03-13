# MiniMind 项目结构详解

本文档详细介绍 MiniMind 项目的目录结构、各文件夹功能及核心文件说明。

---

## 📁 项目根目录结构

```
g:\minimind-master\
├── CODE_OF_CONDUCT.md      # 代码行为准则
├── LICENSE                 # 开源许可证
├── README.md               # 项目说明文档（中文）
├── README_en.md            # 项目说明文档（英文）
├── requirements.txt        # Python 依赖包列表
├── eval_model.py           # 模型评估/测试脚本
├── .gitignore              # Git 忽略文件配置
├── .vscode/                # VSCode 编辑器配置
├── dataset/                # 数据集处理模块
├── model/                  # 模型定义模块
├── trainer/                # 训练脚本模块
├── scripts/                # 工具脚本
└── images/                 # 项目图片资源
```

---

## 📂 核心文件夹详解

### 1️⃣ `model/` - 模型定义模块

**功能**：包含 MiniMind 大语言模型的核心结构定义、配置和分词器

| 文件名 | 功能描述 |
|--------|----------|
| `model_minimind.py` | **核心模型文件**：定义 Transformer Decoder-Only 架构（RMSNorm、RoPE、Attention、MoE） |
| `model_lora.py` | **LoRA 微调模块**：从 0 实现 LoRA 参数高效微调算法 |
| `tokenizer.json` | **分词器词表**：6400 个 token 的词汇表 |
| `tokenizer_config.json` | **分词器配置**：BOS/EOS token、padding 策略等 |

**模型配置参数**：
```python
hidden_size: 512          # 隐藏层维度
num_hidden_layers: 8      # Transformer 层数
num_attention_heads: 8    # 注意力头数
num_key_value_heads: 2    # KV 头数（GQA 分组查询）
vocab_size: 6400          # 词表大小
use_moe: False            # 是否启用混合专家
```

---

### 2️⃣ `dataset/` - 数据集处理模块

**功能**：定义不同训练阶段的数据集加载和处理逻辑

| 文件名 | 功能 |
|--------|------|
| `lm_dataset.py` | 包含 4 种 Dataset 类：PretrainDataset、SFTDataset、DPODataset、RLAIFDataset |

**数据集类别**：
| 类名 | 用途 | 输入格式 |
|------|------|----------|
| `PretrainDataset` | 预训练 | `{"text": "文本内容..."}` |
| `SFTDataset` | 监督微调 | `{"conversations": [...]}` |
| `DPODataset` | 偏好优化 | `{"chosen": [...], "rejected": [...]}` |

---

### 3️⃣ `trainer/` - 训练脚本模块

**功能**：包含各个训练阶段的完整训练代码（PyTorch 原生实现）

| 文件名 | 训练阶段 | 输出权重 |
|--------|----------|----------|
| `train_pretrain.py` | 预训练 | `pretrain_*.pth` |
| `train_full_sft.py` | 监督微调 | `full_sft_*.pth` |
| `train_lora.py` | LoRA 微调 | `lora_xxx_*.pth` |
| `train_dpo.py` | RLHF-DPO | `rlhf_*.pth` |
| `train_distillation.py` | 知识蒸馏 | `full_sft_*.pth` |
| `train_distill_reason.py` | 推理蒸馏 | `reason_*.pth` |

---

### 4️⃣ `scripts/` - 工具脚本模块

| 文件名 | 功能 |
|--------|------|
| `web_demo.py` | WebUI 聊天界面（Streamlit） |
| `chat_openai_api.py` | OpenAI API 客户端示例 |
| `serve_openai_api.py` | OpenAI 兼容 API 服务端 |
| `convert_model.py` | PyTorch ↔ Transformers 模型转换 |
| `train_tokenizer.py` | 分词器训练脚本 |

---

### 5️⃣ `images/` - 图片资源

存放项目文档所需的图片：logo.png、minimind2.gif、LLM-structure.png、dataset.jpg 等

---

## 📖 详细训练过程指南

### 🔹 步骤 0：环境准备

#### 0.1 硬件要求
| 配置 | 最低要求 | 推荐配置 |
|------|----------|----------|
| GPU | NVIDIA 1060 (6GB) | NVIDIA 3090 (24GB) × 8 |
| 内存 | 16 GB | 128 GB |
| 存储 | 50 GB | 500 GB SSD |

#### 0.2 安装依赖
```bash
git clone https://github.com/jingyaogong/minimind.git
cd minimind
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

#### 0.3 验证 CUDA
```python
python -c "import torch; print(torch.cuda.is_available())"
```

#### 0.4 下载数据集
从 ModelScope 或 HuggingFace 下载到 `./dataset/`：

**最小训练组合**：
- `pretrain_hq.jsonl` (1.6GB) - 预训练
- `sft_mini_512.jsonl` (1.2GB) - SFT

---

### 🔹 步骤 1：预训练（Pretrain）

**目标**：让模型学习知识，掌握"词语接龙"能力

#### 1.1 训练命令
```bash
cd trainer

# 单卡
python train_pretrain.py

# 多卡（N 为显卡数）
torchrun --nproc_per_node N train_pretrain.py
```

#### 1.2 关键参数
```bash
python train_pretrain.py \
    --epochs 1 \                    # 训练轮次
    --batch_size 32 \               # 批次大小
    --learning_rate 5e-4 \          # 学习率
    --hidden_size 512 \             # 模型维度
    --num_hidden_layers 8 \         # 层数
    --max_seq_len 512 \             # 序列长度
    --accumulation_steps 8 \        # 梯度累积
    --save_interval 100 \           # 保存间隔
    --data_path "../dataset/pretrain_hq.jsonl"
```

#### 1.3 训练流程
```
1. 加载预训练数据集 (PretrainDataset)
   ↓
2. 初始化 MiniMindForCausalLM 模型
   ↓
3. 前向传播：logits = model(input_ids)
   ↓
4. 计算损失：CrossEntropyLoss(logits, labels)
   ↓
5. 反向传播：loss.backward()
   ↓
6. 梯度裁剪 + 优化器更新
   ↓
7. 每 100 步保存权重到 ./out/pretrain_*.pth
```

#### 1.4 损失计算核心代码
```python
loss_fct = nn.CrossEntropyLoss(reduction='none')
res = model(X)
loss = loss_fct(res.logits.view(-1, vocab_size), Y.view(-1))
loss = (loss * loss_mask).sum() / loss_mask.sum()  # masked loss
loss = loss / args.accumulation_steps  # 梯度累积
```

#### 1.5 学习率调度
```python
def get_lr(current_step, total_steps, lr):
    return lr / 10 + 0.5 * lr * (1 + math.cos(π * current_step / total_steps))
```

#### 1.6 输出文件
```
./out/pretrain_512.pth          # Small 模型
./out/pretrain_768.pth          # Standard 模型
./out/pretrain_512_moe.pth      # MoE 模型
```

#### 1.7 训练日志
```
Epoch:[1/1](0/5000) loss:2.532 lr:0.000500000000 epoch_Time:45min:
Epoch:[1/1](100/5000) loss:1.823 lr:0.000499876543 epoch_Time:42min:
Epoch:[1/1](200/5000) loss:1.654 lr:0.000498765432 epoch_Time:40min:
```

#### 1.8 预计耗时
| 数据集 | 单卡 3090 | 8 卡 3090 |
|--------|-----------|-----------|
| pretrain_hq (1.6GB) | ~1.1 小时 | ~10 分钟 |

---

### 🔹 步骤 2：监督微调（Full SFT）

**目标**：让模型学会对话格式，从"接龙"变"聊天"

#### 2.1 训练命令
```bash
cd trainer
python train_full_sft.py
```

#### 2.2 关键参数
```bash
python train_full_sft.py \
    --epochs 2 \                # 微调轮次
    --batch_size 16 \           # 批次大小
    --learning_rate 5e-7 \      # 学习率（比预训练小）
    --hidden_size 512 \         # 与预训练一致
    --max_seq_len 512 \         # 对话长度
    --data_path "../dataset/sft_mini_512.jsonl"
```

#### 2.3 训练流程
```
1. 加载预训练权重：pretrain_512.pth
   ↓
2. 加载 SFT 数据集 (SFTDataset)
   ↓
3. 构建 ChatML 对话模板
   ↓
4. 生成 loss_mask（仅对 assistant 回复计算 loss）
   ↓
5. 前向传播 + 计算损失
   ↓
6. 反向传播 + 更新参数
   ↓
7. 每 100 步保存到 ./out/full_sft_*.pth
```

#### 2.4 ChatML 模板
```python
# 输入数据
{
  "conversations": [
    {"role": "user", "content": "你好"},
    {"role": "assistant", "content": "你好！我是 MiniMind"}
  ]
}

# 转换为模板（使用 tokenizer.apply_chat_template）

**Loss Mask 生成**：
- 仅对 assistant 的回复计算 loss
- user 的消息 loss_mask = 0

#### 2.5 输出文件
```
./out/full_sft_512.pth          # Small 模型
./out/full_sft_768.pth          # Standard 模型
```

#### 2.6 训练日志
```
Epoch:[1/2](0/3000) loss:1.234 lr:0.000000500000 epoch_Time:30min:
Epoch:[1/2](100/3000) loss:0.876 lr:0.000000498765 epoch_Time:28min:
```

#### 2.7 预计耗时
| 数据集 | 单卡 3090 |
|--------|-----------|
| sft_mini_512 (1.2GB) | ~1 小时 |
| sft_512 (7.5GB) | ~6 小时 |
| sft_2048 (9GB) | ~7.5 小时 |

---

### 🔹 步骤 3：人类反馈强化学习（RLHF-DPO）

**目标**：优化模型回复质量，使其更符合人类偏好

#### 3.1 训练命令
```bash
cd trainer
python train_dpo.py
```

#### 3.2 关键参数
```bash
python train_dpo.py \
    --epochs 2 \
    --batch_size 8 \
    --learning_rate 1e-8 \
    --max_seq_len 1024 \
    --data_path "../dataset/dpo.jsonl"
```

#### 3.3 训练流程
```
1. 加载 SFT 权重：full_sft_512.pth
2. 加载 DPO 数据集（chosen + rejected 对）
3. 加载参考模型（ref_model），冻结参数
4. 前向传播：actor 和 ref 分别输出 probs
5. 计算 DPO Loss
6. 反向传播 + 更新 actor 参数
7. 保存到 ./out/rlhf_*.pth
```

#### 3.4 DPO 损失公式
```
loss = -log_sigmoid(β * ((π_chosen - π_rejected) - (ref_chosen - ref_rejected)))
```

#### 3.5 输出文件
```
./out/rlhf_512.pth
```

#### 3.6 注意事项
- 学习率必须很小（≤1e-8），避免遗忘
- DPO 仅优化"礼貌"，不提升"智力"

---

### 🔹 步骤 4：LoRA 微调

**目标**：让模型学习特定领域知识（医疗、自我认知等）

#### 4.1 训练命令
```bash
cd trainer
python train_lora.py
```

#### 4.2 关键参数
```bash
python train_lora.py \
    --epochs 10 \
    --batch_size 32 \
    --learning_rate 1e-4 \
    --data_path "../dataset/lora_medical.jsonl" \
    --lora_name "lora_medical"
```

#### 4.3 训练流程
```
1. 加载 SFT 权重：full_sft_512.pth
2. 应用 LoRA 适配器（冻结原模型）
3. 仅对 LoRA 参数进行训练
4. 保存到 ./out/lora/lora_xxx_*.pth
```

#### 4.4 输出文件
```
./out/lora/lora_medical_512.pth
./out/lora/lora_identity_512.pth
```

#### 4.5 推理时使用 LoRA
```bash
python eval_model.py --lora_name lora_medical --model_mode 2
```

---

### 🔹 步骤 5：知识蒸馏（可选）

**黑盒蒸馏**（推荐）：
```bash
python train_full_sft.py --data_path ../dataset/sft_1024.jsonl --max_seq_len 1024
```
说明：sft_1024.jsonl 收集自 Qwen2.5-7/72B

---

### 🔹 步骤 6：推理模型蒸馏

#### 6.1 训练命令
```bash
cd trainer
python train_distill_reason.py --data_path ../dataset/r1_mix_1024.jsonl
```

#### 6.2 输出文件
```
./out/reason_512.pth
```

---

## 📊 完整训练流程图

```
步骤 1 预训练 → 步骤 2 监督微调 → 步骤 3 RLHF-DPO → 最终模型
                              ↓
                        步骤 4 LoRA (可选)
                        步骤 5 蒸馏 (可选)
                        步骤 6 推理蒸馏 (可选)
```

---

## 💡 训练方案推荐

### 方案 A：快速复现 Zero 模型（最经济）
- **数据集**：pretrain_hq.jsonl + sft_mini_512.jsonl
- **命令**：
  ```bash
  python train_pretrain.py --epochs 1
  python train_full_sft.py --epochs 2
  ```
- **总耗时**：2.1 小时
- **总成本**：2.73 元（单卡 3090）

### 方案 B：完整训练 MiniMind2-Small
- **数据集**：pretrain_hq + sft_512 + sft_2048 + dpo.jsonl
- **总耗时**：约 38 小时
- **总成本**：约 50 元

### 方案 C：领域适配（LoRA）
```bash
python train_lora.py --data_path ../dataset/lora_medical.jsonl
```

---

## 📋 模型测试

```bash
# 测试预训练模型（仅接龙）
python eval_model.py --model_mode 0

# 测试 SFT 模型（可对话）
python eval_model.py --model_mode 1

# 测试 RLHF 模型（偏好对齐）
python eval_model.py --model_mode 2

# 测试推理模型
python eval_model.py --model_mode 3

# 测试 LoRA 模型
python eval_model.py --lora_name lora_medical --model_mode 2
```

---

## 🔗 相关资源

- **HuggingFace**: https://huggingface.co/collections/jingyaogong/minimind-66caf8d999f5c7fa64f399e5
- **ModelScope**: https://www.modelscope.cn/datasets/gongjy/minimind_dataset/files
- **项目视频**: https://www.bilibili.com/video/BV12dHPeqE72

---

*文档最后更新：2026 年 3 月*