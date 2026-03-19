# RNewmind 数据集下载工具使用说明 (增强交互版)

## 📋 目录

- [新功能特性](#新功能特性)
- [环境准备](#环境准备)
- [安装依赖](#安装依赖)
- [运行方式](#运行方式)
- [界面预览](#界面预览)
- [配置说明](#配置说明)
- [故障排除](#故障排除)

---

## ✨ 新功能特性

### 🎨 增强可视化界面

- **Rich TUI 仪表板**: 实时显示所有数据集下载状态
- **动态进度条**: 每个数据集的实时下载进度
- **速度监控**: 实时显示下载速度和预估剩余时间
- **彩色状态标识**: ⏳ 等待 / 🔄 下载中 / ✅ 完成 / ❌ 失败 / ⏭️ 跳过

### 🎮 交互式操作

- **数据集选择菜单**: 可视化选择要下载的数据集
- **配置向导**: 逐步引导设置下载参数
- **启动菜单**: 多种启动模式可选

### 📊 实时统计

- 总下载量 (TB/GB)
- 平均下载速度 (MB/s)
- 已用时间 / 预估剩余时间
- 成功/失败/跳过数量

---

## 环境准备

### 系统要求

| 项目 | 要求 |
|------|------|
| **操作系统** | Linux / macOS / Windows |
| **Python** | 3.8+ |
| **磁盘空间** | ≥ 3TB 可用空间 |
| **网络** | 稳定的互联网连接 (建议 ≥ 100Mbps) |

### 推荐配置 (Dell T640)

```
CPU: 双路 Intel Xeon Gold
内存：256GB+
存储：4TB+ NVMe SSD (或 RAID 阵列)
网络：1Gbps+ 以太网
```

---

## 安装依赖

### 步骤 1: 安装基础依赖

```bash
cd /path/to/Newmind-master/datasetload
pip install huggingface_hub hf_transfer
```

### 步骤 2: 安装美化库 (推荐，获得增强界面)

```bash
pip install rich
```

### 步骤 3: 验证安装

```bash
# 验证基础依赖
python -c "import huggingface_hub, hf_transfer; print('✓ 基础依赖 OK')"

# 验证 rich (可选)
python -c "import rich; print('✓ Rich 美化库 OK')"
```

---

## 运行方式

### 方式 1: Shell 脚本启动 (推荐)

```bash
# 赋予执行权限
chmod +x download.sh

# 运行 (显示交互菜单)
./download.sh

# 或直接传递参数
./download.sh --output /mnt/data/RNewmind_Datasets --workers 16
```

### 方式 2: Python 直接运行

```bash
# 交互模式 (推荐)
python download_datasets.py --interactive

# 快速模式 (默认配置)
python download_datasets.py

# 自定义配置
python download_datasets.py \
    --output /mnt/data/RNewmind_Datasets \
    --workers 16 \
    --sequential
```

---

## 🎨 界面预览

### 交互模式菜单

```
╔═══════════════════════════════════════════════════════════╗
║                                                           ║
║   🎓  RNewmind 教育 AI 助教 - 数据集下载工具                  ║
║       High-Performance Dataset Downloader                 ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝

📋 可选数据集列表:

ID   状态   数据集                                             预估大小    
────────────────────────────────────────────────────────────────────────────────
0    [✓]   HuggingFaceTB/cosmopedia (auto_math_text)         ~50GB       
1    [✓]   HuggingFaceTB/cosmopedia (science)                ~100GB      
2    [✓]   HuggingFaceTB/cosmopedia (stanford_courses)       ~200GB      
3    [✓]   openbmb/UltraData-Math                            ~100GB      
4    [✓]   arxiv/arxiv                                       ~800GB      
5    [✓]   bigcode/the-stack-v2                              ~1.5TB      
6    [✓]   OpenAssistant/oasst1                              ~50GB       
────────────────────────────────────────────────────────────────────────────────

已选：7/7

操作指南:
  - 输入数字切换选择 (如：0 或 0,1,2)
  - 输入 'a' 全选 / 'n' 全不选
  - 输入 'c' 确认开始下载
  - 输入 'q' 退出程序

请输入选择：
```

### Rich TUI 仪表板 (实时下载监控)

```
╭──────────────────────────────────────────────────────────────────────────────╮
│                    🎓 RNewmind 数据集下载监控中心                             │
╰──────────────────────────────────────────────────────────────────────────────╯

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                               📊 统计信息                                    ┃
┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫
┃  总数据集    7                    已完成      3                              ┃
┃  已跳过      1                    失败        0                              ┃
┃  总下载量    1.25 TB              平均速度    85.3 MB/s                      ┃
┃  耗时        4523s (1.3h)         状态        🟢 下载中                      ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

╭───────────────────────────── 数据集下载状态 ─────────────────────────────╮
│ 数据集                  │ 状态 │ 进度    │ 速度       │ 耗时             │
├───────────────────────────────────────────────────────────────────────────┤
│ HuggingFaceTB/cosmopedia│ ✅   │ -       │ -          │ 245s             │
│   └─ auto_math_text                                             │
│ HuggingFaceTB/cosmopedia│ 🔄   │ 45.2%   │ 92.1 MB/s  │ 180s             │
│   └─ science                                                  │
│ bigcode/the-stack-v2    │ ⏳   │ -       │ -          │ -                │
│ OpenAssistant/oasst1    │ ⏳   │ -       │ -          │ -                │
╰───────────────────────────────────────────────────────────────────────────╯
```

### 基础模式输出 (无 rich 时)

```
🎓  RNewmind 教育 AI 助教 - 数据集下载工具

开始下载 7 个数据集
下载路径：/mnt/data/RNewmind_Datasets
并发数：12

[1/7] ✓ HuggingFaceTB/cosmopedia (auto_math_text)
[2/7] ✓ HuggingFaceTB/cosmopedia (science)
[3/7] 🔄 bigcode/the-stack-v2 - 尝试 1/5
...

============================================================
下载完成摘要
============================================================
总数据集数：7
成功：6
失败：0
跳过 (已存在): 1
总下载量：2.85 TB
总耗时：12450.5s (3.46 小时)
平均速度：78.52 MB/s
============================================================
```

---

## 配置说明

### 命令行参数

| 参数 | 简写 | 说明 | 默认值 |
|------|------|------|--------|
| `--interactive` | `-i` | 交互模式 (可视化选择) | `False` |
| `--output` | `-o` | 下载目标路径 | `/mnt/data/RNewmind_Datasets` |
| `--workers` | `-w` | 最大并发下载数 | `12` |
| `--token` | `-t` | Hugging Face API Token | `None` |
| `--sequential` | | 顺序下载 (禁用并发) | `False` |
| `--skip-disk-check` | | 跳过磁盘空间检查 | `False` |
| `--min-disk-gb` | | 最小可用磁盘空间 (GB) | `3000` |
| `--log-level` | | 日志级别 | `INFO` |
| `--log-dir` | | 日志目录 | `./logs` |
| `--repos` | | 仅下载指定数据集 | `None` |
| `--debug` | | 调试模式 | `False` |

### 环境变量

```bash
# 设置 Hugging Face 镜像站
export HF_ENDPOINT=https://hf-mirror.com

# 启用 hf_transfer 加速
export HF_HUB_ENABLE_HF_TRANSFER=1

# 设置最大并发数
export HF_HUB_MAX_CONCURRENT=16

# 设置 Token
export HF_TOKEN=hf_xxxxxxxxxxxxx
```

---

## 数据集清单

| 数据集 | 子集 | 描述 | 预估大小 |
|--------|------|------|----------|
| `HuggingFaceTB/cosmopedia` | auto_math_text | 自动生成的数学文本 | ~50GB |
| `HuggingFaceTB/cosmopedia` | science | 科学领域数据 | ~100GB |
| `HuggingFaceTB/cosmopedia` | stanford_courses | 斯坦福课程数据 | ~200GB |
| `openbmb/UltraData-Math` | - | 高质量数学数据集 | ~100GB |
| `arxiv/arxiv` | - | 数学、信息论、信号处理论文 | ~800GB |
| `bigcode/the-stack-v2` | - | Python、LaTeX、C++ 代码 | ~1.5TB |
| `OpenAssistant/oasst1` | - | 高质量对话数据 | ~50GB |

**预计总大小**: ~2.8TB

---

## 故障排除

### 问题 1: 界面显示异常

```bash
# 确保安装了 rich 库
pip install rich

# 或使用基础模式 (不依赖 rich)
python download_datasets.py
```

### 问题 2: 下载速度慢

```bash
# 确保使用镜像站
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_ENABLE_HF_TRANSFER=1

# 增加并发数
python download_datasets.py --workers 16
```

### 问题 3: 下载中断

```bash
# 直接重新运行，支持断点续传
python download_datasets.py --output /mnt/data/RNewmind_Datasets
```

### 问题 4: 磁盘空间不足

```bash
# 检查磁盘空间
df -h /mnt/data

# 跳过检查 (谨慎使用)
python download_datasets.py --skip-disk-check
```

### 问题 5: 数据集访问受限

```bash
# 使用 Token
python download_datasets.py --token hf_xxxxxxxxxxxxx
```

---

## 日志查看

```bash
# 查看最新日志
tail -f logs/download_*.log

# 查看完成记录
grep "\[完成\]" logs/download_*.log

# 查看失败记录
grep "\[失败\]" logs/download_*.log
```

---

## 验证下载

```bash
# 检查目录结构
tree /mnt/data/RNewmind_Datasets -L 2

# 检查文件大小
du -sh /mnt/data/RNewmind_Datasets/*

# 验证数据集
python -c "
from datasets import load_dataset
ds = load_dataset('HuggingFaceTB/cosmopedia', 'auto_math_text', 
                  data_dir='/mnt/data/RNewmind_Datasets/cosmopedia/auto_math_text')
print(f'数据集加载成功，样本数：{len(ds[\"train\"])}')
"
```

---

## 快速命令参考

```bash
# 交互模式 (第一次使用推荐)
./download.sh

# 快速下载全部数据集
python download_datasets.py

# 指定路径和并发数
python download_datasets.py -o /mnt/data/RNewmind_Datasets -w 16

# 仅下载特定数据集
python download_datasets.py --repos "HuggingFaceTB/cosmopedia" "OpenAssistant/oasst1"

# 顺序下载 (网络不稳定时)
python download_datasets.py --sequential

# 后台运行
nohup python download_datasets.py > download.log 2>&1 &
```

---

*最后更新：2026-03-19 | RNewmind Dataset Downloader v2.0*
