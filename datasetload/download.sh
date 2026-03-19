#!/bin/bash
# =============================================================================
# RNewmind 数据集下载启动脚本 (增强交互版)
# =============================================================================
# 用法：./download.sh [选项]
# 示例：./download.sh --interactive
# =============================================================================

set -e

# 脚本目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# 打印横幅
print_banner() {
    echo -e "${CYAN}"
    echo "╔═══════════════════════════════════════════════════════════╗"
    echo "║                                                           ║"
    echo "║   🎓  RNewmind 教育 AI 助教 - 数据集下载工具                  ║"
    echo "║       High-Performance Dataset Downloader                 ║"
    echo "║                                                           ║"
    echo "╚═══════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

# 检查 Python 环境
check_python() {
    if ! command -v python3 &> /dev/null; then
        echo -e "${RED}错误：未找到 Python3${NC}"
        exit 1
    fi
    
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
    echo -e "${GREEN}✓ Python 版本：${PYTHON_VERSION}${NC}"
}

# 检查依赖
check_dependencies() {
    echo -e "${YELLOW}检查依赖...${NC}"
    
    python3 -c "import huggingface_hub" 2>/dev/null || {
        echo -e "${RED}错误：未安装 huggingface_hub${NC}"
        echo "请运行：pip install huggingface_hub hf_transfer"
        exit 1
    }
    echo -e "${GREEN}✓ huggingface_hub 已安装${NC}"
    
    python3 -c "import hf_transfer" 2>/dev/null && {
        echo -e "${GREEN}✓ hf_transfer 加速模块已启用${NC}"
    } || {
        echo -e "${YELLOW}⚠ hf_transfer 未安装 (可选，建议安装以提升速度)${NC}"
        echo "  运行：pip install hf_transfer"
    }
    
    python3 -c "import rich" 2>/dev/null && {
        echo -e "${GREEN}✓ rich 美化库已安装 (增强界面可用)${NC}"
    } || {
        echo -e "${YELLOW}⚠ rich 未安装 (建议安装以获得更好界面)${NC}"
        echo "  运行：pip install rich"
    }
}

# 显示菜单
show_menu() {
    echo ""
    echo -e "${CYAN}请选择启动模式:${NC}"
    echo ""
    echo -e "  ${GREEN}1)${NC} 交互模式 (推荐 - 可视化选择数据集)"
    echo -e "  ${GREEN}2)${NC} 快速模式 (使用默认配置)"
    echo -e "  ${GREEN}3)${NC} 自定义配置 (命令行参数)"
    echo -e "  ${GREEN}0)${NC} 退出"
    echo ""
}

# 交互模式
interactive_mode() {
    echo -e "${CYAN}启动交互模式...${NC}"
    python3 "$SCRIPT_DIR/download_datasets.py" --interactive "$@"
}

# 快速模式
quick_mode() {
    echo -e "${CYAN}启动快速模式 (默认配置)...${NC}"
    echo ""
    echo -e "${YELLOW}配置信息:${NC}"
    echo "  下载路径：/mnt/data/RNewmind_Datasets"
    echo "  并发数：12"
    echo "  数据集：全部"
    echo ""
    python3 "$SCRIPT_DIR/download_datasets.py" "$@"
}

# 自定义配置
custom_mode() {
    echo -e "${CYAN}请输入配置参数:${NC}"
    echo ""
    
    read -p "下载路径 [/mnt/data/RNewmind_Datasets]: " output_path
    read -p "并发数 [12]: " workers
    read -p "是否顺序下载 (y/N): " sequential
    
    args=""
    
    if [[ -n "$output_path" ]]; then
        args="$args --output $output_path"
    fi
    
    if [[ -n "$workers" ]]; then
        args="$args --workers $workers"
    fi
    
    if [[ "$sequential" =~ ^[Yy]$ ]]; then
        args="$args --sequential"
    fi
    
    python3 "$SCRIPT_DIR/download_datasets.py" $args "$@"
}

# 主函数
main() {
    print_banner
    check_python
    check_dependencies
    
    # 如果有命令行参数，直接运行
    if [[ $# -gt 0 ]]; then
        python3 "$SCRIPT_DIR/download_datasets.py" "$@"
        exit $?
    fi
    
    # 否则显示菜单
    show_menu
    
    read -p "请选择 [0-3]: " choice
    
    case $choice in
        1)
            interactive_mode
            ;;
        2)
            quick_mode
            ;;
        3)
            custom_mode
            ;;
        0)
            echo -e "${GREEN}再见!${NC}"
            exit 0
            ;;
        *)
            echo -e "${RED}无效选择!${NC}"
            exit 1
            ;;
    esac
}

# 运行主函数
main "$@"
