#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RNewmind 教育 AI 助教数据集下载脚本 (增强交互版)
===============================================

功能：从 Hugging Face 镜像站高速、稳定地下载约 3TB 的教育数据集
支持：断点续传、多线程并发、磁盘空间检查、失败重试、详细日志
新增：交互式 TUI 界面、实时进度条、动态速度显示、数据集选择菜单

作者：Deep Learning Engineer
日期：2026-03-19
"""

import os
import sys
import time
import logging
import shutil
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import json

# 配置 Hugging Face 镜像站 (中国用户推荐)
HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
os.environ["HF_ENDPOINT"] = HF_ENDPOINT

# 启用 hf_transfer 加速 (Rust 编写，可榨干带宽)
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

# 设置最大并发下载数
os.environ["HF_HUB_MAX_CONCURRENT"] = "16"

# 尝试导入 rich 库用于美化界面
try:
    from rich.console import Console
    from rich.progress import (
        Progress,
        BarColumn,
        TextColumn,
        DownloadColumn,
        TransferSpeedColumn,
        TimeRemainingColumn,
        SpinnerColumn,
        TaskProgressColumn,
        ProgressColumn,
    )
    from rich.table import Table
    from rich.panel import Panel
    from rich.layout import Layout
    from rich.live import Live
    from rich.text import Text
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("⚠️  未安装 rich 库，将使用基础界面模式")
    print("   安装命令：pip install rich")

from huggingface_hub import HfApi, snapshot_download, login
from huggingface_hub.utils import RepositoryNotFoundError, RevisionNotFoundError

# =============================================================================
# 配置区域 - 可根据需要修改
# =============================================================================

# 默认下载路径 (支持自定义)
DEFAULT_DOWNLOAD_PATH = "/mnt/data/RNewmind_Datasets"

# 日志配置
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# 重试配置
MAX_RETRIES = 5
RETRY_DELAY = 30  # 秒

# 并发配置
DEFAULT_MAX_WORKERS = 12  # 最大并行下载任务数

# 最小可用磁盘空间 (GB)
MIN_DISK_SPACE_GB = 3000  # 3TB

# =============================================================================
# 数据集清单 - RNewmind 教育 AI 助教所需数据
# =============================================================================

@dataclass
class DatasetConfig:
    """数据集配置类"""
    repo_id: str
    subsets: Optional[List[str]] = None
    split: Optional[str] = None
    include_patterns: Optional[List[str]] = None
    exclude_patterns: Optional[List[str]] = None
    revision: str = "main"
    priority: int = 1
    description: str = ""
    estimated_size: str = ""  # 预估大小


# 预设数据集清单
DATASETS: List[DatasetConfig] = [
    # ==================== 数学类数据集 ====================
    DatasetConfig(
        repo_id="HuggingFaceTB/cosmopedia",
        subsets=["auto_math_text"],
        split="train",
        priority=1,
        description="📐 Cosmopedia - 自动生成的数学文本数据",
        estimated_size="~50GB"
    ),
    DatasetConfig(
        repo_id="HuggingFaceTB/cosmopedia",
        subsets=["science"],
        split="train",
        priority=1,
        description="🔬 Cosmopedia - 科学领域数据",
        estimated_size="~100GB"
    ),
    DatasetConfig(
        repo_id="HuggingFaceTB/cosmopedia",
        subsets=["stanford_courses"],
        split="train",
        priority=1,
        description="🎓 Cosmopedia - 斯坦福课程数据",
        estimated_size="~200GB"
    ),
    DatasetConfig(
        repo_id="openbmb/UltraData-Math",
        subsets=None,
        split="train",
        priority=1,
        description="🧮 UltraData-Math - 高质量数学数据集",
        estimated_size="~100GB"
    ),

    # ==================== Arxiv 论文数据集 ====================
    DatasetConfig(
        repo_id="arxiv/arxiv",
        subsets=None,
        split="train",
        include_patterns=["math.*", "cs.IT", "eess.SP"],
        priority=2,
        description="📄 Arxiv - 数学、信息论、信号处理论文",
        estimated_size="~800GB"
    ),

    # ==================== 代码类数据集 ====================
    DatasetConfig(
        repo_id="bigcode/the-stack-v2",
        subsets=None,
        split="train",
        include_patterns=["*.py", "*.tex", "*.latex", "*.cpp", "*.h", "*.hpp"],
        priority=2,
        description="💻 The Stack v2 - Python、LaTeX、C++ 代码",
        estimated_size="~1.5TB"
    ),

    # ==================== 对话类数据集 ====================
    DatasetConfig(
        repo_id="OpenAssistant/oasst1",
        subsets=None,
        split="train",
        priority=1,
        description="💬 OpenAssistant OASST1 - 高质量对话数据",
        estimated_size="~50GB"
    ),
]

# =============================================================================
# 终端颜色 (不使用 rich 时)
# =============================================================================

class Colors:
    """ANSI 颜色代码"""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


def clear_screen():
    """清屏"""
    os.system('cls' if os.name == 'nt' else 'clear')


def print_banner():
    """打印横幅"""
    banner = """
╔═══════════════════════════════════════════════════════════╗
║                                                           ║
║   🎓  RNewmind 教育 AI 助教 - 数据集下载工具                  ║
║       High-Performance Dataset Downloader                 ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝
    """
    if RICH_AVAILABLE:
        console = Console()
        console.print(Panel(banner, style="bold blue"))
    else:
        print(f"{Colors.CYAN}{banner}{Colors.END}")


# =============================================================================
# 交互式菜单 (不使用 rich 时)
# =============================================================================

def show_dataset_menu(datasets: List[DatasetConfig], selected: List[int]) -> str:
    """显示数据集选择菜单"""
    output = []
    output.append(f"\n{Colors.BOLD}📋 可选数据集列表:{Colors.END}\n")
    output.append(f"{'ID':<4} {'状态':<6} {'数据集':<50} {'预估大小':<12}")
    output.append("─" * 80)
    
    for i, ds in enumerate(datasets):
        status = "✓" if i in selected else " "
        name = f"{ds.repo_id}"
        if ds.subsets:
            name += f" ({', '.join(ds.subsets)})"
        output.append(f"{i:<4} [{status}]   {name:<50} {ds.estimated_size:<12}")
    
    output.append("─" * 80)
    output.append(f"\n{Colors.GREEN}已选：{len(selected)}/{len(datasets)}{Colors.END}")
    output.append(f"\n{Colors.YELLOW}操作指南:{Colors.END}")
    output.append("  - 输入数字切换选择 (如：0 或 0,1,2)")
    output.append("  - 输入 'a' 全选 / 'n' 全不选")
    output.append("  - 输入 'c' 确认开始下载")
    output.append("  - 输入 'q' 退出程序")
    
    return "\n".join(output)


def interactive_select(datasets: List[DatasetConfig]) -> List[DatasetConfig]:
    """交互式选择数据集"""
    selected = list(range(len(datasets)))  # 默认全选
    
    while True:
        clear_screen()
        print_banner()
        print(show_dataset_menu(datasets, selected))
        
        try:
            choice = input(f"\n{Colors.CYAN}请输入选择：{Colors.END}").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n")
            sys.exit(0)
        
        if not choice:
            continue
        
        if choice.lower() == 'q':
            print("再见!")
            sys.exit(0)
        elif choice.lower() == 'c':
            if not selected:
                print(f"{Colors.RED}请至少选择一个数据集!{Colors.END}")
                time.sleep(1)
                continue
            return [datasets[i] for i in selected]
        elif choice.lower() == 'a':
            selected = list(range(len(datasets)))
        elif choice.lower() == 'n':
            selected = []
        else:
            # 解析数字输入
            try:
                indices = [int(x.strip()) for x in choice.split(',')]
                for idx in indices:
                    if 0 <= idx < len(datasets):
                        if idx in selected:
                            selected.remove(idx)
                        else:
                            selected.append(idx)
            except ValueError:
                pass


# =============================================================================
# 日志配置
# =============================================================================

def setup_logging(log_dir: str, log_level: str = "INFO") -> logging.Logger:
    """配置日志系统"""
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger("RNewmind_Downloader")
    logger.setLevel(getattr(logging, log_level.upper()))
    logger.handlers.clear()
    
    # 文件 handler
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(log_dir) / f"download_{timestamp}.log"
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    return logger


# =============================================================================
# 磁盘空间检查
# =============================================================================

def check_disk_space(path: str, required_gb: int = MIN_DISK_SPACE_GB) -> Tuple[bool, Dict]:
    """检查目标路径的可用磁盘空间"""
    try:
        stat = shutil.disk_usage(path)
        
        total_gb = stat.total / (1024 ** 3)
        used_gb = stat.used / (1024 ** 3)
        free_gb = stat.free / (1024 ** 3)
        
        info = {
            "total_gb": round(total_gb, 2),
            "used_gb": round(used_gb, 2),
            "free_gb": round(free_gb, 2),
            "required_gb": required_gb,
            "sufficient": free_gb >= required_gb
        }
        
        return info["sufficient"], info
    except Exception as e:
        return False, {"error": str(e)}


# =============================================================================
# 下载统计与状态
# =============================================================================

@dataclass
class DatasetStatus:
    """单个数据集的下载状态"""
    config: DatasetConfig
    status: str = "pending"  # pending, downloading, completed, failed, skipped
    progress: float = 0.0
    size_downloaded: int = 0
    size_total: int = 0
    speed: float = 0.0
    elapsed: float = 0.0
    retries: int = 0
    error_msg: str = ""


class DownloadStats:
    """下载统计信息"""
    
    def __init__(self):
        self.total_datasets = 0
        self.successful = 0
        self.failed = 0
        self.skipped = 0
        self.total_bytes = 0
        self.start_time = time.time()
        self.lock = threading.Lock()
        self.dataset_statuses: Dict[str, DatasetStatus] = {}
    
    def add_success(self, bytes_downloaded: int = 0):
        with self.lock:
            self.successful += 1
            self.total_bytes += bytes_downloaded
    
    def add_failed(self):
        with self.lock:
            self.failed += 1
    
    def add_skipped(self):
        with self.lock:
            self.skipped += 1
    
    def get_elapsed_time(self) -> float:
        return time.time() - self.start_time
    
    def get_speed_mbps(self) -> float:
        elapsed = self.get_elapsed_time()
        if elapsed <= 0:
            return 0
        return (self.total_bytes / (1024 * 1024)) / elapsed
    
    def update_dataset_status(self, repo_id: str, **kwargs):
        with self.lock:
            if repo_id not in self.dataset_statuses:
                self.dataset_statuses[repo_id] = DatasetStatus(
                    config=next((d for d in DATASETS if d.repo_id == repo_id), None)
                )
            status = self.dataset_statuses[repo_id]
            for key, value in kwargs.items():
                if hasattr(status, key):
                    setattr(status, key, value)


# =============================================================================
# Rich TUI 界面
# =============================================================================

if RICH_AVAILABLE:
    class DownloadDashboard:
        """下载仪表板 (Rich TUI)"""
        
        def __init__(self, stats: DownloadStats):
            self.stats = stats
            self.console = Console()
            self.progress = Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]{task.description}"),
                BarColumn(bar_width=40),
                TaskProgressColumn(),
                DownloadColumn(),
                TransferSpeedColumn(),
                TimeRemainingColumn(),
                refresh_per_second=4,
            )
            self.task_ids: Dict[str, int] = {}
        
        def create_layout(self) -> Layout:
            """创建布局"""
            layout = Layout()
            
            # 头部
            header = Panel(
                Text("🎓 RNewmind 数据集下载监控中心", justify="center", style="bold cyan"),
                style="bold blue"
            )
            
            # 统计信息
            stats_table = Table(show_header=False, box=box.SIMPLE, expand=True)
            stats_table.add_column("项目", style="cyan")
            stats_table.add_column("数值", style="green")
            stats_table.add_column("项目", style="cyan")
            stats_table.add_column("数值", style="green")
            
            elapsed = self.stats.get_elapsed_time()
            speed = self.stats.get_speed_mbps()
            
            stats_table.add_row(
                "总数据集", f"{self.stats.total_datasets}",
                "已完成", f"{self.stats.successful}"
            )
            stats_table.add_row(
                "已跳过", f"{self.stats.skipped}",
                "失败", f"{self.stats.failed}"
            )
            stats_table.add_row(
                "总下载量", f"{self.stats.total_bytes / (1024**4):.2f} TB",
                "平均速度", f"{speed:.1f} MB/s"
            )
            stats_table.add_row(
                "耗时", f"{elapsed:.0f}s ({elapsed/3600:.1f}h)",
                "状态", "🟢 下载中" if self.stats.failed == 0 else "🔴 有错误"
            )
            
            # 数据集状态表
            status_table = Table(title="数据集下载状态", box=box.ROUNDED)
            status_table.add_column("数据集", style="cyan", no_wrap=True)
            status_table.add_column("状态", justify="center")
            status_table.add_column("进度", justify="right")
            status_table.add_column("速度", justify="right")
            status_table.add_column("耗时", justify="right")
            
            for repo_id, status in self.stats.dataset_statuses.items():
                if status.status == "pending":
                    status_icon = "⏳"
                    status_style = "yellow"
                elif status.status == "downloading":
                    status_icon = "🔄"
                    status_style = "cyan"
                elif status.status == "completed":
                    status_icon = "✅"
                    status_style = "green"
                elif status.status == "failed":
                    status_icon = "❌"
                    status_style = "red"
                elif status.status == "skipped":
                    status_icon = "⏭️"
                    status_style = "blue"
                else:
                    status_icon = "❓"
                    status_style = "white"
                
                name = status.config.repo_id if status.config else repo_id
                if status.config and status.config.subsets:
                    name += f"\n  └─ {', '.join(status.config.subsets)}"
                
                progress_text = f"{status.progress:.1f}%" if status.status == "downloading" else "-"
                speed_text = f"{status.speed:.1f} MB/s" if status.status == "downloading" else "-"
                elapsed_text = f"{status.elapsed:.0f}s" if status.elapsed > 0 else "-"
                
                status_table.add_row(
                    name,
                    f"[{status_style}]{status_icon}[/{status_style}]",
                    progress_text,
                    speed_text,
                    elapsed_text
                )
            
            # 组合布局
            layout.split(
                Layout(header, name="header", size=3),
                Layout(
                    Panel(stats_table, title="📊 统计信息", style="bold green"),
                    name="stats",
                    size=10
                ),
                Layout(Panel(status_table, style="dim"), name="status"),
            )
            
            return layout
        
        def __call__(self):
            """返回当前布局"""
            return self.create_layout()


# =============================================================================
# 下载管理器
# =============================================================================

class DatasetDownloader:
    """数据集下载管理器"""
    
    def __init__(
        self,
        download_path: str,
        logger: logging.Logger,
        max_workers: int = DEFAULT_MAX_WORKERS,
        max_retries: int = MAX_RETRIES,
        retry_delay: int = RETRY_DELAY,
        hf_token: Optional[str] = None,
        use_rich: bool = RICH_AVAILABLE
    ):
        self.download_path = Path(download_path)
        self.logger = logger
        self.max_workers = max_workers
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.stats = DownloadStats()
        self.use_rich = use_rich
        
        self.download_path.mkdir(parents=True, exist_ok=True)
        self.api = HfApi()
        
        if hf_token:
            try:
                login(token=hf_token)
            except Exception as e:
                self.logger.warning(f"登录失败：{e}")
    
    def _download_single_dataset(self, config: DatasetConfig) -> Tuple[bool, int]:
        """下载单个数据集 (带重试逻辑)"""
        repo_id = config.repo_id
        subset = config.subsets[0] if config.subsets else None
        display_name = f"{repo_id} ({subset})" if subset else repo_id
        
        # 构建目标目录
        if subset:
            target_dir = self.download_path / repo_id.split("/")[-1] / subset
        else:
            target_dir = self.download_path / repo_id.split("/")[-1]
        
        # 检查是否已下载
        if target_dir.exists():
            marker_file = target_dir / ".huggingface_download_complete"
            if marker_file.exists():
                self.logger.info(f"[跳过] {display_name} - 已存在完整下载")
                self.stats.update_dataset_status(
                    repo_id, status="skipped", progress=100.0
                )
                self.stats.add_skipped()
                return True, 0
        
        # 更新状态为下载中
        self.stats.update_dataset_status(
            repo_id, status="downloading", start_time=time.time()
        )
        
        # 重试逻辑
        for attempt in range(1, self.max_retries + 1):
            try:
                self.logger.info(f"[下载中] {display_name} - 尝试 {attempt}/{self.max_retries}")
                self.stats.update_dataset_status(repo_id, retries=attempt - 1)
                
                start_time = time.time()
                
                download_kwargs = {
                    "repo_id": repo_id,
                    "repo_type": "dataset",
                    "local_dir": str(target_dir),
                    "revision": config.revision,
                    "max_workers": self.max_workers,
                    "force_download": False,
                    "resume_download": True,
                }
                
                if subset:
                    download_kwargs["subfolder"] = subset
                
                if config.include_patterns:
                    download_kwargs["allow_patterns"] = config.include_patterns
                
                if config.exclude_patterns:
                    download_kwargs["ignore_patterns"] = config.exclude_patterns
                
                snapshot_download(**download_kwargs)
                
                elapsed = time.time() - start_time
                
                total_size = sum(
                    f.stat().st_size for f in target_dir.rglob("*") if f.is_file()
                )
                
                self.logger.info(
                    f"[完成] {display_name} - "
                    f"耗时：{elapsed:.1f}s, 大小：{total_size / (1024**3):.2f} GB"
                )
                
                marker_file.touch()
                
                speed = (total_size / (1024 * 1024)) / elapsed if elapsed > 0 else 0
                
                self.stats.update_dataset_status(
                    repo_id,
                    status="completed",
                    progress=100.0,
                    size_downloaded=total_size,
                    speed=speed,
                    elapsed=elapsed
                )
                
                self.stats.add_success(total_size)
                return True, total_size
                
            except RepositoryNotFoundError as e:
                self.logger.error(f"[错误] 仓库不存在：{repo_id} - {e}")
                self.stats.update_dataset_status(
                    repo_id, status="failed", error_msg=str(e)
                )
                self.stats.add_failed()
                return False, 0
                
            except RevisionNotFoundError as e:
                self.logger.error(f"[错误] 版本不存在：{config.revision} - {e}")
                self.stats.update_dataset_status(
                    repo_id, status="failed", error_msg=str(e)
                )
                self.stats.add_failed()
                return False, 0
                
            except Exception as e:
                self.logger.warning(f"[重试] {display_name} - 错误：{e}")
                
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay * attempt)
                else:
                    self.logger.error(f"[失败] {display_name} - 达到最大重试次数")
                    self.stats.update_dataset_status(
                        repo_id, status="failed", error_msg=str(e)
                    )
                    self.stats.add_failed()
                    return False, 0
        
        return False, 0
    
    def download_all(
        self,
        datasets: List[DatasetConfig],
        sequential: bool = False
    ) -> DownloadStats:
        """下载所有数据集"""
        self.stats.total_datasets = len(datasets)
        sorted_datasets = sorted(datasets, key=lambda x: x.priority)
        
        # 初始化所有数据集状态
        for ds in sorted_datasets:
            self.stats.update_dataset_status(ds.repo_id, config=ds)
        
        if self.use_rich:
            return self._download_with_rich(sorted_datasets, sequential)
        else:
            return self._download_basic(sorted_datasets, sequential)
    
    def _download_with_rich(
        self,
        datasets: List[DatasetConfig],
        sequential: bool
    ) -> DownloadStats:
        """使用 Rich TUI 界面下载"""
        dashboard = DownloadDashboard(self.stats)
        
        with Live(dashboard, refresh_per_second=4, screen=True) as live:
            if sequential:
                for config in datasets:
                    dashboard.stats.update_dataset_status(
                        config.repo_id, status="downloading"
                    )
                    self._download_single_dataset(config)
            else:
                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    futures = {
                        executor.submit(self._download_single_dataset, config): config
                        for config in datasets
                    }
                    
                    for future in as_completed(futures):
                        config = futures[future]
                        try:
                            future.result()
                        except Exception as e:
                            self.logger.error(f"[异常] {config.repo_id}: {e}")
        
        self._print_summary()
        return self.stats
    
    def _download_basic(
        self,
        datasets: List[DatasetConfig],
        sequential: bool
    ) -> DownloadStats:
        """基础模式下载"""
        print(f"\n{Colors.CYAN}开始下载 {len(datasets)} 个数据集{Colors.END}")
        print(f"{Colors.CYAN}下载路径：{self.download_path}{Colors.END}")
        print(f"{Colors.CYAN}并发数：{self.max_workers}{Colors.END}\n")
        
        if sequential:
            for i, config in enumerate(datasets, 1):
                print(f"\n{Colors.YELLOW}[{i}/{len(datasets)}] {config.repo_id}{Colors.END}")
                self._download_single_dataset(config)
        else:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {
                    executor.submit(self._download_single_dataset, config): config
                    for config in datasets
                }
                
                for i, future in enumerate(as_completed(futures), 1):
                    config = futures[future]
                    try:
                        success, _ = future.result()
                        status = f"{Colors.GREEN}✓{Colors.END}" if success else f"{Colors.RED}✗{Colors.END}"
                        print(f"[{i}/{len(datasets)}] {status} {config.repo_id}")
                    except Exception as e:
                        print(f"[{i}/{len(datasets)}] {Colors.RED}✗{Colors.END} {config.repo_id}: {e}")
        
        self._print_summary()
        return self.stats
    
    def _print_summary(self):
        """打印下载摘要"""
        elapsed = self.stats.get_elapsed_time()
        speed = self.stats.get_speed_mbps()
        
        if RICH_AVAILABLE:
            console = Console()
            summary = Table(title="📊 下载完成摘要", box=box.DOUBLE)
            summary.add_column("项目", style="cyan")
            summary.add_column("数值", style="green")
            
            summary.add_row("总数据集数", str(self.stats.total_datasets))
            summary.add_row("成功", f"{self.stats.successful}")
            summary.add_row("失败", f"{self.stats.failed}")
            summary.add_row("跳过 (已存在)", f"{self.stats.skipped}")
            summary.add_row("总下载量", f"{self.stats.total_bytes / (1024**4):.2f} TB")
            summary.add_row("总耗时", f"{elapsed:.1f}s ({elapsed/3600:.2f} 小时)")
            summary.add_row("平均速度", f"{speed:.2f} MB/s")
            
            console.print(summary)
        else:
            print(f"\n{Colors.BOLD}{'=' * 60}{Colors.END}")
            print(f"{Colors.BOLD}下载完成摘要{Colors.END}")
            print(f"{Colors.BOLD}{'=' * 60}{Colors.END}")
            print(f"总数据集数：{self.stats.total_datasets}")
            print(f"成功：{self.stats.successful}")
            print(f"失败：{self.stats.failed}")
            print(f"跳过 (已存在): {self.stats.skipped}")
            print(f"总下载量：{self.stats.total_bytes / (1024**4):.2f} TB")
            print(f"总耗时：{elapsed:.1f}s ({elapsed/3600:.2f} 小时)")
            print(f"平均速度：{speed:.2f} MB/s")
            print(f"{'=' * 60}\n")


# =============================================================================
# 配置向导
# =============================================================================

def config_wizard() -> Dict:
    """交互式配置向导"""
    console = Console() if RICH_AVAILABLE else None
    
    if console:
        console.print(Panel.fit(
            "🔧 配置向导\nConfiguration Wizard",
            style="bold yellow"
        ))
    
    config = {
        "output": DEFAULT_DOWNLOAD_PATH,
        "workers": DEFAULT_MAX_WORKERS,
        "token": None,
        "sequential": False,
        "skip_disk_check": False,
    }
    
    # 下载路径
    if console:
        console.print("\n[bold cyan]1. 设置下载路径[/bold cyan]")
    else:
        print(f"\n{Colors.CYAN}1. 设置下载路径{Colors.END}")
    
    default_path = DEFAULT_DOWNLOAD_PATH
    user_input = input(f"   下载路径 [{default_path}]: ").strip()
    if user_input:
        config["output"] = user_input
    
    # 并发数
    if console:
        console.print("\n[bold cyan]2. 设置并发下载数[/bold cyan]")
    else:
        print(f"\n{Colors.CYAN}2. 设置并发下载数{Colors.END}")
    
    user_input = input(f"   并发数 [{DEFAULT_MAX_WORKERS}]: ").strip()
    if user_input and user_input.isdigit():
        config["workers"] = min(int(user_input), 32)
    
    # 顺序下载
    if console:
        console.print("\n[bold cyan]3. 下载模式[/bold cyan]")
    else:
        print(f"\n{Colors.CYAN}3. 下载模式{Colors.END}")
    
    user_input = input("   是否使用顺序下载 (y/N): ").strip().lower()
    config["sequential"] = user_input in ('y', 'yes')
    
    # Token
    if console:
        console.print("\n[bold cyan]4. Hugging Face Token (可选)[/bold cyan]")
    else:
        print(f"\n{Colors.CYAN}4. Hugging Face Token (可选){Colors.END}")
    
    user_input = input("   Token: ").strip()
    if user_input:
        config["token"] = user_input
    
    return config


# =============================================================================
# 主函数
# =============================================================================

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="🎓 RNewmind 教育 AI 助教数据集下载工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 交互模式 (推荐)
  python download_datasets.py --interactive

  # 基本用法 (使用默认配置)
  python download_datasets.py

  # 指定下载路径
  python download_datasets.py --output /mnt/data/RNewmind_Datasets

  # 指定并发数
  python download_datasets.py --workers 16

  # 顺序下载 (更稳定)
  python download_datasets.py --sequential

  # 使用 Hugging Face Token
  python download_datasets.py --token hf_xxxxx

  # 仅下载特定数据集
  python download_datasets.py --repos "HuggingFaceTB/cosmopedia" "OpenAssistant/oasst1"

  # 调试模式
  python download_datasets.py --debug
        """
    )
    
    parser.add_argument(
        "-i", "--interactive",
        action="store_true",
        help="交互模式 (显示数据集选择菜单和配置向导)"
    )
    
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=DEFAULT_DOWNLOAD_PATH,
        help=f"下载目标路径 (默认：{DEFAULT_DOWNLOAD_PATH})"
    )
    
    parser.add_argument(
        "-w", "--workers",
        type=int,
        default=DEFAULT_MAX_WORKERS,
        help=f"最大并发下载数 (默认：{DEFAULT_MAX_WORKERS})"
    )
    
    parser.add_argument(
        "-t", "--token",
        type=str,
        default=None,
        help="Hugging Face API Token (可选)"
    )
    
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="顺序下载 (禁用并发，更稳定)"
    )
    
    parser.add_argument(
        "--skip-disk-check",
        action="store_true",
        help="跳过磁盘空间检查"
    )
    
    parser.add_argument(
        "--min-disk-gb",
        type=int,
        default=MIN_DISK_SPACE_GB,
        help=f"最小可用磁盘空间 GB (默认：{MIN_DISK_SPACE_GB})"
    )
    
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="日志级别 (默认：INFO)"
    )
    
    parser.add_argument(
        "--log-dir",
        type=str,
        default="./logs",
        help="日志目录 (默认：./logs)"
    )
    
    parser.add_argument(
        "--repos",
        type=str,
        nargs="+",
        default=None,
        help="仅下载指定的数据集 (空格分隔 repo_id)"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="调试模式 (等同于 --log-level DEBUG)"
    )
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    # 打印横幅
    print_banner()
    
    # 交互模式
    if args.interactive:
        # 配置向导
        config = config_wizard()
        args.output = config["output"]
        args.workers = config["workers"]
        args.sequential = config["sequential"]
        if config["token"]:
            args.token = config["token"]
        
        # 数据集选择
        selected_datasets = interactive_select(DATASETS)
    else:
        # 筛选数据集
        selected_datasets = DATASETS
        if args.repos:
            selected_datasets = [d for d in DATASETS if d.repo_id in args.repos]
            if not selected_datasets:
                print(f"{Colors.RED}未找到匹配的数据集：{args.repos}{Colors.END}")
                sys.exit(1)
    
    # 设置日志级别
    log_level = "DEBUG" if args.debug else args.log_level
    
    # 初始化日志
    logger = setup_logging(args.log_dir, log_level)
    logger.info("RNewmind 数据集下载器启动")
    
    # 磁盘空间检查
    if not args.skip_disk_check:
        sufficient, info = check_disk_space(args.output, args.min_disk_gb)
        
        if "error" in info:
            print(f"{Colors.YELLOW}⚠ 磁盘检查失败：{info['error']}{Colors.END}")
        else:
            if RICH_AVAILABLE:
                console = Console()
                status_style = "green" if sufficient else "red"
                status_icon = "✓" if sufficient else "✗"
                console.print(f"[{status_style}] {status_icon} 磁盘空间："
                             f"总计 {info['total_gb']}GB | "
                             f"已用 {info['used_gb']}GB | "
                             f"可用 {info['free_gb']}GB[/{status_style}]")
            else:
                print(f"{Colors.CYAN}磁盘空间：总计 {info['total_gb']}GB, "
                     f"已用 {info['used_gb']}GB, 可用 {info['free_gb']}GB{Colors.END}")
            
            if not sufficient:
                print(f"{Colors.RED}❌ 磁盘空间不足! 需要 {args.min_disk_gb}GB, "
                     f"可用 {info['free_gb']}GB{Colors.END}")
                print(f"{Colors.YELLOW}提示：使用 --skip-disk-check 跳过此检查或清理磁盘空间{Colors.END}")
                sys.exit(1)
    
    # 创建下载器
    downloader = DatasetDownloader(
        download_path=args.output,
        logger=logger,
        max_workers=args.workers,
        hf_token=args.token,
        use_rich=RICH_AVAILABLE
    )
    
    # 开始下载
    try:
        stats = downloader.download_all(
            selected_datasets,
            sequential=args.sequential
        )
        
        if stats.failed > 0:
            print(f"{Colors.YELLOW}⚠ 有 {stats.failed} 个数据集下载失败{Colors.END}")
            sys.exit(1)
        
        if RICH_AVAILABLE:
            console = Console()
            console.print("\n[bold green]✓ 所有数据集下载完成![/bold green]")
        else:
            print(f"{Colors.GREEN}✓ 所有数据集下载完成!{Colors.END}")
        
        sys.exit(0)
        
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}用户中断下载{Colors.END}")
        sys.exit(130)
    except Exception as e:
        print(f"{Colors.RED}未知错误：{e}{Colors.END}")
        sys.exit(1)


if __name__ == "__main__":
    main()
