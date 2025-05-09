import traceback
from errno import EACCES

import numpy as np
import pandas as pd
import re
import seaborn as sns
from textwrap import shorten

import matplotlib as mpl

mpl.use('Qt5Agg')

from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

import sys
import tempfile
import subprocess
from PyQt5.QtWidgets import (QMainWindow, QApplication, QTabWidget, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QComboBox, QPushButton, QFileDialog, QFrame, QTableView,
                             QWidget, QVBoxLayout, QScrollArea, QSizePolicy, QProgressDialog)
from PyQt5.QtCore import QSettings, Qt, pyqtSignal, QThread
from PyQt5.QtGui import QStandardItemModel, QStandardItem

mpl.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei']  # 多个中文字体备选
mpl.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

print(plt.style.available)
plt.style.use('seaborn-v0_8')


symbol_type_map = {
    # 代码段符号
    'T': 'Global text symbol (executable code)',
    't': 'Local text symbol (executable code)',

    # 数据段符号
    'D': 'Global initialized data',
    'd': 'Local initialized data',
    'B': 'Global uninitialized data (BSS)',
    'b': 'Local uninitialized data (BSS)',

    # 只读数据
    'R': 'Global read-only data',
    'r': 'Local read-only data',

    # 特殊类型
    'U': 'Undefined symbol',
    'V': 'Global weak object symbol',
    'W': 'Global weak symbol (not tagged as object)',
    'w': 'Local weak symbol',

    # 其他类型
    'C': 'Common symbol',
    'A': 'Absolute symbol',
    'S': 'Global small object',
    's': 'Local small object',
    'G': 'Global optimized/grouped data',
    'g': 'Local optimized/grouped data',
    'N': 'Debugging symbol (noreturn)',

    # 特殊场景
    'I': 'Indirect symbol (PLT entry)',
    'i': 'Local indirect symbol',
    'P': 'Global procedure linkage table',
    'p': 'Local procedure linkage table',
    'F': 'Global file symbol',
    'f': 'Local file symbol',

    # 废弃类型（部分旧系统）
    '?': 'Unknown symbol type',
    'Z': 'Global zero-initialized (HP-UX)',
    'z': 'Local zero-initialized (HP-UX)'
}


def parse_nm_output(file_path):
    data = []
    current_file = None
    pattern = re.compile(
        r'^\s*'  # 起始空格
        r'([0-9a-fA-F]{8})\s+'  # 地址（8位十六进制）
        r'([0-9a-fA-F]{8})\s+'  # 大小（8位十六进制）
        r'([A-Za-z])\s+'  # 符号类型（单个字母）
        r'(.+)$'  # 符号名（包含空格）
    )

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()

            # 处理文件名行
            if line.endswith(':'):
                current_file = line[:-1]
                continue

            # 跳过空行
            if not line:
                continue

            # 解析符号行
            match = pattern.match(line)
            if match:
                address, size_hex, sym_type, symbol = match.groups()

                # 转换数值类型
                try:
                    size = int(size_hex, 16)
                    address = int(address, 16)
                except ValueError:
                    continue

                data.append({
                    'Filename': current_file,
                    'Address': f"0x{address:08x}",
                    'Size': size,
                    'Type': sym_type,
                    'Symbol': symbol
                })
            else:
                print(f"无法解析的行: {line}")

    return pd.DataFrame(data)


def plot_file_sizes(df):
    """带数据标签的文件大小分布图"""
    # 数据处理
    file_sizes = df.groupby('Filename')['Size'].sum().sort_values()

    # 创建图表
    fig, ax = plt.subplots(figsize=(12, max(6, len(file_sizes) * 0.4)))
    bars = ax.barh(file_sizes.index, file_sizes.values, color='steelblue')

    # 自动调整X轴范围
    max_size = file_sizes.max()
    ax.set_xlim(right=max_size * 1.15)  # 为标签留出空间

    # 添加数据标签
    for bar in bars:
        width = bar.get_width()
        ax.text(width + max_size * 0.02,  # 偏移量设为最大值的2%
                bar.get_y() + bar.get_height() / 2,
                f'{width:,}',  # 添加千分位分隔符
                va='center',
                ha='left',
                fontsize=8)

    plt.title('File Size Distribution with Value Labels')
    plt.xlabel('Total Size (bytes)')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


def plot_top_symbols(df):
    """带数据标签的Top符号分布图"""
    # 数据处理
    top_symbols = df.nlargest(80, 'Size').sort_values('Size')

    # 创建图表
    fig, ax = plt.subplots(figsize=(12, 16))
    bars = ax.barh(top_symbols['Symbol'], top_symbols['Size'], color='forestgreen')

    # 设置标签参数
    max_size = top_symbols['Size'].max()
    ax.set_xlim(right=max_size * 1.2)  # 扩展X轴范围

    # 添加数据标签
    for bar in bars:
        width = bar.get_width()
        ax.text(width + max_size * 0.02,
                bar.get_y() + bar.get_height() / 2,
                f'{width:,}',
                va='center',
                ha='left',
                fontsize=8)

    plt.title('Top 80 Symbols with Value Labels')
    plt.xlabel('Symbol Size (bytes)')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


def plot_nm_output(output_file: str):
    """
    Command reference: `nm --print-size --size-sort --demangle lib.a > output_file`
    :param output_file: Output file name
    :return: None
    """
    df = parse_nm_output(output_file)
    df = df.sort_values('Size', ascending=False)

    print(f'Total symbol count = {len(df)}')
    print(f'Total symbol size = {df["Size"].sum()}')

    plot_file_sizes(df)
    plot_top_symbols(df)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.df = None  # 存储分析结果

        # 加载历史记录
        self.nm_combo.load_history()
        self.elf_combo.load_history()

    def init_ui(self):
        self.setWindowTitle('ELF符号空间分析工具')
        self.setGeometry(200, 200, 1280, 720)

        # 主布局
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)

        # 第一行：nm路径选择
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("nm路径:"), 1)
        self.nm_combo = HistoryComboBox('SleepySoft', 'ParseMap', 'nm')
        self.nm_combo.setEditable(True)
        self.nm_combo.setToolTip("选择nm工具路径\n示例：C:/NXP/.../arm-none-eabi-nm")
        row1.addWidget(self.nm_combo, 8)
        row1.addWidget(QPushButton("浏览", clicked=self.select_nm), 1)

        # 第二行：ELF文件选择
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("目标文件:"), 1)
        self.elf_combo = HistoryComboBox('SleepySoft', 'ParseMap', 'binary')
        self.elf_combo.setToolTip("选择要分析的ELF文件或库文件")
        row2.addWidget(self.elf_combo, 7)
        row2.addWidget(QPushButton("浏览", clicked=self.select_elf), 1)
        self.analyze_btn = QPushButton("分析", clicked=self.start_analysis)
        row2.addWidget(self.analyze_btn, 1)

        # Tab区域
        self.tabs = QTabWidget()
        self.list_tab = SymbolListView()
        self.plot_tab = StatisticsView()
        self.tabs.addTab(self.list_tab, "符号列表")
        self.tabs.addTab(self.plot_tab, "统计分析")

        # 组装布局
        main_layout.addLayout(row1)
        main_layout.addLayout(row2)
        main_layout.addWidget(QFrame(frameShape=QFrame.HLine))
        main_layout.addWidget(self.tabs, stretch=1)

    def select_nm(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择nm工具",
            self.nm_combo.currentText(),  # 从当前路径开始浏览
            "可执行文件 (nm* arm-none-eabi-nm* *.exe);;所有文件 (*)"
        )
        if path:
            # 转换Windows路径分隔符
            self.nm_combo.add_history(path.replace("\\", "/"))

    def select_elf(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择目标文件",
            self.elf_combo.currentText(),
            "二进制文件 (*.elf *.a *.so *.o);;所有文件 (*)"
        )
        if path:
            # 处理带空格的路径
            self.elf_combo.add_history(f'"{path}"' if " " in path else path)

    def start_analysis(self):
        nm_path = self.nm_combo.currentText().strip()
        elf_path = self.elf_combo.currentText().strip()

        if not nm_path or not elf_path:
            self.statusBar().showMessage("请先选择nm路径和目标文件", 5000)
            return

        try:
            # 执行nm命令
            cmd = f'"{nm_path}" --print-size --size-sort --demangle "{elf_path}"'
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)

            # 解析输出
            with tempfile.NamedTemporaryFile(mode='w+', delete=False) as tmp:
                tmp.write(result.stdout)
                tmp.seek(0)
                self.df = parse_nm_output(tmp.name)

            # 更新视图
            self.list_tab.update_data(self.df)
            self.plot_tab.update_data(self.df)
            self.statusBar().showMessage(f"分析完成，共找到 {len(self.df)} 个符号", 5000)

        except subprocess.CalledProcessError as e:
            print(str(e))
            self.statusBar().showMessage(f"命令执行失败: {e.stderr}", 10000)
        except Exception as e:
            print(str(e))
            self.statusBar().showMessage(f"分析错误: {str(e)}", 10000)


from PyQt5.QtCore import QSettings, QStandardPaths


class HistoryComboBox(QComboBox):
    def __init__(self, org_name: str, app_name: str, config_group="History", max_items=10):
        super().__init__()
        self.max_items = max_items
        self.config_group = config_group
        self.settings = QSettings(
            QSettings.IniFormat,
            QSettings.UserScope,
            org_name,
            app_name
        )
        self.settings.setIniCodec("UTF-8")
        self.load_history()

    def add_history(self, path):
        self.insertItem(0, path)
        self.setCurrentIndex(0)

        # 保持最大条目数
        while self.count() > self.max_items:
            self.removeItem(self.count() - 1)

        self.save_history()

    def load_history(self):
        """跨平台安全加载"""
        self.settings.beginGroup(self.config_group)
        history = self.settings.value("paths", [], type=list)  # 明确指定类型
        self.clear()
        self.addItems([str(p) for p in history if p])  # 过滤空值
        self.settings.endGroup()

    def save_history(self):
        """原子化写入"""
        self.settings.beginGroup(self.config_group)
        self.settings.setValue("paths", [self.itemText(i) for i in range(self.count())])
        self.settings.endGroup()
        self.settings.sync()  # 强制写入磁盘

    def select_nm(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择nm工具",
            self.nm_combo.currentText(),  # 从当前路径开始浏览
            "可执行文件 (nm* arm-none-eabi-nm* *.exe);;所有文件 (*)"
        )
        if path:
            # 转换Windows路径分隔符
            self.nm_combo.add_history(path.replace("\\", "/"))

    def select_elf(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择目标文件",
            self.elf_combo.currentText(),
            "二进制文件 (*.elf *.a *.so *.o);;所有文件 (*)"
        )
        if path:
            # 处理带空格的路径
            self.elf_combo.add_history(f'"{path}"' if " " in path else path)


class SymbolListView(QWidget):
    """符号列表视图"""

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        self.table = QTableView()
        layout.addWidget(self.table)

    def update_data(self, df):
        model = QStandardItemModel()
        model.setHorizontalHeaderLabels(['文件名', '地址', '大小', '类型', '符号'])

        for _, row in df.iterrows():
            items = [
                QStandardItem(row['Filename']),
                QStandardItem(row['Address']),
                QStandardItem(str(row['Size'])),
                QStandardItem(row['Type']),
                QStandardItem(row['Symbol'])
            ]
            model.appendRow(items)

        self.table.setModel(model)
        self.table.resizeColumnsToContents()


class StatisticsView(QScrollArea):
    def __init__(self):
        super().__init__()
        # Add debug code to verify font configuration
        self.chart_type = None
        print("Current font configuration:", mpl.rcParams['font.sans-serif'])
        print("Available Chinese fonts:", [f.name for f in mpl.font_manager.fontManager.ttflist if 'Hei' in f.name])

        self.figure_pool = []  # Figure object pool
        self.df = pd.DataFrame()
        self._init_ui()
        self.progress_dialog = None  # Loading progress dialog

    def _init_ui(self):
        # Scroll area configuration
        self.setWidgetResizable(True)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        # Main container
        self.container = QWidget()
        self.container.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.container.setMinimumSize(800, 0)  # 允许垂直扩展
        self.container.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.MinimumExpanding  # 容器随内容增长
        )
        self.setWidget(self.container)

        # Main layout
        self.layout = QVBoxLayout(self.container)
        self.layout.setContentsMargins(20, 20, 20, 20)
        self.layout.setSpacing(30)
        self.layout.setAlignment(Qt.AlignTop)

        print(f"滚动区域尺寸策略: {self.sizePolicy().horizontalPolicy()},{self.sizePolicy().verticalPolicy()}")
        print(f"容器尺寸限制: min={self.container.minimumSize()}, max={self.container.maximumSize()}")
        print(
            f"滚动条状态: vertical={self.verticalScrollBar().isVisible()}, horizontal={self.horizontalScrollBar().isVisible()}")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # 强制计算滚动条
        self.verticalScrollBar().setVisible(True)  # 始终显示
        self.horizontalScrollBar().setVisible(False)
        # 更新策略
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)

    def update_data(self, df: pd.DataFrame, chart_type='both'):
        """Update data and generate specified charts
        Args:
            chart_type: 'file_size' | 'symbols' | 'both'
            :param chart_type:
            :param df:
        """
        self._show_loading()

        try:
            self.df = self._pre_process_data(df)
            self.chart_type = chart_type.lower()
            self._async_generate_charts()
        except Exception as e:
            print(f'Chart plt fail: {str(e)}')
            print(traceback.format_exc())
        finally:
            self._hide_loading()

    def _pre_process_data(self, df: pd.DataFrame):
        try:
            # Type conversion (safe mode)
            self.df['Size'] = pd.to_numeric(self.df['Size'], errors='coerce')
            valid_df = self.df.dropna(subset=['Size']).copy()

            # Debug output
            print(f"[Debug] Valid data count after preprocessing: {len(valid_df)}")
            print(f"[Debug] Size distribution:\n{valid_df['Size'].describe()}")

            return valid_df
        except Exception as e:
            print(f'Preprocess data error: {str(e)}')
            return df.copy()

    def _async_generate_charts(self):
        QApplication.processEvents()
        self._clear_charts()

        # self._create_dynamic_chart('symbols')

        if self.chart_type in ('file_size', 'both'):
            self._create_dynamic_chart('file_size')
        if self.chart_type in ('symbols', 'both'):
            self._create_dynamic_chart('symbols')
        if self.chart_type in ('type', 'both'):
            self._create_dynamic_chart('type')

    def _create_dynamic_chart(self, chart_type):
        """Complete solution for empty grouping results"""
        try:
            if self.df.empty:
                self._add_placeholder("Empty data")
                return
            try:
                if chart_type == 'file_size':
                    display_data, y_labels, title = self._data_analysis_file_size(self.df)
                    plot = self._plot_column_chart
                    params = (display_data, y_labels, title, chart_type)
                elif chart_type == 'symbols':
                    display_data, y_labels, title = self._data_analysis_symbol(self.df)
                    plot = self._plot_column_chart
                    params = (display_data, y_labels, title, chart_type)
                elif chart_type == 'type':
                    display_data, labels, title = self._data_analysis_symbol_type(self.df)
                    plot = self._plot_pie_chart
                    params = (display_data, labels, title)
                else:
                    raise ValueError(f"Unknown chart type: {chart_type}")
            except Exception as e:
                self._add_placeholder(f"Aggregation failed: {str(e)}")
                return
            if display_data.empty:
                self._add_placeholder("No valid data to display")
                return

            plot(*params)

        except KeyError as ke:
            self._add_placeholder(f"Missing required field: {str(ke)}")
        except ValueError as ve:
            self._add_placeholder(f"Data format error: {str(ve)}")
        except Exception as e:
            self._add_placeholder(f"Rendering exception: {str(e)}")
            import traceback
            traceback.print_exc()

    def _plot_column_chart(self, display_data, y_labels, title, chart_type):
        # Dynamic size calculation
        fig_height = self._calculate_fig_height(y_labels)
        fig = Figure(figsize=(12, fig_height), dpi=100)
        self.figure_pool.append(fig)
        ax = fig.add_subplot(111)

        # Core plotting logic
        max_value = display_data.max()
        bars = ax.barh(y_labels, display_data.values,
                       color=self._get_chart_color(chart_type))

        # Style configuration
        ax.set_xlim(right=max_value * 1.2)
        ax.set_title(title, fontsize=12, pad=15)
        ax.set_xlabel("Occupied Space", fontsize=10)
        ax.grid(axis='x', linestyle='--', alpha=0.7)

        # Dynamic label layout
        for bar in bars:
            width = bar.get_width()
            ax.text(width + max_value * 0.02,
                    bar.get_y() + bar.get_height() / 2,
                    f"{width:,}",
                    va='center',
                    ha='left',
                    fontsize=8,
                    fontfamily='SimHei')
        # Embed canvas
        self._embed_canvas(fig, v_policy=QSizePolicy.Expanding)

    def _plot_pie_chart(self, display_data, labels, title):
        """饼状图绘制逻辑（新增方法）"""
        fig = Figure(figsize=(8, 6), dpi=100)
        self.figure_pool.append(fig)
        ax = fig.add_subplot(111)

        # 自动颜色生成
        colors = plt.get_cmap('tab20c')(np.linspace(0, 1, len(labels)))

        # 核心绘图参数（参考网页1、4、5）
        wedges, texts, autotexts = ax.pie(
            display_data,
            labels=labels,
            autopct=lambda pct: f'{pct:.1f}%\n({int(pct * sum(display_data) / 100):,}B)',
            startangle=140,
            colors=colors,
            wedgeprops=dict(width=0.3, edgecolor='w'),
            textprops={'fontsize': 8, 'fontfamily': 'SimHei'}
        )

        # 样式优化（参考网页5、8）
        ax.set_title(title, fontsize=12, pad=20, fontfamily='SimHei')
        ax.axis('equal')

        # 图例显示优化
        legend = ax.legend(
            wedges,
            labels,
            title="Types",
            loc="center left",
            bbox_to_anchor=(1, 0, 0.5, 1),
            prop={'family': 'SimHei', 'size': 8}
        )
        legend.get_title().set_fontproperties({'family': 'SimHei', 'size': 10})

        self._embed_canvas(fig, v_policy=QSizePolicy.Fixed)

    def _data_analysis_file_size(self, df: pd.DataFrame):
        # Filename validity check
        df['Filename'] = df['Filename'].fillna('UNKNOWN')
        invalid_names = df['Filename'].str.strip().eq('')
        if invalid_names.any():
            print(f"[Debug] Found {invalid_names.sum()} empty filenames")
            df.loc[invalid_names, 'Filename'] = 'UNKNOWN'

        # Group aggregation (simplified)
        grouped = df.groupby('Filename')['Size'].sum()
        if grouped.empty:
            self._add_placeholder("No data after grouping (check filenames)")
            return

        # display_data = grouped.nlargest(15).sort_values(ascending=True)
        display_data = grouped.sort_values(ascending=True)
        print(f"[Debug] Grouping results:\n{display_data.head()}")

        y_labels = display_data.index.astype(str)  # Explicitly get labels <-- Fix point

        # Debug output
        print(f"[Debug] File label samples: {y_labels[:3]}")

        title = "File Size Distribution (Bytes)"

        return display_data, y_labels, title

    def _data_analysis_symbol(self, df: pd.DataFrame, limit: int = 150):
        # Symbol distribution logic
        if 'Symbol' not in df.columns:
            raise KeyError('Symbol')

        # Filter processing
        filtered = df[['Symbol', 'Size']].nlargest(limit, 'Size')
        # filtered = df[['Symbol', 'Size']]
        sorted_data = filtered.sort_values('Size', ascending=True)

        # Format conversion
        display_data = sorted_data.set_index('Symbol')['Size']
        y_labels = display_data.index.astype(str)
        title = f"TOP {limit} Symbol Size Distribution (Bytes)"

        return display_data, y_labels, title

    def _data_analysis_symbol_type(self, df: pd.DataFrame):
        """按Type分类统计（新增核心方法）"""
        if 'Type' not in df.columns:
            raise KeyError('Type field not found')

        # 数据预处理
        df['Type'] = df['Type'].fillna('UNKNOWN').str.strip()
        df.loc[df['Type'] == '', 'Type'] = 'UNKNOWN'

        # 按Type聚合
        grouped = df.groupby('Type')['Size'].sum()
        if grouped.empty:
            return pd.Series(dtype=float), [], "Type Distribution"

        # 过滤过小项（优化显示）
        threshold = grouped.sum() * 0.01  # 小于1%的合并为Other
        main_types = grouped[grouped >= threshold]
        other_size = grouped[grouped < threshold].sum()

        if not main_types.empty and other_size > 0:
            main_types['Other'] = other_size

        display_data = main_types.sort_values(ascending=False)
        return display_data, display_data.index.tolist(), "Type Size Distribution"

    # 动态计算高度（按标签密度）
    def _calculate_fig_height(self, labels, font_size=8, dpi=100):
        # 每个标签需要约40像素（0.4英寸*100dpi）
        base_height_inch = len(labels) * 0.4
        # 兼容高DPI屏幕（如缩放200%则自动翻倍）
        screen_dpi = self.screen().logicalDotsPerInch()
        scaled_height = base_height_inch * (screen_dpi / 100)
        return max(6, scaled_height)  # 最低保障6英寸

    def _get_chart_color(self, chart_type):
        """Get color mapping for chart types"""
        return {
            'file_size': '#1f77b4',  # Steel blue
            'symbols': '#2ca02c'  # Forest green
        }.get(chart_type, '#444444')

    def _show_loading(self):
        """Show loading dialog"""
        self.progress_dialog = QProgressDialog("Generating charts...", None, 0, 0, self)
        self.progress_dialog.setWindowModality(Qt.WindowModal)
        self.progress_dialog.setCancelButton(None)
        self.progress_dialog.show()

    def _hide_loading(self):
        """Close loading dialog"""
        if self.progress_dialog:
            self.progress_dialog.close()

    def _clear_charts(self):
        """Safely clear all charts"""
        while self.layout.count():
            item = self.layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        # Release figure resources
        for fig in self.figure_pool:
            fig.clf()
            plt.close(fig)
        self.figure_pool.clear()

    def _embed_canvas(self, fig, v_policy=QSizePolicy.Fixed):
        """通用画布嵌入方法"""
        canvas = FigureCanvas(fig)
        print("[DEBUG] 原始尺寸:", fig.get_size_inches())  # 输出(12,6)
        print("[DEBUG] DPI设置:", fig.dpi)  # 输出100
        print("[DEBUG] Qt建议尺寸:", canvas.sizeHint())  # 应显示1200x600
        print("[DEBUG] 屏幕DPI:", canvas.screen().logicalDotsPerInch())

        canvas.setSizePolicy(QSizePolicy.Expanding, v_policy)
        print("[DEBUG] 实际渲染尺寸:", canvas.size().height())  # 检查实际值

        self.layout.addWidget(canvas)
        canvas.draw_idle()
        print(f"布局项数量: {self.layout.count()}")  # 验证控件添加

        fig.subplots_adjust(
            left=0.4,  # 增加左边距（给长标签留空间）
            right=0.95,  # 减小右边距
            top=0.92,  # 上边距微调
            bottom=0.08,  # 下边距增加
            hspace=0.4  # 子图纵向间距
        )

    def _add_placeholder(self, text, fig=None):
        """添加空数据占位提示"""
        if fig is None:
            fig = Figure(figsize=(8, 3), dpi=100)
            self.figure_pool.append(fig)
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, text,
                ha='center', va='center',
                fontsize=12, color='gray')
        ax.axis('off')
        self._embed_canvas(fig)


if __name__ == '__main__':
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    # QCoreApplication.setOrganizationName("SleepySoft")
    # QCoreApplication.setApplicationName("BinarySizeAnalyzer")
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
