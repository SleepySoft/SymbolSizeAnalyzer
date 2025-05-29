import datetime
import re
import csv
import sys
import tempfile
import subprocess
import traceback
import numpy as np
import pandas as pd

import matplotlib as mpl

mpl.use('Qt5Agg')
mpl.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei']  # 多个中文字体备选
mpl.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from PyQt5.QtGui import QStandardItemModel, QStandardItem
from PyQt5.QtCore import QSettings, Qt
from PyQt5.QtWidgets import (QMainWindow, QApplication, QTabWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QComboBox, QPushButton, QFileDialog, QFrame, QTableView,
                             QWidget, QScrollArea, QSizePolicy, QProgressDialog,
                             QDialog, QCheckBox, QDialogButtonBox, QMessageBox,
                             QTableWidget, QTableWidgetItem, QHeaderView)


print(plt.style.available)
plt.style.use('seaborn-v0_8')


symbol_type_map = {
    # === Code Section Symbols ===
    'T': 'Global text symbol [.text]',
    't': 'Local text symbol [.text]',

    # === Data Section Symbols ===
    'D': 'Global initialized data [.data]',
    'd': 'Local initialized data [.data]',
    'B': 'Global uninitialized data [.bss]',
    'b': 'Local uninitialized data [.bss]',

    # === Read-Only Data ===
    'R': 'Global read-only data [.rodata]',
    'r': 'Local read-only data [.rodata]',

    # === Special Types ===
    'U': 'Undefined symbol [NO SECTION]',
    'V': 'Global weak object symbol [.data/.bss]',
    'W': 'Global weak symbol [.dynsym]',
    'w': 'Local weak symbol [.dynsym]',

    # === Optimization Sections ===
    'S': 'Global small object [.sdata]',
    's': 'Local small object [.sbss]',
    'G': 'Global optimized data [.sdata/.data.gnu]',
    'g': 'Local optimized data [.sbss/.bss.gnu]',

    # === Linker & Debug ===
    'C': 'Common symbol [COMMON]',
    'A': 'Absolute symbol [NO SECTION]',
    'N': 'Debugging symbol [.debug]',

    # === Dynamic Linking ===
    'I': 'Indirect symbol [.plt]',
    'i': 'Local indirect symbol [.plt]',
    'P': 'Global PLT entry [.got.plt]',
    'p': 'Local PLT entry [.got.plt]',

    # === File Metadata ===
    'F': 'Global file symbol [NO SECTION]',
    'f': 'Local file symbol [NO SECTION]',

    # === Legacy/System-Specific ===
    'Z': 'Global zero-init [.zbss] (HP-UX)',
    'z': 'Local zero-init [.zbss] (HP-UX)',
    '?': 'Unknown type [NO SECTION]'
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


import pandas as pd


def symbol_diff(df_current, df_base):
    """
    Compute the size difference between two DataFrames based on 'Symbol'.

    Performs an outer join on 'Symbol', calculates the difference (right - left).
    If a symbol is missing in one side, its size is considered 0.

    Args:
        df_current (pd.DataFrame): Left DataFrame with columns including 'Symbol' and 'Size'.
        df_base (pd.DataFrame): Right DataFrame with columns including 'Symbol' and 'Size'.

    Returns:
        pd.DataFrame: Merged DataFrame with 'diff' column indicating size difference.
    """
    # Prepare df_base by keeping only 'Symbol' and 'Size' (renamed to 'Size_base')
    df_base = df_base[['Symbol', 'Size']].copy()
    df_base.rename(columns={'Size': 'Size_base'}, inplace=True)

    # Perform outer join on 'Symbol'
    merged_df = pd.merge(df_current, df_base, on='Symbol', how='outer')

    # Fill missing sizes with 0 for both sides
    merged_df['Size'] = merged_df['Size'].fillna(0)
    merged_df['Size_base'] = merged_df['Size_base'].fillna(0)

    # Calculate the difference (right - left)
    merged_df['diff'] = merged_df['Size'] - merged_df['Size_base']

    return merged_df


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
        self.df = None
        self.df_filtered = None
        self.symbol_list = list(symbol_type_map.keys())

        self.plot_tab = None
        self.list_tab = None
        self.tabs = None

        self.sym_type_btn = None
        self.analyze_btn = None
        self.nm_combo = None
        self.elf_combo = None

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('Symbol Size Analyzer - By Sleepy - v0.1')
        self.setGeometry(200, 200, 1280, 720)

        # 主布局
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)

        # 第一行：nm路径选择
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("nm:"), 1)
        self.nm_combo = HistoryComboBox('SleepySoft', 'ParseMap', 'nm')
        self.nm_combo.setEditable(True)
        self.nm_combo.setToolTip("Select nm path\nExample：C:/NXP/.../arm-none-eabi-nm")
        row1.addWidget(self.nm_combo, 8)
        row1.addWidget(QPushButton("Browse", clicked=self.select_nm), 1)

        # 第二行：ELF文件选择
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Binary:"), 1)
        self.elf_combo = HistoryComboBox('SleepySoft', 'ParseMap', 'binary')
        self.elf_combo.setToolTip("Select ELF or library for analysis.")
        row2.addWidget(self.elf_combo, 7)
        row2.addWidget(QPushButton("Browse", clicked=self.select_elf), 1)
        self.analyze_btn = QPushButton("Analysis", clicked=self.start_analysis)
        row2.addWidget(self.analyze_btn, 1)

        self.nm_combo.load_history()
        self.elf_combo.load_history()

        # Row 3: Option area
        row3 = QHBoxLayout()
        row3.addWidget(QPushButton('Symbol Type Filter', clicked=self.filter_symbol_type), 1)
        row3.addWidget(QPushButton('Export Table', clicked=self.export_table_content), 1)
        row3.addWidget(QPushButton('Export Chart', clicked=self.export_chart_diagram), 1)
        row3.addWidget(QPushButton('Symbol Diff', clicked=self.show_symbol_diff), 1)
        row3.addWidget(QLabel(), 99)

        # Tab区域
        self.tabs = QTabWidget()
        self.tabs.setTabsClosable(True)
        self.tabs.tabCloseRequested.connect(self.close_tab)
        self.list_tab = SymbolListView()
        self.plot_tab = StatisticsView()
        self.tabs.addTab(self.list_tab, "Symbol List")
        self.tabs.addTab(self.plot_tab, "Symbol Statistics")

        # 组装布局
        main_layout.addLayout(row1)
        main_layout.addLayout(row2)
        main_layout.addLayout(row3)
        main_layout.addWidget(QFrame(frameShape=QFrame.HLine))
        main_layout.addWidget(self.tabs, stretch=1)

    def select_nm(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select nm",
            self.nm_combo.currentText(),  # 从当前路径开始浏览
            "Runnable (nm* arm-none-eabi-nm* *.exe);;All Files (*)"
        )
        if path:
            # 转换Windows路径分隔符
            self.nm_combo.add_history(path.replace("\\", "/"))

    def select_elf(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select binary",
            self.elf_combo.currentText(),
            "Binary (*.elf *.a *.so *.o);;All Files (*)"
        )
        if path:
            # 处理带空格的路径
            self.elf_combo.add_history(f'"{path}"' if " " in path else path)

    def start_analysis(self):
        nm_path = self.nm_combo.currentText().strip()
        elf_path = self.elf_combo.currentText().strip()

        df = self.do_symbol_analysis(nm_path, elf_path)
        if df is not None:
            self.df_filtered = self.df = df
            self.update_view()

    def do_symbol_analysis(self, nm_path: str, elf_path: str) -> pd.DataFrame or None:
        if not nm_path or not elf_path:
            self.statusBar().showMessage("Please select nm path and target file first", 5000)
            return None

        try:
            # 执行nm命令
            cmd = f'"{nm_path}" --print-size --size-sort --demangle "{elf_path}"'
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)

            # 解析输出
            with tempfile.NamedTemporaryFile(mode='w+', delete=False) as tmp:
                tmp.write(result.stdout)
                tmp.seek(0)
                df = parse_nm_output(tmp.name)
            self.statusBar().showMessage(f"分析完成，共找到 {len(df)} 个符号", 5000)
            return df
        except subprocess.CalledProcessError as e:
            print(str(e))
            traceback.print_exc()
            self.statusBar().showMessage(f"命令执行失败: {e.stderr}", 10000)
            return None
        except Exception as e:
            print(str(e))
            traceback.print_exc()
            self.statusBar().showMessage(f"分析错误: {str(e)}", 10000)
            return None

    def filter_symbol_type(self):
        selected, ok = SymbolSelector.get_symbols(
            symbol_map=symbol_type_map,
            initial_selection=self.symbol_list,
            parent=self
        )

        if selected:
            self.symbol_list = selected
            self.df_filtered = self.df[self.df['Type'].isin(self.symbol_list)].copy()
            self.update_view()

    def export_table_content(self):
        self.list_tab.export_to_csv()

    def export_chart_diagram(self):
        self.plot_tab.export_charts()

    def show_symbol_diff(self):
        if self.df is None:
            QMessageBox.information(self, "Diff Error", "No Diff Base.\nPlease analysis one binary first.")
            return

        elf_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select compare binary",
            "",
            "Binary (*.elf *.a *.so *.o);;所有文件 (*)"
        )

        if elf_path:
            nm_path = self.nm_combo.currentText().strip()
            df = self.do_symbol_analysis(nm_path, elf_path)
            if df is not None and not df.empty:
                diff = symbol_diff(self.df, df)
                diff_view = SymbolDiffView(diff, )
                self.tabs.addTab(diff_view, "Symbol Diff ")

    def close_tab(self, index):
        sender_widget = self.sender()
        if index > 1 and isinstance(sender_widget, QTabWidget):
            sender_widget.removeTab(index)

    def update_view(self):
        self.list_tab.update_data(self.df_filtered)
        self.plot_tab.update_data(self.df_filtered)


class SymbolSelector(QDialog):
    def __init__(self, symbol_map, selected_symbols, parent=None):
        super().__init__(parent)
        self.select_all_cb = None
        self.symbol_map = symbol_map
        self.initial_selection = set(selected_symbols)
        self.checkboxes = {}
        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(15, 10, 15, 10)
        main_layout.setSpacing(8)

        self.select_all_cb = QCheckBox("Select All", self)
        self.select_all_cb.setTristate(True)
        self.select_all_cb.stateChanged.connect(self.on_select_all_changed)
        main_layout.addWidget(self.select_all_cb)

        # 创建符号复选框
        for symbol, desc in self.symbol_map.items():
            cb = QCheckBox(f"{symbol} - {desc}", self)
            cb.setChecked(symbol in self.initial_selection)
            cb.stateChanged.connect(self.update_select_all_state)

            self.checkboxes[symbol] = cb
            main_layout.addWidget(cb)

        # 添加操作按钮
        button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
            Qt.Horizontal, self
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        main_layout.addWidget(button_box)

        self.setLayout(main_layout)
        self.setWindowTitle("Enhanced Symbol Selector")
        self.resize(480, 360)
        self.update_select_all_state()  # 初始化全选状态 [2](@ref)

    def on_select_all_changed(self, state):
        for cb in self.checkboxes.values():
            cb.blockSignals(True)

        # 设置所有子复选框状态 [2](@ref)
        if state == Qt.Checked:
            for cb in self.checkboxes.values():
                cb.setChecked(True)
        elif state == Qt.Unchecked:
            for cb in self.checkboxes.values():
                cb.setChecked(False)

        # 解除信号阻塞
        for cb in self.checkboxes.values():
            cb.blockSignals(False)

    def update_select_all_state(self):
        """更新全选复选框状态"""
        checked_count = sum(1 for cb in self.checkboxes.values() if cb.isChecked())
        total = len(self.checkboxes)

        if checked_count == 0:
            new_state = Qt.Unchecked
        elif checked_count == total:
            new_state = Qt.Checked
        else:
            new_state = Qt.PartiallyChecked

        self.select_all_cb.blockSignals(True)
        self.select_all_cb.setCheckState(new_state)
        self.select_all_cb.blockSignals(False)

    def get_selection(self):
        """获取当前选中项"""
        return [sym for sym, cb in self.checkboxes.items() if cb.isChecked()]

    @classmethod
    def get_symbols(cls, symbol_map, initial_selection, parent=None):
        """执行对话框并返回结果"""
        dialog = cls(symbol_map, initial_selection, parent)
        result = dialog.exec_()
        return dialog.get_selection(), result == QDialog.Accepted


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
        model.setHorizontalHeaderLabels(['File', 'Address', 'Size', 'Type', 'Symbol'])

        for _, row in df.iterrows():
            size_item = QStandardItem()
            size_item.setData(row['Size'], Qt.DisplayRole)  # 以数字类型存储

            items = [
                QStandardItem(row['Filename']),
                QStandardItem(row['Address']),
                size_item,
                QStandardItem(row['Type']),
                QStandardItem(row['Symbol'])
            ]
            model.appendRow(items)

        self.table.setModel(model)
        self.table.setSortingEnabled(True)
        self.table.resizeColumnsToContents()

    def export_to_csv(self):
        model = self.table.model()
        if model.rowCount() == 0:
            return

        # 弹出文件保存对话框
        filename, _ = QFileDialog.getSaveFileName(
            self, "保存 CSV 文件", "", "CSV 文件 (*.csv)"
        )
        if not filename:
            return

        # 写入 CSV
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # 写入表头
            header = [model.headerData(i, Qt.Horizontal) for i in range(model.columnCount())]
            writer.writerow(header)
            # 写入数据行
            for row in range(model.rowCount()):
                row_data = []
                for col in range(model.columnCount()):
                    item = model.item(row, col)
                    row_data.append(item.text() if item else "")
                writer.writerow(row_data)


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

    def export_charts(self):
        try:
            path, selected_filter = QFileDialog.getSaveFileName(
                self, "Save Charts", "",
                "PNG Files (*.png);;PDF Files (*.pdf);;SVG Files (*.svg);;All Files (*)"
            )
            if not path:
                return

            # 自动识别文件格式[6,7](@ref)
            format_map = {
                "png": "png",
                "pdf": "pdf",
                "svg": "svg"
            }
            file_ext = path.split('.')[-1].lower() if '.' in path else ""
            format = format_map.get(file_ext, "png")  # 默认PNG格式

            # 创建进度对话框
            progress = QProgressDialog("Saving charts...", "Cancel", 0, len(self.figure_pool), self)
            progress.setWindowModality(Qt.WindowModal)

            for i, fig in enumerate(self.figure_pool):
                if progress.wasCanceled():
                    break

                progress.setValue(i)
                QApplication.processEvents()  # 保持UI响应

                try:
                    # 动态生成文件名
                    filename = f"{path.rsplit('.', 1)[0]}_{i + 1}.{format}" if '.' in path else f"{path}_{i + 1}.{format}"

                    fig.savefig(
                        filename,
                        dpi=300,
                        format=format,
                        facecolor='white',
                        transparent=False,
                        bbox_inches="tight",
                        metadata={
                            'Creator': f"{self.__class__.__name__} Export",
                            'CreationDate': datetime.datetime.now().isoformat()
                        }
                    )
                except PermissionError as pe:
                    QMessageBox.critical(self, "Permission Error",
                                         f"Cannot write to {filename}:\n{str(pe)}\nCheck file permissions.")
                    break
                except IOError as ioe:
                    QMessageBox.critical(self, "IO Error",
                                         f"Failed to save {filename}:\n{str(ioe)}\nCheck disk space/path validity.")
                    break
                except Exception as e:
                    QMessageBox.critical(self, "Unexpected Error",
                                         f"Failed to save chart {i + 1}:\n{str(e)}")
                    break

            progress.close()

        except Exception as e:
            QMessageBox.critical(self, "Export Failed",
                                 f"Chart export aborted:\n{str(e)}")
        finally:
            if 'progress' in locals():
                progress.close()

    def _pre_process_data(self, df: pd.DataFrame):
        try:
            # Type conversion (safe mode)
            df['Size'] = pd.to_numeric(df['Size'], errors='coerce')
            valid_df = df.dropna(subset=['Size']).copy()

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

            if chart_type == 'file_size':
                display_data, y_labels, title = self._data_analysis_file_size(self.df)
                self._plot_column_chart(display_data, y_labels, title, chart_type)
            elif chart_type == 'symbols':
                display_data, y_labels, title = self._data_analysis_symbol(self.df)
                self._plot_column_chart(display_data, y_labels, title, chart_type)
            elif chart_type == 'type':
                display_data, labels, title = self._data_analysis_symbol_type(self.df)

                # Extra operation for column chart.
                display_data_col = display_data.sort_values(ascending=True)
                labels_col = display_data_col.index.tolist()
                self._plot_column_chart(display_data_col, labels_col, title, 'type')

                self._plot_pie_chart(display_data, labels, title)
            else:
                raise ValueError(f"Unknown chart type: {chart_type}")

        except KeyError as ke:
            self._add_placeholder(f"Missing required field: {str(ke)}")
        except ValueError as ve:
            self._add_placeholder(f"Data format error: {str(ve)}")
        except Exception as e:
            self._add_placeholder(f"Rendering exception: {str(e)}")
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
        return display_data, display_data.index.tolist(), "Symbol Type Size Distribution"

    # 动态计算高度（按标签密度）
    def _calculate_fig_height(self, labels, font_size=8, dpi=100):
        # Each label needs to be about 40 pixels (0.4 inches * 100 dpi)
        base_height_inch = len(labels) * 0.4
        # Compatible with high DPI screens (automatically doubles when zoomed in 200%)
        screen_dpi = self.screen().logicalDotsPerInch()
        scaled_height = base_height_inch * (screen_dpi / 100)
        return max(8, scaled_height)

    def _get_chart_color(self, chart_type):
        """Color mapping for visualization elements"""
        return {
            'file_size': '#1f77b4', # Primary blue for quantitative data
            'symbols': '#2ca02c',   # Balanced green for categorical metrics
            'type': '#d62728',      # High-contrast red for key categories
            'reserve1': '#9467bd',  # Muted purple for auxiliary datasets
            'reserve2': '#ff7f0e',  # Warm orange for secondary indicators
            'reserve3': '#8c564b',  # Neutral brown for background elements
            'reserve4': '#e377c2',  # Vibrant pink for special cases
            'reserve5': '#7f7f7f',  # Medium gray for neutral visualization
        }.get(chart_type, '#17becf')  # Fallback color (cyan-blue)

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
        print("[DEBUG] Original size:", fig.get_size_inches())
        print("[DEBUG] DPI setting:", fig.dpi)
        print("[DEBUG] Qt recommended size:", canvas.sizeHint())
        print("[DEBUG] Screen DPI:", canvas.screen().logicalDotsPerInch())

        canvas.setSizePolicy(QSizePolicy.Expanding, v_policy)
        print("[DEBUG] Actual rendering size:", canvas.size().height())

        self.layout.addWidget(canvas)
        canvas.draw_idle()
        print(f"Number of layout items: {self.layout.count()}")

        fig.subplots_adjust(
            left=0.4,       # Increase left margin (leave space for long labels)
            right=0.95,     # Reduce right margin
            top=0.92,       # Fine-tune top margin
            bottom=0.08,    # Increase bottom margin
            hspace=0.4      # Vertical spacing between sub-images
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


class SymbolDiffView(QWidget):
    def __init__(self, diff_df, parent=None):
        super().__init__(parent)
        self.diff_df = diff_df  # 输入的差异计算结果DataFrame
        self.init_ui()

    def init_ui(self):
        # 基础布局设置
        self.setWindowTitle("Symbol Differences")
        self.layout = QVBoxLayout(self)

        # 创建表格控件
        self.table = QTableWidget()
        self.table.setRowCount(len(self.diff_df))
        self.table.setColumnCount(len(self.diff_df.columns))

        # 设置表头
        self.table.setHorizontalHeaderLabels(self.diff_df.columns.tolist())
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)  # 自动调整列宽
        self.table.horizontalHeader().setSortIndicatorShown(True)  # 显示排序箭头

        # 填充数据
        for row_idx, row_data in self.diff_df.iterrows():
            for col_idx, col_name in enumerate(self.diff_df.columns):
                cell_value = row_data[col_name]
                item = QTableWidgetItem()

                # 数值类型特殊处理（确保正确排序）
                if isinstance(cell_value, (int, float)):
                    item.setData(Qt.DisplayRole, cell_value)  # 设置数值类型数据
                else:
                    item.setText(str(cell_value))
                self.table.setItem(row_idx, col_idx, item)

        # 启用交互功能
        self.table.setSortingEnabled(True)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)

        # 布局添加控件
        self.layout.addWidget(self.table)


if __name__ == '__main__':
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
