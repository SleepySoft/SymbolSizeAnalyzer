import re
import sys
import traceback
import pandas as pd
from typing import Optional, Tuple
from pycparser import c_parser, c_ast

from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QLineEdit, QPushButton, QPlainTextEdit, QFileDialog, QTreeWidget, QTreeWidgetItem, QTreeWidgetItemIterator
)
from PyQt5.QtCore import QSettings, Qt, QAbstractTableModel


class CGrammarParser:
    """
    C Language grammar parser.
    Parses C code to extract struct layouts and global variable declarations.
    """

    def __init__(self, basic_type_sizes=None):
        """
        Initialize the C code parser with type size mappings
        :param basic_type_sizes: Dictionary of basic C type sizes (e.g., {'int': 4, 'float': 4})
        """
        # Default type sizes if not provided
        self.basic_type_sizes = basic_type_sizes or {
            'char': 1, 'unsigned char': 1,
            'short': 2, 'unsigned short': 2,
            'int': 4, 'unsigned int': 4,
            'long': 4, 'unsigned long': 4,
            'float': 4, 'double': 8,
        }
        self.typedef_map = {}  # Maps typedef aliases to actual types
        self.struct_defs = {}  # Stores struct definitions
        self.global_vars = {}  # Stores global variable declarations
        self.struct_layouts = {}  # Stores computed struct layouts
        self.undefined_symbols = set()  # Tracks unresolved symbols

    def _preprocess_code(self, code):
        """
        Preprocess C code by removing comments and normalizing whitespace
        :param code: Raw C code string
        :return: Cleaned code without comments or extra whitespace
        """
        code = re.sub(r'//.*', '', code)  # Remove single-line comments
        code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)  # Remove multi-line comments
        return re.sub(r'\s+', ' ', code).strip()  # Normalize whitespace

    def _resolve_type(self, type_name):
        """
        Resolve typedef chains to get the underlying type
        :param type_name: Type name to resolve
        :return: Base type after following all typedefs
        """
        visited = set()
        while type_name in self.typedef_map:
            if type_name in visited:  # Prevent infinite loops
                break
            visited.add(type_name)
            type_name = self.typedef_map[type_name]
        return type_name

    def _get_alignment(self, typename):
        """
        Calculate alignment requirements for a type
        :param typename: Type name to check
        :return: Alignment size in bytes
        """
        resolved_type = self._resolve_type(typename)

        # Handle pointer types (4-byte alignment)
        if resolved_type.endswith('*'):
            return 4

        # Handle basic types
        if resolved_type in self.basic_type_sizes:
            return self.basic_type_sizes[resolved_type]

        # Handle struct types (use max member alignment)
        if resolved_type in self.struct_layouts:
            max_align = 1
            for member_info in self.struct_layouts[resolved_type]['members'].values():
                align = self._get_alignment(member_info['type'])
                max_align = max(max_align, align)
            return max_align

        # Default alignment
        return 4

    def _get_type_size(self, typename):
        """
        Get size of a type (basic type, struct, or pointer)
        :param typename: Type name to check
        :return: Size in bytes, 0 if unknown
        """
        resolved_type = self._resolve_type(typename)

        # Basic types
        if resolved_type in self.basic_type_sizes:
            return self.basic_type_sizes[resolved_type]

        # Struct types
        if resolved_type in self.struct_layouts:
            return self.struct_layouts[resolved_type]['size']

        # Pointer types
        if resolved_type.endswith('*'):
            return 4

        # Arrays (calculate total size)
        if '[' in resolved_type:
            base_type = resolved_type.split('[')[0].strip()
            dims = [int(dim) for dim in re.findall(r'$$(\d+)$$', resolved_type)]
            total_size = self._get_type_size(base_type)
            for dim in dims:
                total_size *= dim
            return total_size

        # Unresolved type
        self.undefined_symbols.add(resolved_type)
        return 0

    def _calculate_array_size(self, array_decl):
        """
        Calculate total size of an array declaration
        :param array_decl: AST array declaration node
        :return: Total size in bytes
        """
        base_type = array_decl
        total_size = 1
        dims = []

        # Extract array dimensions
        while isinstance(base_type, c_ast.ArrayDecl):
            if isinstance(base_type.dim, c_ast.Constant):
                dims.append(int(base_type.dim.value))
            base_type = base_type.type

        # Get base type size
        if isinstance(base_type, c_ast.TypeDecl):
            base_typename = ' '.join(base_type.type.names)
            base_size = self._get_type_size(base_typename)
        else:
            base_size = self._get_type_size(self._get_type_name(base_type))

        # Calculate total size
        for dim in dims:
            total_size *= dim
        return total_size * base_size

    def _get_type_name(self, node):
        """
        Extract type name from AST node
        :param node: AST node to process
        :return: Type name as string
        """
        if isinstance(node, c_ast.TypeDecl):
            return ' '.join(node.type.names)
        elif isinstance(node, c_ast.Struct):
            return f'struct {node.name}' if node.name else 'anonymous'
        elif isinstance(node, c_ast.IdentifierType):
            return ' '.join(node.names)
        elif isinstance(node, c_ast.ArrayDecl):
            base_type = self._get_type_name(node.type)
            dim = node.dim.value if node.dim else ''
            return f"{base_type}[{dim}]"
        elif isinstance(node, c_ast.PtrDecl):
            base_type = self._get_type_name(node.type)
            return f"{base_type}*"
        return 'unknown'

    def _is_local_variable(self, node):
        """
        Determine if a variable is local (function-scoped)
        :param node: AST node to check
        :return: True if variable is local, False if global
        """
        parent = node
        while hasattr(parent, 'parent') and parent.parent is not None:
            parent = parent.parent
            if isinstance(parent, (c_ast.FuncDef, c_ast.Compound)):
                return True
        return False

    def _process_declaration(self, decl, current_offset):
        """
        Process a struct member declaration
        :param decl: AST declaration node
        :param current_offset: Current byte offset in struct
        :return: Member info and updated offset
        """
        member_name = decl.name
        member_type = decl.type

        # Get type information
        typename = self._get_type_name(member_type)
        resolved_typename = self._resolve_type(typename)

        # Calculate size and alignment
        if isinstance(member_type, c_ast.ArrayDecl):
            member_size = self._calculate_array_size(member_type)
            align_size = self._get_alignment(resolved_typename)
        else:
            member_size = self._get_type_size(resolved_typename)
            align_size = self._get_alignment(resolved_typename)

        # Apply alignment padding
        padding = (align_size - (current_offset % align_size)) % align_size
        current_offset += padding

        # Store member information
        member_info = {
            'offset': current_offset,
            'size': member_size,
            'type': resolved_typename,
            'align': align_size
        }

        # Update offset
        current_offset += member_size

        return member_info, current_offset

    def _process_struct(self, node):
        """
        Process a struct definition
        :param node: AST struct node
        :return: Struct name
        """
        struct_name = node.name
        if not struct_name:
            struct_name = f'anonymous_{id(node)}'

        # Store struct definition
        self.struct_defs[struct_name] = node

        # Process members
        members = {}
        current_offset = 0
        max_align = 1  # Track maximum alignment requirement

        if node.decls:
            for decl in node.decls:
                if not isinstance(decl, c_ast.Decl) or not decl.name:
                    continue

                member_info, current_offset = self._process_declaration(decl, current_offset)
                members[decl.name] = member_info
                max_align = max(max_align, member_info['align'])

        # Apply final struct padding
        padding = (max_align - (current_offset % max_align)) % max_align
        struct_size = current_offset + padding

        # Store layout
        self.struct_layouts[struct_name] = {
            'size': struct_size,
            'members': members,
            'align': max_align
        }

        return struct_name

    def _process_typedef(self, node):
        """
        Process typedef declarations
        :param node: AST typedef node
        """
        alias_name = node.name
        actual_type = None

        # Handle different typedef patterns
        if isinstance(node.type, c_ast.TypeDecl):
            if isinstance(node.type.type, c_ast.IdentifierType):
                actual_type = ' '.join(node.type.type.names)
            elif isinstance(node.type.type, c_ast.Struct):
                struct_node = node.type.type
                if struct_node.name:
                    actual_type = f"struct {struct_node.name}"
                    self.struct_defs[struct_node.name] = struct_node
                else:
                    actual_type = alias_name
                    self.struct_defs[alias_name] = struct_node
        elif isinstance(node.type, c_ast.Struct):
            struct_node = node.type
            if struct_node.name:
                actual_type = f"struct {struct_node.name}"
                self.struct_defs[struct_node.name] = struct_node
            else:
                actual_type = alias_name
                self.struct_defs[alias_name] = struct_node

        # Store mapping
        if actual_type:
            self.typedef_map[alias_name] = actual_type
        else:
            self.undefined_symbols.add(f"Unresolved type: {alias_name}")

    def _process_variable_declaration(self, decl_node):
        """
        Process global variable declarations
        :param decl_node: AST declaration node
        """
        if self._is_local_variable(decl_node):
            return  # Skip local variables

        var_name = decl_node.name
        type_name = self._get_type_name(decl_node.type)
        # resolved_type = self._resolve_type(type_name)

        self.global_vars[var_name] = type_name

    def parse(self, code) -> bool:
        """
        Parse C code and extract struct layouts + global variables
        :param code: C source code to parse
        :return: True if successful, False otherwise
        """
        # Reset state
        self.typedef_map.clear()
        self.struct_defs.clear()
        self.global_vars.clear()
        self.struct_layouts.clear()
        self.undefined_symbols.clear()

        try:
            # Parse code into AST
            preprocessed = self._preprocess_code(code)
            parser = c_parser.CParser()
            ast = parser.parse(preprocessed)
        except Exception as e:
            print(f"Parsing error: {e}")
            traceback.print_exc()
            return False

        # First pass: Process typedefs and struct definitions
        for item in ast.ext:
            if isinstance(item, c_ast.Typedef):
                self._process_typedef(item)
            elif isinstance(item, c_ast.Decl) and isinstance(item.type, c_ast.Struct):
                self._process_struct(item.type)

        # Second pass: Process global variables
        for item in ast.ext:
            if isinstance(item, c_ast.Decl) and not isinstance(item.type, c_ast.FuncDef):
                self._process_variable_declaration(item)

        # Process all collected structs
        for struct_name, node in self.struct_defs.items():
            if struct_name not in self.struct_layouts:
                self._process_struct(node)

        # Map typedef aliases to struct layouts
        for alias, base_type in self.typedef_map.items():
            if base_type.startswith('struct ') and base_type[7:] in self.struct_layouts:
                struct_tag = base_type[7:]
                self.struct_layouts[alias] = self.struct_layouts[struct_tag]

        return True

    def get_member_layout(self):
        """
        Get formatted struct layouts
        :return: Dictionary of struct layouts
        """
        return {
            name: {m: (info['offset'], info['size'], info['type'])
                   for m, info in layout['members'].items()}
            for name, layout in self.struct_layouts.items()
        }

    def get_global_variables(self):
        """
        Get extracted global variables
        :return: Dictionary of global variables
        """
        return self.global_vars


class MapFileParser:
    def get_symbol_info(self, normalized_name) -> dict:
        return {
            'address': 0,
            'size': 0,
            'end_address': 0,
            'binding': ''
        }

    def find_symbol_by_address(self, address) -> list:
        return [{
            'address': 0,
            'size': 0,
            'end_address': 0,
            'binding': ''
        }]


import xml.etree.ElementTree as ET


class AdiXmlMapFileParser(MapFileParser):
    def __init__(self, xml_file):
        self.original_symbols = {}  # 存储原始XML中的符号信息
        self.normalized_map = {}    # 按规范化名称索引的符号表
        self._parse(xml_file)
        self._build_normalized_map()

    def _parse(self, xml_file):
        """解析XML文件，存储所有SYMBOL原始数据"""
        tree = ET.parse(xml_file)
        for symbol in tree.iter('SYMBOL'):
            name = symbol.get('name')
            address = symbol.get('address')
            size = int(symbol.get('size'), 0)
            if not (name and address and size):
                continue

            try:
                self.original_symbols[name] = {
                    'address': int(address, 0),
                    'size': size,
                    'end_address': int(address, 0) + size,
                    'binding': symbol.get('binding', '')
                }
            except (ValueError, TypeError):
                continue

    def _normalize_name(self, name):
        """将编译器生成的符号名转换为原始名称"""
        # 规则1：去除结尾的单个点（如 `PostProcess_BankTable.` → `PostProcess_BankTable`）
        if name.endswith('.'):
            name = name[:-1]
        # 规则2：去除以点开头的段结束标记（如 `.PostProcess_BankTable..end` → `PostProcess_BankTable`）
        if name.startswith('.') and name.endswith('..end'):
            base_name = name[1:-5]  # 移除首尾修饰
            return base_name.rsplit('.', 1)[0] if '.' in base_name else base_name
        # 规则3：保留其他名称不变
        return name

    def _build_normalized_map(self):
        """构建以原始名称为键的符号映射"""
        for orig_name, info in self.original_symbols.items():
            norm_name = self._normalize_name(orig_name)
            # 合并同一原始名称的多个符号（如主体+end标记）
            if norm_name not in self.normalized_map:
                self.normalized_map[norm_name] = []
            self.normalized_map[norm_name].append({
                **info,
                'original_name': orig_name  # 保留原始名称用于追溯
            })

    def get_symbol_info(self, normalized_name) -> dict:
        """通过原始名称查询符号信息"""
        return self.normalized_map.get(normalized_name, [{}])[0]

    def find_symbol_by_address(self, address):
        """按地址查询符号（返回原始名称和规范名称）"""
        results = []
        for norm_name, entries in self.normalized_map.items():
            for entry in entries:
                if entry['address'] <= address < entry['end_address']:
                    results.append({**entry, 'normalized_name': norm_name})
        return results


# ----------------------------------------------------------------------------------------------------------------------

def analysis_variable_layout(c_grammar_parser: CGrammarParser, map_file_parser: AdiXmlMapFileParser):

    global_variables = c_grammar_parser.get_global_variables()
    structure_member_layouts = c_grammar_parser.get_member_layout()

    column_var_name = []
    column_struct_name = []
    column_member_name = []
    column_member_type = []
    column_member_offset = []
    column_member_address = []
    column_member_size = []

    for var_name, var_type in global_variables.items():
        symbol_info = map_file_parser.get_symbol_info(var_name)

        if not symbol_info:
            print(f'Cannot find symbol {var_name} in map file.')
            continue

        symbol_start_address = symbol_info['address']

        for struct_name, members in structure_member_layouts.items():
            if struct_name == var_type:
                for member, (offset, size, member_type) in members.items():
                    column_var_name.append(var_name)
                    column_struct_name.append(var_type)

                    member_address = symbol_start_address + offset

                    column_member_name.append(member)
                    column_member_type.append(member_type)
                    column_member_offset.append(offset)
                    column_member_size.append(size)
                    column_member_address.append(member_address)
                break

    return pd.DataFrame(
        {
            'variant': column_var_name,
            'struct': column_struct_name,
            'member': column_member_name,
            'type': column_member_type,
            'offset': column_member_offset,
            'size': column_member_size,
            'address': column_member_address,
        }).reset_index()


class PandasModel(QAbstractTableModel):
    def __init__(self, data):
        super().__init__()
        self._data = data

    def rowCount(self, parent=None):
        return self._data.shape[0]

    def columnCount(self, parent=None):
        return self._data.shape[1]

    def data(self, index, role=Qt.DisplayRole):
        if role == Qt.DisplayRole and index.isValid():
            return str(self._data.iloc[index.row(), index.column()])
        return None

    def headerData(self, section, orientation, role):
        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                return str(self._data.columns[section])
            else:
                return str(self._data.index[section])
        return None


def display_dataframe_as_table(table_view, dataframe):
    """
    在PyQt5表格中显示DataFrame内容（自动清空原有数据）

    参数:
        table_view (QTableView): 要显示数据的表格视图对象
        dataframe (pd.DataFrame): 要显示的Pandas DataFrame
    """
    # 清空表格现有内容（通过重置模型实现）
    table_view.setModel(None)

    # 创建自定义模型并设置到表格视图
    model = PandasModel(dataframe)
    table_view.setModel(model)

    # 自动调整列宽适应内容
    table_view.resizeColumnsToContents()

    # 可选：设置交替行颜色增强可读性
    table_view.setAlternatingRowColors(True)


def display_dataframe_as_tree_list(tree: QTreeWidget, df: pd.DataFrame,
                                   group_by_column: str, display_column: [str], unique_column: str):
    # 设置列标题（分组列 + 展示列）
    headers = [group_by_column] + display_column
    tree.setHeaderLabels(headers)
    tree.setColumnCount(len(headers))

    # 按分组列聚合数据
    grouped = df.groupby(group_by_column)
    for group_name, group_df in grouped:
        # 创建分组根节点
        root_item = QTreeWidgetItem(tree)
        root_item.setText(0, str(group_name))  # 分组列显示在首列

        # 添加子节点（每组内的数据行）
        for _, row in group_df.iterrows():
            child = QTreeWidgetItem(root_item)
            if unique_column:
                child.setData(0, Qt.UserRole, row[unique_column])
            for col_idx, col in enumerate(display_column, start=1):  # 从第2列开始填充
                child.setText(col_idx, str(row[col]))
    tree.expandAll()


class StructLayoutAnalyzerUI(QWidget):
    def __init__(self, injection = None):
        super().__init__()
        self.bin_edit = None
        self.map_edit = None
        self.input_text: Optional[QPlainTextEdit] = None
        self.result_text: Optional[QPlainTextEdit] = None
        # self.result_table: Optional[QTableView] = None
        self.result_tree: Optional[QTreeWidget] = None
        self.df_symbol = pd.DataFrame()
        self.df_display = pd.DataFrame()
        self.display_column = []
        self.injection = injection                  # Injection to change UI behaviour
        self.analyze_btn = None
        self.map_parser: Optional[AdiXmlMapFileParser] = None
        self.struct_parser: Optional[CGrammarParser] = None
        self.settings = QSettings("SleepySoft", "MapAnalyzer")
        self.init_ui()
        self.load_settings()

    def init_ui(self):
        # 主窗口设置
        self.setWindowTitle("CCES Map Analyzer")
        self.setMinimumSize(1280, 800)

        # 主垂直布局
        main_layout = QVBoxLayout()
        main_layout.setSpacing(15)  # 行间距

        # 第一行: Map文件选择
        map_layout = QHBoxLayout()
        self.map_edit = QLineEdit()
        map_btn = QPushButton("Select Map File")
        map_btn.setFixedWidth(120)  # 固定按钮宽度
        map_btn.clicked.connect(lambda: self.select_file(self.map_edit, "XML Files (*.xml)"))
        map_layout.addWidget(self.map_edit)
        map_layout.addWidget(map_btn)

        # 第二行: Bin文件选择
        bin_layout = QHBoxLayout()
        self.bin_edit = QLineEdit()
        bin_btn = QPushButton("Select Bin File")
        bin_btn.setFixedWidth(120)
        bin_btn.clicked.connect(lambda: self.select_file(self.bin_edit, "BIN Files (*.bin)"))
        bin_layout.addWidget(self.bin_edit)
        bin_layout.addWidget(bin_btn)

        # 第三行: 分析按钮 (右侧对齐)
        analyze_layout = QHBoxLayout()
        analyze_layout.addStretch()  # 添加弹性空间使按钮右对齐
        self.analyze_btn = QPushButton("Start Analysis")
        # self.analyze_btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)  # 保持原始大小
        self.analyze_btn.clicked.connect(self.start_analysis)
        analyze_layout.addWidget(self.analyze_btn)

        # 第四行: 多行只读文本框
        self.input_text = QPlainTextEdit()
        self.input_text.setPlaceholderText("Put your struct declaration here...")

        self.result_text = QPlainTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setPlaceholderText("Analysis results will appear here...")

        # self.result_table = QTableView()
        self.result_tree = QTreeWidget()

        text_area_layout = QSplitter()
        text_area_layout.addWidget(self.input_text)
        text_area_layout.addWidget(self.result_text)
        # text_area_layout.addWidget(self.result_table)
        text_area_layout.addWidget(self.result_tree)

        # 添加所有行到主布局
        main_layout.addLayout(map_layout)
        main_layout.addLayout(bin_layout)
        main_layout.addLayout(analyze_layout)
        main_layout.addWidget(text_area_layout)

        self.setLayout(main_layout)

        try:
            self.injection.init_ui(
                main_layout, [map_layout, bin_layout, analyze_layout, text_area_layout])
        except Exception as e:
            print('Try call injection.init_ui...')
            print(str(e))

    def select_file(self, edit_widget, file_filter):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select File", "", file_filter
        )
        if file_path:
            edit_widget.setText(file_path)
            self.save_settings()  # 每次选择后立即保存

    def start_analysis(self):
        self.clean()

        # 获取文件路径
        map_file = self.map_edit.text()
        bin_file = self.bin_edit.text()

        struct_declare = self.input_text.toPlainText()
        struct_declare = struct_declare.strip()
        self.settings.setValue("structure_declare", struct_declare)

        if map_file:
            self.result_text.appendPlainText("Map file specified, parsing...")
            self.map_parser = AdiXmlMapFileParser(map_file)
            self.result_text.appendPlainText("Map file parse done.")
        else:
            self.map_parser = None

        if struct_declare:
            self.result_text.appendPlainText("Struct declaration specified, parsing...")
            self.struct_parser = CGrammarParser()
            if not self.struct_parser.parse(struct_declare):
                self.struct_parser = None
                self.result_text.appendPlainText("Struct declaration parse fail.")
            else:
                self.result_text.appendPlainText("Struct declaration parse done.")
        else:
            self.struct_parser = None

        if not self.struct_parser or not self.map_parser:
            self.output("Analysis fail.\n")
            return

        analysis_result = analysis_variable_layout(self.struct_parser, self.map_parser)
        try:
            analysis_result = self.injection.extra_analysis(analysis_result)
        except Exception as e:
            print('Try call injection.extra_analysis...')
            print(str(e))
        self.df_symbol = analysis_result

        self.dump_analysis_result()
        self.prepare_display_data()

        display_col = ['Member', 'Type', 'Offset', 'Size', 'Address']
        try:
            display_col = self.injection.adjust_display_column(display_col)
        except Exception as e:
            print('Try call injection.adjust_display_column...')
            print(str(e))

        # display_dataframe_as_table(self.result_table, self.df_display)
        display_dataframe_as_tree_list(self.result_tree, self.df_display,
                                       'Variant', display_col, 'Address')
        self.display_column = display_col

        self.output('')
        self.output('')
        self.output("Analysis completed!\n")

    @staticmethod
    def format_address_hex(address: int) -> str:
        return f"0x{address:08X}"

    @staticmethod
    def map_tree_items(tree_widget, func):
        iterator = QTreeWidgetItemIterator(tree_widget)
        while iterator.value():
            item = iterator.value()
            func(item)
            iterator += 1

    def get_column_index(self, column_name) -> int:
        try:
            return self.display_column.index(column_name)
        except Exception:
            return -1

    def find_symbols_in_offset_range(self, offset: int, size: int = 1):
        start_offset = offset
        end_offset = offset + size

        # Create a temporary DataFrame with the 'offset_until' column (offset + size)
        temp_df = self.df_symbol.assign(offset_until=self.df_symbol['offset'] + self.df_symbol['size'])

        # Filter rows that overlap with the query interval
        overlap_condition = (temp_df['offset'] < end_offset) & (temp_df['offset_until'] > start_offset)
        result_df = temp_df.loc[overlap_condition].copy()

        return result_df

    def find_symbols_in_address_range(self, address: int, size: int = 1):
        start_address = address
        end_address = address + size

        # Create a temporary DataFrame with the 'offset_until' column (offset + size)
        temp_df = self.df_symbol.assign(address_until=self.df_symbol['address'] + self.df_symbol['size'])

        # Filter rows that overlap with the query interval
        overlap_condition = (temp_df['address'] < end_address) & (temp_df['offset_until'] > start_address)
        result_df = temp_df.loc[overlap_condition].copy()

        return result_df

    def select_tree_items_by_df_symbol_lines(self, df_symbol: pd.DataFrame):
        tree_item_user_data = [self.format_address_hex(address) for address in df_symbol['address']]
        selected_tree_items = []
        other_tree_items = []

        def select_tree_items(item: QTreeWidgetItem):
            user_data = item.data(0, Qt.UserRole)
            if user_data and user_data in tree_item_user_data:
                selected_tree_items.append(item)
            else:
                other_tree_items.append(item)
        self.map_tree_items(self.result_tree, select_tree_items)

        return selected_tree_items, other_tree_items

    def dump_analysis_result(self):
        grouped = self.df_symbol.groupby('variant')
        for group_name, group_df in grouped:
            symbol_start_address = None
            for row in group_df.itertuples():
                if not symbol_start_address:
                    symbol_start_address = row.address
                    self.output('')
                    self.output('')
                    self.output(f"{row.struct} {group_name};")
                    self.output(f"- Start address: 0x{symbol_start_address:08X}")
                    self.output(f"- Size: {group_df['size'].sum()}")

                offset = row.address - symbol_start_address
                self.output(f"    |--{row.member}: offset: {offset}, size: {row.size}, "
                            f": [0x{row.address:08X}, 0x{(row.address + row.size):08X})")

    def prepare_display_data(self):
        self.df_display = self.df_symbol.copy()
        self.df_display['address'] = self.df_display['address'].apply(self.format_address_hex)
        self.df_display.columns = [col.capitalize() for col in self.df_display.columns]

        try:
            df_display = self.injection.prepare_display_data(self.df_display)
            if df_display is not None:
                self.df_display = df_display
        except Exception as e:
            print('Try call injection.prepare_display_data...')
            print(str(e))

    def load_settings(self):
        """加载上次保存的设置[9](@ref)"""
        self.map_edit.setText(self.settings.value("map_file", ""))
        self.bin_edit.setText(self.settings.value("bin_file", ""))
        self.input_text.setPlainText(self.settings.value("structure_declare", ""))

    def save_settings(self):
        """保存当前设置"""
        self.settings.setValue("map_file", self.map_edit.text())
        self.settings.setValue("bin_file", self.bin_edit.text())

    def closeEvent(self, event):
        """窗口关闭时保存设置"""
        self.save_settings()
        super().closeEvent(event)

    def clean(self):
        self.result_text.setPlainText('')

    def output(self, text: str):
        self.result_text.appendPlainText(text)


def main():
    # verify_struct_layout_calc()

    app = QApplication(sys.argv)
    window = StructLayoutAnalyzerUI()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
