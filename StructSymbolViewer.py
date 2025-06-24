import re
import traceback
from typing import Optional

from PyQt5.QtWidgets import QSplitter
from aiosqlite import connect
from pycparser import c_parser, c_ast, c_generator

import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QLineEdit, QPushButton, QPlainTextEdit, QSizePolicy, QFileDialog
)
from PyQt5.QtCore import QSettings, Qt


class StructLayoutCalculator:
    def __init__(self, basic_type_sizes=None):
        """
        初始化结构体布局计算器
        :param basic_type_sizes: 基本类型的大小映射，例如 {'int': 4, 'float': 4}
        """
        self.basic_type_sizes = basic_type_sizes or {
            'char': 1, 'unsigned char': 1,
            'short': 2, 'unsigned short': 2,
            'int': 4, 'unsigned int': 4,
            'long': 8, 'unsigned long': 8,
            'float': 4, 'double': 8,
        }
        self.typedef_map = {}  # 存储typedef别名到实际类型的映射
        self.struct_defs = {}  # 存储结构体定义
        self.global_vars = {}
        self.struct_layouts = {}  # 存储结构体布局结果
        self.undefined_symbols = set()  # 存储未定义的符号

    def _preprocess_code(self, code):
        """
        预处理代码：移除单行注释并标准化空格
        """
        # 移除单行注释（//...）
        code = re.sub(r'//.*', '', code)
        # 标准化空格和换行
        code = re.sub(r'\s+', ' ', code).strip()
        return code

    def _resolve_type(self, type_name):
        """递归解析typedef链，返回最终类型"""
        visited = set()
        while type_name in self.typedef_map:
            if type_name in visited:  # 防止循环引用
                break
            visited.add(type_name)
            type_name = self.typedef_map[type_name]
        return type_name

    def _get_type_size(self, typename):
        """
        获取类型的大小（基本类型/结构体）
        """
        resolved_type = self._resolve_type(typename)

        # 基本类型
        if resolved_type in self.basic_type_sizes:
            return self.basic_type_sizes[resolved_type]

        # 结构体类型
        if resolved_type in self.struct_layouts:
            return self.struct_layouts[resolved_type]['size']

        # 指针类型（假设4字节）
        if resolved_type.endswith('*'):
            return 4

        # 未知类型
        self.undefined_symbols.add(resolved_type)
        return 0  # 未知大小

    def _calculate_array_size(self, array_decl):
        """计算数组总大小（维度乘积）"""
        base_type = array_decl
        total_size = 1
        dims = []

        # 递归提取所有维度
        while isinstance(base_type, c_ast.ArrayDecl):
            if isinstance(base_type.dim, c_ast.Constant):
                dims.append(int(base_type.dim.value))
            base_type = base_type.type

        # 获取基础类型大小
        if isinstance(base_type, c_ast.TypeDecl):
            base_typename = ' '.join(base_type.type.names)
            base_size = self._get_type_size(base_typename)
        else:
            base_size = self._get_type_size(self._get_type_name(base_type))

        # 计算总大小 = 基础大小 × 所有维度乘积
        for dim in dims:
            total_size *= dim
        return total_size * base_size

    def _get_type_name(self, node):
        """
        从AST节点中提取类型名称
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
        return 'unknown'

    def _process_declaration(self, decl, current_offset):
        """
        处理单个成员声明，计算其偏移和大小
        """
        member_name = decl.name
        member_type = decl.type

        # 获取类型名称
        typename = self._get_type_name(member_type)
        resolved_typename = self._resolve_type(typename)

        # 计算大小
        if isinstance(member_type, c_ast.ArrayDecl):
            member_size = self._calculate_array_size(resolved_typename)
            align_size = self._get_type_size(decl.type.type.type.names[0])
        else:
            member_size = self._get_type_size(resolved_typename)
            align_size = member_size

        # 对齐处理（假设4字节对齐）
        padding = (align_size - (current_offset % align_size)) % align_size
        current_offset += padding

        # 记录成员信息
        member_info = {
            'offset': current_offset,
            'size': member_size,
            'type': resolved_typename
        }

        # 更新偏移量
        current_offset += member_size

        return member_info, current_offset

    def _process_struct(self, node):
        """
        处理单个结构体定义
        """
        struct_name = node.name
        if not struct_name:
            # 匿名结构体处理
            struct_name = f'anonymous_{id(node)}'

        # 存储结构体定义
        self.struct_defs[struct_name] = node

        # 处理成员
        members = {}
        current_offset = 0

        if node.decls:
            for decl in node.decls:
                # 跳过函数声明和非成员
                if not isinstance(decl, c_ast.Decl) or not decl.name:
                    continue

                member_info, current_offset = self._process_declaration(decl, current_offset)
                members[decl.name] = member_info

        # 结构体总大小（按4字节对齐）
        struct_size = current_offset
        padding = (4 - (struct_size % 4)) % 4
        struct_size += padding

        # 存储布局结果
        self.struct_layouts[struct_name] = {
            'size': struct_size,
            'members': members
        }

        return struct_name

    def _process_typedef(self, node):
        """处理typedef定义，正确提取别名和实际类型"""
        alias_name = node.name  # typedef后的别名（如"Omega_Target_PostProcess0_type"）
        actual_type = None

        # 1. 处理基本类型和结构体引用（IdentifierType）
        if isinstance(node.type, c_ast.TypeDecl):
            if isinstance(node.type.type, c_ast.IdentifierType):
                # 处理基本类型（如"struct MyStruct"）
                actual_type = ' '.join(node.type.type.names)
            elif isinstance(node.type.type, c_ast.Struct):
                # 处理内联结构体定义（typedef与结构体定义在一起）
                struct_node = node.type.type
                if struct_node.name:
                    # 有标签名：映射到"struct <标签名>"
                    actual_type = f"struct {struct_node.name}"
                    self.struct_defs[struct_node.name] = struct_node
                else:
                    # 匿名结构体：直接使用别名作为类型名
                    actual_type = alias_name
                    self.struct_defs[alias_name] = struct_node
        # 2. 处理独立结构体定义（typedef struct {...}）
        elif isinstance(node.type, c_ast.Struct):
            struct_node = node.type
            if struct_node.name:
                actual_type = f"struct {struct_node.name}"
                self.struct_defs[struct_node.name] = struct_node
            else:
                actual_type = alias_name
                self.struct_defs[alias_name] = struct_node

        # 存储映射关系
        if actual_type:
            self.typedef_map[alias_name] = actual_type
        else:
            self.undefined_symbols.add(f"无法解析的类型: {alias_name}")

    def _process_variable_declaration(self, decl_node):
        var_name = decl_node.name  # 变量名（如"Omega_Target_PostProcess0"）
        # 提取类型名（需处理基础类型、typedef别名、结构体等）
        if isinstance(decl_node.type, c_ast.TypeDecl):
            type_name = ' '.join(decl_node.type.type.names)
        elif isinstance(decl_node.type, c_ast.IdentifierType):
            type_name = ' '.join(decl_node.type.names)
        else:
            type_name = ''
        # 存储到符号表
        if type_name:
            self.global_vars[var_name] = type_name
        else:
            print('Warning: Cannot get type name.')

    def _is_function_context(self, node):
        """检查节点是否在函数内部（即局部作用域）"""
        parent = node
        while hasattr(parent, 'parent') and parent.parent is not None:
            parent = parent.parent
            if isinstance(parent, c_ast.FuncDef):
                return True  # 当前节点位于函数内
        return False  # 全局作用域

    def parse(self, code) -> bool:
        """
        解析C代码，计算所有结构体布局
        :return: 结构体布局字典和未定义符号集合
        """

        self.typedef_map.clear()
        self.struct_defs.clear()
        self.global_vars.clear()

        self.struct_layouts.clear()
        self.undefined_symbols.clear()

        try:
            preprocessed = self._preprocess_code(code)
            parser = c_parser.CParser()
            ast = parser.parse(preprocessed)
        except Exception as e:
            print(e)
            traceback.format_exc()
            return False

        # 遍历AST处理所有定义
        for item in ast.ext:
            if isinstance(item, c_ast.Typedef):
                # 处理typedef及内联结构体
                self._process_typedef(item)
            elif isinstance(item, c_ast.Decl):
                # 处理显式的结构体定义（无typedef）
                if isinstance(item.type, c_ast.Struct):
                    self._process_struct(item.type)
                elif isinstance(item.type, c_ast.TypeDecl):
                    # 处理变量声明中的结构体
                    if 'struct' in ' '.join(item.type.type.names):
                        struct_name = re.search(r'struct\s+(\w+)', ' '.join(item.type.type.names))
                        if struct_name and struct_name.group(1) in self.struct_defs:
                            self._process_struct(self.struct_defs[struct_name.group(1)])
                elif self._is_function_context(item):
                    self._process_variable_declaration(item)

        # 处理所有已收集的结构体定义
        for struct_name, node in self.struct_defs.items():
            if struct_name not in self.struct_layouts:
                self._process_struct(node)

        # 将别名指向对应的结构体布局
        for alias in list(self.typedef_map.keys()):
            base_type = self._resolve_type(alias)
            if base_type.startswith('struct ') or base_type[7:] in self.struct_layouts:
                struct_tag = base_type[7:]
                self.struct_layouts[alias] = self.struct_layouts[struct_tag]

        return True

    def get_member_layout(self):
        """
        获取结构体布局的友好表示
        """
        results = {}

        for struct_name, layout in self.struct_layouts.items():
            # 提取成员偏移和大小
            member_layout = {
                name: (info['offset'], info['size'])
                for name, info in layout['members'].items()
            }
            results[struct_name] = member_layout

        return results

    def get_global_variables(self):
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


class AdiXmlMapFileParser:
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

def verify_struct_layout_calc():
    # 示例结构体定义
    defines = """
    typedef signed char int8_T;
    typedef unsigned char uint8_T;
    typedef short int16_T;
    typedef unsigned short uint16_T;
    typedef int int32_T;
    typedef unsigned int uint32_T;
    typedef float real32_T;
    typedef double real64_T;
    
    typedef double real_T;
    typedef double time_T;
    typedef unsigned char boolean_T;
    typedef int int_T;
    typedef unsigned int uint_T;
    typedef unsigned long ulong_T;
    typedef char char_T;
    typedef unsigned char uchar_T;
    typedef char_T byte_T;
    """
    code = """
    typedef struct Omega_Target_PostProcess0_tag
    {
        int32_T output_router_EnableAux;
        int32_T output_router_MainSelect[22];
        real32_T AllMuteRampTime;
        real32_T AuxSelectMap[22];
        // ... 其他成员
    } Omega_Target_PostProcess0_type;
    
    
    typedef struct Omega_Target_PreAmp0_tag
    {
        int32_T LevelDetectMusicDelaySamples;
        int32_T MedusaAlignmentDelay;
        int32_T MedusaBassDelayDec;
        int32_T MedusaBassDelayFull;
        int32_T MedusaMonoDetectorEnable;
        int32_T MedusaTrebleDelay;
        int32_T audiopilot_HfNoiseRefLatencySamples;
        int32_T audiopilot_LfNoiseRefLatencySamples;
        int32_T audiopilot_MidrangeLpfAlignDelaySamples;
    };
    """

    # 创建计算器
    calculator = StructLayoutCalculator()

    # 获取结构体布局
    layouts, undefined = calculator.get_layout(defines + code)

    # 打印结果
    for struct_name, members in layouts.items():
        print(f"结构体: {struct_name}")
        print("总大小:", layouts[struct_name].get('size', '未知'))
        for member, (offset, size) in members.items():
            print(f"  {member}: 偏移={offset}, 大小={size}")
        print()

    # 处理未定义符号
    if undefined:
        print("未定义符号:", ", ".join(undefined))
        print("请补充这些类型的定义")


class MapAnalyzerApp(QWidget):
    def __init__(self):
        super().__init__()
        self.bin_edit = None
        self.map_edit = None
        self.input_text: Optional[QPlainTextEdit] = None
        self.result_text: Optional[QPlainTextEdit] = None
        self.analyze_btn = None
        self.map_parser = None
        self.struct_parser = None
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

        text_area_layout = QSplitter()
        text_area_layout.addWidget(self.input_text)
        text_area_layout.addWidget(self.result_text)

        # 添加所有行到主布局
        main_layout.addLayout(map_layout)
        main_layout.addLayout(bin_layout)
        main_layout.addLayout(analyze_layout)
        main_layout.addWidget(text_area_layout)

        self.setLayout(main_layout)

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
            self.map_parser = NULL

        if struct_declare:
            self.result_text.appendPlainText("Struct declaration specified, parsing...")
            self.struct_parser = StructLayoutCalculator()
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

        global_variables = self.struct_parser.get_global_variables()
        for var in global_variables:
            print(var)

        # layouts, undefined = self.struct_parser.get_member_layout()
        # for struct_name, members in layouts.items():
        #     struct_symbol_info = self.map_parser.get_symbol_info(struct_name)
        #     if not struct_symbol_info:
        #         print(f'Cannot find structure symbol {struct_name} in map file.')
        #         continue
        #
        #     start_address = struct_symbol_info['address']
        #
        #     self.output(f"Structure: {struct_name}")
        #     self.output(f"- Start address: {start_address}")
        #     self.output(f"- Size: {struct_symbol_info['size']}")
        #
        #     self.output("- Members: ")
        #     for member, (offset, size) in members.items():
        #         member_address = start_address + offset
        #         self.print(f"    |--{member}: offset: {offset}, size: {size}, "
        #                    f": [{member_address}, {member_address + size})")
        self.output("Analysis completed!\n")

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
    window = MapAnalyzerApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
