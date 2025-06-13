import json
import re
from typing import List, Dict, Any, Tuple

class Section:
    def __init__(self, title: str = "", content: List[str] = None, subsections: List['Section'] = None):
        self.title = title
        self.content = content or []
        self.subsections = subsections or []

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "title": self.title,
            "content": self.content
        }
        if self.subsections:
            result["subsections"] = [s.to_dict() for s in self.subsections]
        return result

def get_indent_level(line: str) -> int:
    """计算行的缩进级别"""
    indent = 0
    for char in line:
        if char.isspace():
            indent += 1
        else:
            break
    return indent

def process_indented_content(lines: List[str], start_idx: int, base_indent: int) -> Tuple[List[str], int]:
    """处理缩进内容，返回内容列表和下一个非缩进内容的索引"""
    content = []
    i = start_idx
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            i += 1
            continue
            
        current_indent = get_indent_level(line)
        if current_indent <= base_indent:
            break
            
        content.append(line.strip())
        i += 1
    return content, i

def debug(msg):
    print(f"[DEBUG] {msg}")

def parse_markdown(lines, level=1, start=0, end=None):
    if end is None:
        end = len(lines)
    sections = []
    i = start
    while i < end:
        line = lines[i]
        header_match = re.match(r'^(#+)\s*(.*)', line)
        if header_match:
            header_level = len(header_match.group(1))
            title = header_match.group(2).strip()
            debug(f"发现标题: level={header_level}, title={title}, 行号={i+1}")
            if header_level == level:
                # 找到下一个同级标题或文档结尾
                next_section_start = i + 1
                while next_section_start < end:
                    next_line = lines[next_section_start]
                    next_header = re.match(r'^(#+)\s*', next_line)
                    if next_header and len(next_header.group(1)) <= level:
                        break
                    next_section_start += 1
                # 递归查找子标题
                content_lines = []
                sub_sections = []
                j = i + 1
                while j < next_section_start:
                    sub_header = re.match(r'^(#+)\s*(.*)', lines[j])
                    if sub_header and len(sub_header.group(1)) == level + 1:
                        debug(f"递归进入子标题: {lines[j].strip()} 行号={j+1}")
                        sub_result, sub_end = parse_markdown(lines, level+1, j, next_section_start)
                        sub_sections.extend(sub_result)
                        j = sub_end
                    else:
                        if not re.match(r'^(#+)', lines[j]):
                            content_lines.append(lines[j].strip())
                        j += 1
                section = {
                    "title": title,
                    "content": [c for c in content_lines if c],
                }
                if sub_sections:
                    section["subsections"] = sub_sections
                sections.append(section)
                i = next_section_start
            else:
                # 跳过非本级标题
                i += 1
        else:
            i += 1
    return sections, i

def Process_text(text: str) -> str:
    # 按行分割并过滤空行
    lines = [line for line in text.strip().split('\n')]
    
    # 存储所有区块
    blocks = []
    current_block = None
    current_section_stack = []
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # 检查标题层级
        if line.startswith('#'):
            level = len(line) - len(line.lstrip('#'))
            title = line.lstrip('#').strip()
            
            # 创建新的section
            new_section = Section(title=title)
            
            if level == 1:  # 一级标题
                if current_block:
                    blocks.append(current_block)
                current_block = new_section
                current_section_stack = [current_block]
            else:
                # 回退到正确的层级
                while len(current_section_stack) >= level:
                    current_section_stack.pop()
                
                # 添加到父级section，需保证栈不为空
                if current_section_stack:
                    parent = current_section_stack[-1]
                    parent.subsections.append(new_section)
                    current_section_stack.append(new_section)
                else:
                    # 没有父级，直接跳过或作为一级标题处理（可选）
                    # blocks.append(new_section)  # 如果想保留孤立section可取消注释
                    current_section_stack = [new_section]
            
            i += 1
        else:
            # 处理缩进内容
            if current_section_stack:
                current_indent = get_indent_level(line)
                content, next_idx = process_indented_content(lines, i, current_indent)
                
                if content:
                    # 如果内容有缩进，创建新的子section
                    if current_indent > 0:
                        new_section = Section(title=content[0], content=content[1:])
                        current_section_stack[-1].subsections.append(new_section)
                    else:
                        # 没有缩进的内容直接添加到当前section
                        current_section_stack[-1].content.extend(content)
                
                i = next_idx
            else:
                i += 1
    
    # 添加最后一个区块
    if current_block:
        blocks.append(current_block)
    
    # 转换为JSON
    result = [block.to_dict() for block in blocks]
    return json.dumps(result, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    with open('content.md', 'r', encoding='utf-8') as f:
        lines = [line.rstrip('\n') for line in f]
    debug(f"总行数: {len(lines)}")
    result, _ = parse_markdown(lines)
    json_str = json.dumps(result, ensure_ascii=False, indent=2)
    print(json_str) 