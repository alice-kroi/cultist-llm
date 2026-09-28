"""《密教模拟器》内容文件的容错 JSON 读取器。

游戏的 content/*.json 并不是严格合法的 JSON，实测存在三类问题：

1. 无引号的键名   —— ``{id: "stagriddles", spec: [...]}``
2. 尾随逗号       —— ``"craftable": false, }``
3. 字符串内裸控制符 —— 描述文本里直接换行 / 制表（未转义）

标准 ``json`` 模块会在这三类位置上抛错（全库 1170 个文件中有 25 个无法直接解析）。
这里用一次性字符扫描做保守修复，再交给 ``json.loads``，修复过程只影响上述三类位置，
不改变任何字符串的内容语义。
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

# 字符串内需要转义的控制字符
_ESCAPE_MAP = {"\r": "\\r", "\n": "\\n", "\t": "\\t"}

# 标识符允许的字符（用于识别无引号键名）
_IDENT_EXTRA = "_$-"

# 尾随逗号：逗号后只跟空白与 } 或 ]
_TRAILING_COMMA_RE = re.compile(r",(\s*[}\]])")


def repair(text: str) -> str:
    """把游戏的非严格 JSON 文本修复为标准 JSON 文本。"""
    out: list[str] = []
    i = 0
    n = len(text)
    in_string = False
    escaped = False
    quote_char = '"'

    while i < n:
        ch = text[i]

        if in_string:
            if escaped:
                out.append(ch)
                escaped = False
            elif ch == "\\":
                out.append(ch)
                escaped = True
            elif ch == quote_char:
                # 统一输出为双引号字符串
                out.append('"')
                in_string = False
            elif ch in _ESCAPE_MAP:
                out.append(_ESCAPE_MAP[ch])
            elif ord(ch) < 0x20:
                out.append("\\u%04x" % ord(ch))
            else:
                out.append(ch)
            i += 1
            continue

        # ——— 字符串之外 ———
        if ch in "\"'":
            in_string = True
            quote_char = ch
            out.append('"')
            i += 1
            continue

        if ch.isalpha() or ch in "_$":
            j = i
            while j < n and (text[j].isalnum() or text[j] in _IDENT_EXTRA):
                j += 1
            ident = text[i:j]
            k = j
            while k < n and text[k] in " \t\r\n":
                k += 1
            # 后面跟着冒号 → 这是一个无引号键名
            out.append('"%s"' % ident if (k < n and text[k] == ":") else ident)
            i = j
            continue

        out.append(ch)
        i += 1

    repaired = "".join(out)
    # 去掉尾随逗号：, 后面只跟空白与 } 或 ]
    return _TRAILING_COMMA_RE.sub(r"\1", repaired)


def load_json(path: str) -> Any:
    """读取单个内容文件；严格解析失败时回退到修复后解析。"""
    with open(path, encoding="utf-8-sig") as fh:
        raw = fh.read()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return json.loads(repair(raw))


def iter_json_files(root: str):
    """按相对路径排序遍历目录下所有 .json 文件，逐个 yield (相对路径, 绝对路径)。"""
    found: list[tuple[str, str]] = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            if name.lower().endswith(".json"):
                abs_path = os.path.join(dirpath, name)
                found.append((os.path.relpath(abs_path, root), abs_path))
    for rel, abs_path in sorted(found):
        yield rel, abs_path