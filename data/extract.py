"""从《密教模拟器》/《司辰之书》游戏目录抽取全部人类可读文本，产出中英对照语料。

支持两款 Weather Factory 游戏，内容目录结构相同（core/ 英文 + loc_zh-hans/ 简体中文），
只是字段命名略有差异：密教模拟器用 description，司辰之书用 desc（中文侧是纯本地化
覆盖文件，只有 id/label/desc，机制数据要从英文侧取）。

输出（默认写到 data/corpus/）：
  text_units.jsonl    每条「实体 + 字段」一条记录，含 zh / en 两列与 game 标记
  aspect_labels.json  理念(aspect) id → 中英名称，用于构造提示词里的"类别"
  stats.json          规模统计

用法：
    # 只抽《密教模拟器》（行为与旧版一致）
    python extract.py --game-content "<Steam 库>\\steamapps\\common\\Cultist Simulator\\cultistsimulator_Data\\StreamingAssets\\content"

    # 同时抽《司辰之书》，合并进同一份语料
    python extract.py --game-content ... --bh-content "...\\Book of Hours\\bh_Data\\StreamingAssets\\bhcontent"
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cs_json import iter_json_files, load_json  # noqa: E402

# Steam 默认安装位置，仅作占位；游戏装在别处时用 --game-content 指定
DEFAULT_CONTENT = (
    r"<Steam 库>\steamapps\common\Cultist Simulator"
    r"\cultistsimulator_Data\StreamingAssets\content"
)
DEFAULT_BH_CONTENT = (
    r"<Steam 库>\steamapps\common\Book of Hours"
    r"\bh_Data\StreamingAssets\bhcontent"
)

# 每种实体里承载自然语言的字段（按游戏内部命名，大小写与空白会被归一化后匹配）。
# 密教模拟器(CS)与司辰之书(BH)共用同一套实体类型，文本字段略有差异。
TEXT_FIELDS_CS: dict[str, list[str]] = {
    "elements": ["label", "description"],
    "recipes": ["label", "startdescription", "description"],
    "endings": ["label", "description", "flavour"],
    "legacies": ["label", "description", "startdescription"],
    "verbs": ["label", "description"],
    "decks": ["label", "description", "drawmessages", "defaultdrawmessages"],
    "achievements": [
        "label",
        "descriptionunlocked",
        "descriptionlocked",
        "unlockmessage",
    ],
}

# 司辰之书：文本字段叫 desc（密教模拟器叫 description）；中文侧是本地化覆盖，
# 机制字段只在英文侧。这里用逻辑字段名，抽取时通过 FIELD_ALIASES 解析实际键名，
# 输出统一为逻辑名，下游 build_dataset 无需感知两款游戏的字段差异。
TEXT_FIELDS_BH: dict[str, list[str]] = {
    "elements": ["label", "description"],
    "recipes": ["label", "startdescription", "description"],
    "endings": ["label", "description", "flavour"],
    "legacies": ["label", "description", "startdescription"],
    "verbs": ["label", "description"],
    "decks": ["label", "description"],
    "achievements": ["label", "descriptionunlocked"],
}

# 逻辑字段名 → 司辰之书里的实际键名（大小写不敏感，get_field 会归一化匹配）
FIELD_ALIASES: dict[str, str] = {"description": "desc"}

# 开发者内容、调试实体、更新日志等，不属于玩家可读叙事文本
SKIP_STEMS = {
    "versionnews",
    "credits",
    "settings",
    "dicta",
    "xperimental",
    "experimental",
    "tlg",
    "z_levers",
    "legacies",  # 只有 136 字节的桩文件，真内容在 1_legacies.json
    "study_research",
    "1_debug",
    "_debug",
    "__debug",
    "z_legacyrecipepreservation",
    "z_old+work_arcane",
}

# 调试实体 id：_开头、或形如 a_debug 的临时条目
SKIP_ID_RE = re.compile(r"^(_|z_|[a-z]_debug|debug)", re.IGNORECASE)

CJK_RE = re.compile(r"[\u4e00-\u9fff]")
WS_RE = re.compile(r"\s+")


def norm_key(key: str) -> str:
    """键名归一化：去掉全部空白并转小写（游戏里存在 actionId / actionid 混用，以及键名内嵌换行的脏数据）。"""
    return WS_RE.sub("", str(key)).lower()


def get_field(obj: dict, name: str):
    """大小写/空白不敏感地取字段。"""
    target = norm_key(name)
    for k, v in obj.items():
        if norm_key(k) == target:
            return v
    return None


def as_text(value) -> str:
    """把字段值统一成文本；列表/字典按行拼接，非字符串忽略。"""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts = [as_text(v) for v in value]
        return "\n".join(p for p in parts if p)
    if isinstance(value, dict):
        parts = [as_text(v) for v in value.values()]
        return "\n".join(p for p in parts if p)
    return ""


def text_len(text: str) -> int:
    return len(CJK_RE.findall(text))


def is_skipped_file(rel_path: str) -> bool:
    stem = os.path.splitext(os.path.basename(rel_path))[0]
    if stem in SKIP_STEMS:
        return True
    # 其他语言的本地化文件不参与（只取 core 与 loc_zh-hans）
    return stem.endswith(("_spa", "_fre", "_ger", "_jpn", "_rus"))


def index_entities(root: str) -> tuple[dict, collections.Counter]:
    """遍历一个内容根目录，返回 {etype: {id: (entity, source_rel)}} 与 id 冲突计数。

    同一 id 在多个文件中出现时，按文件路径排序后者覆盖前者（与游戏 loading order 一致的近似）。
    """
    index: dict[str, dict[str, tuple[dict, str]]] = collections.defaultdict(dict)
    conflicts: collections.Counter = collections.Counter()

    for rel_path, abs_path in iter_json_files(root):
        if is_skipped_file(rel_path):
            continue
        try:
            obj = load_json(abs_path)
        except Exception as exc:  # noqa: BLE001
            print(f"  [警告] 解析失败，已跳过 {rel_path}: {exc}", file=sys.stderr)
            continue
        if not isinstance(obj, dict):
            continue
        for etype, items in obj.items():
            if not isinstance(items, list):
                continue
            for item in items:
                if not isinstance(item, dict):
                    continue
                eid = item.get("id")
                if not isinstance(eid, str) or not eid:
                    continue
                if etype in index and eid in index[etype]:
                    conflicts[etype] += 1
                index[etype][eid] = (item, rel_path)
    return index, conflicts


def extract_game(
    content_root: str,
    loc_subdir: str,
    text_fields: dict[str, list[str]],
    game: str,
) -> tuple[list[dict], dict[str, dict[str, str]], dict]:
    """抽取单个游戏的内容根目录，返回 (units, aspect_labels, stats)。"""
    core_root = os.path.join(content_root, "core")
    loc_root = os.path.join(content_root, loc_subdir)
    for path in (core_root, loc_root):
        if not os.path.isdir(path):
            raise SystemExit(f"目录不存在：{path}")

    print(f"[{game}] 读取中文内容：{loc_root}")
    zh_index, zh_conf = index_entities(loc_root)
    print(f"[{game}] 读取英文内容：{core_root}")
    en_index, en_conf = index_entities(core_root)

    # ---- 抽取文本单元 ----
    units: list[dict] = []
    stat_etype = collections.Counter()
    stat_field = collections.Counter()
    stat_pairs = collections.Counter()
    aspect_labels: dict[str, dict[str, str]] = {}

    for etype, fields in text_fields.items():
        for eid, (zh_ent, zh_src) in zh_index.get(etype, {}).items():
            if SKIP_ID_RE.match(eid):
                continue
            en_ent = en_index.get(etype, {}).get(eid, (None, None))[0]

            # 记录 label 映射，供提示词使用
            zh_label = as_text(get_field(zh_ent, "label"))
            en_label = as_text(get_field(en_ent, "label")) if en_ent else ""

            # 理念(理念/属性) id → 名称；elements 中 isAspect 为真者。
            # 中文侧是纯本地化覆盖（无机制字段），所以 isAspect 得看英文侧。
            if etype == "elements" and eid:
                flag = get_field(en_ent or zh_ent, "isAspect")
                if flag is True and zh_label:
                    aspect_labels[eid] = {"zh": zh_label, "en": en_label or eid}

            ctx: dict = {}
            # 机制字段（aspects / actionid）以英文侧为准，中文侧缺失时回退中文侧
            src = en_ent if en_ent is not None else zh_ent
            aspects = get_field(src, "aspects")
            if isinstance(aspects, dict):
                ctx["aspects"] = [k for k, v in aspects.items() if v]
            actionid = get_field(src, "actionid")
            if isinstance(actionid, str) and actionid:
                ctx["actionid"] = actionid
            ctx["actionid_en"] = actionid or ""

            for field in fields:
                # 逻辑字段名可能对应不同实际键名（司辰之书 desc ↔ 密教 description），
                # 先按逻辑名取，取不到再按别名取；输出统一用逻辑名
                actual = field
                zh_text = as_text(get_field(zh_ent, actual))
                if not zh_text and field in FIELD_ALIASES:
                    actual = FIELD_ALIASES[field]
                    zh_text = as_text(get_field(zh_ent, actual))
                if not zh_text:
                    continue
                en_text = as_text(get_field(en_ent, actual)) if en_ent else ""
                units.append(
                    {
                        "etype": etype,
                        "id": eid,
                        "field": norm_key(field),
                        "source": zh_src.replace("\\", "/"),
                        "game": game,
                        "zh": zh_text,
                        "en": en_text,
                        "zh_label": zh_label,
                        "en_label": en_label,
                        "ctx": ctx,
                    }
                )
                stat_etype[etype] += 1
                stat_field[f"{etype}.{norm_key(field)}"] += 1

    # 去重：同 etype 内 (field, zh) 完全相同的保留首条（游戏里 alt 结果常有重复文本）
    seen: set[tuple[str, str, str]] = set()
    deduped: list[dict] = []
    dup = 0
    for u in units:
        key = (u["etype"], u["field"], u["zh"])
        if key in seen:
            dup += 1
            continue
        seen.add(key)
        deduped.append(u)

    for u in deduped:
        if u["en"]:
            stat_pairs["both"] += 1
        else:
            stat_pairs["zh_only"] += 1

    zh_chars = sum(text_len(u["zh"]) for u in deduped)
    en_chars = sum(len(u["en"]) for u in deduped if u["en"])
    stats = {
        "text_units": len(deduped),
        "deduplicated": dup,
        "zh_cjk_chars": zh_chars,
        "en_chars": en_chars,
        "paired_units": stat_pairs["both"],
        "zh_only_units": stat_pairs["zh_only"],
        "aspects_indexed": len(aspect_labels),
        "by_etype": dict(stat_etype.most_common()),
        "by_field": dict(sorted(stat_field.items())),
        "id_conflicts_zh": dict(zh_conf),
        "id_conflicts_en": dict(en_conf),
    }

    print(f"[{game}] 文本单元        : {len(deduped)}  (去重丢弃 {dup})")
    print(f"[{game}] 中文字符数      : {zh_chars:,}")
    print(f"[{game}] 英文字符数      : {en_chars:,}")
    print(f"[{game}] 中英成对        : {stat_pairs['both']}   仅中文: {stat_pairs['zh_only']}")
    print(f"[{game}] 理念索引        : {len(aspect_labels)}")
    return deduped, aspect_labels, stats


def main() -> None:
    ap = argparse.ArgumentParser(description="抽取《密教模拟器》/《司辰之书》文本为中英对照语料")
    ap.add_argument("--game-content", default=DEFAULT_CONTENT, help="密教模拟器 StreamingAssets/content 目录")
    ap.add_argument("--bh-content", default="", help="司辰之书 StreamingAssets/bhcontent 目录；给则一并抽取")
    ap.add_argument("--loc-subdir", default="loc_zh-hans", help="中文本地化子目录名")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "corpus"))
    args = ap.parse_args()

    games: list[tuple[str, str, dict[str, list[str]]]] = [
        (args.game_content, "cs", TEXT_FIELDS_CS),
    ]
    if args.bh_content:
        games.append((args.bh_content, "bh", TEXT_FIELDS_BH))

    all_units: list[dict] = []
    aspect_labels: dict[str, dict[str, str]] = {}
    per_game: dict[str, dict] = {}
    for content_root, game, text_fields in games:
        units, labels, stats = extract_game(content_root, args.loc_subdir, text_fields, game)
        all_units.extend(units)
        aspect_labels.update(labels)
        per_game[game] = stats

    os.makedirs(args.out, exist_ok=True)

    units_path = os.path.join(args.out, "text_units.jsonl")
    with open(units_path, "w", encoding="utf-8") as fh:
        for u in all_units:
            fh.write(json.dumps(u, ensure_ascii=False) + "\n")

    with open(os.path.join(args.out, "aspect_labels.json"), "w", encoding="utf-8") as fh:
        json.dump(aspect_labels, fh, ensure_ascii=False, indent=2)

    stats = {
        "games": {g: per_game[g] for g in per_game},
        "text_units": len(all_units),
    }
    with open(os.path.join(args.out, "stats.json"), "w", encoding="utf-8") as fh:
        json.dump(stats, fh, ensure_ascii=False, indent=2)

    print("\n===== 抽取完成 =====")
    print(f"游戏            : {', '.join(per_game)}")
    print(f"文本单元合计    : {len(all_units):,}")
    print(f"理念索引合计    : {len(aspect_labels)}")
    print(f"\n输出目录        : {args.out}")


if __name__ == "__main__":
    main()