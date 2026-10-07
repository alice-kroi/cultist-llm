"""把抽取出的中英对照语料构造成多任务指令数据集（LLaMA-Factory alpaca 格式）。

任务设计（每条语料可派生出多个样本）：
  中文/英文 双向生成
    - 卡牌描述      elements.description   ← 卡牌名 + 类别 + 主题
    - 行动开场叙事  recipes.startdescription ← 行动类型 + 事件名
    - 行动结算叙事  recipes.description
    - 结局正文      endings.description
    - 结局余韵      endings.flavour
    - 成就说明      achievements.descriptionunlocked
    - 传承叙述      legacies.description / startdescription
    - 事件牌堆      decks.description
    - 卡牌命名      elements.label          ← 仅主题
  风格化创作
    - 主题创作      以准则（灯/铸/刃…）为主题写卡牌描述
    - 文本续写      给出开头，按原文风格续写完整
  中英互译
    - 中译英 / 英译中

用法：
    python build_dataset.py
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))

# 每款游戏的展示名，用于提示词与系统提示
GAME_ZH = {"cs": "密教模拟器", "bh": "司辰之书"}
GAME_EN = {"cs": "Cultist Simulator", "bh": "Book of Hours"}


def system_prompt(game: str) -> str:
    """按游戏生成系统提示。两款游戏都强调 Weather Factory 的晦涩文风。"""
    zh = GAME_ZH[game]
    en = GAME_EN[game]
    return (
        f"你是《{zh}》（{en}）的文本生成模型。"
        "你精通该游戏晦涩、克制、带着书卷气与不祥暗示的文风，"
        "能够按给定的题材、类别和主题创作中文与英文游戏文本，也能在中英文之间准确互译。"
    )

# 九大准则及扩展准则：作为「主题」出现在提示词中
PRINCIPLES = [
    "lantern", "forge", "edge", "winter", "heart", "grail",
    "moth", "knock", "secrethistories", "sky", "moon", "nectar", "scale",
]

# 卡牌类别：优先从这些 aspect 里取，用于提示词中的「类别」
CARD_KINDS = [
    "lore", "text", "tool", "ingredient", "influence", "memory", "follower",
    "summoned", "location", "vault", "op", "job", "asset", "domicile",
    "desire", "ability", "prisoner", "commission", "project", "article",
    "employment", "acquaintance", "spirit", "currency",
]

CJK_RE = re.compile(r"[\u4e00-\u9fff]")
WS_RE = re.compile(r"\s+")
SENT_SPLIT_RE = re.compile(r"(?<=[。！？…；!?])")
# 游戏文本里的富文本标记（<b> 标注重点、<i> 斜体、<br> 换行）
RT_TAG_RE = re.compile(r"</?(?:br|b|i|em|strong|u|s)\s*/?>", re.IGNORECASE)

# (实体类型, 字段) → 中文提示词 / 英文提示词 模板；{...} 为上下文占位
FIELD_PROMPTS: dict[tuple[str, str], dict[str, tuple[str, str]]] = {
    ("elements", "description"): {
        "zh": (
            "为《{game_zh}》中的一张卡牌撰写描述文本。",
            "语言：中文\n卡牌名称：{zh_label}{kind_zh}{theme_zh}",
        ),
        "en": (
            "Write the description text for a card in {game_en}.",
            "Language: English\nCard name: {en_label}{kind_en}{theme_en}",
        ),
    },
    ("recipes", "startdescription"): {
        "zh": (
            "为《{game_zh}》撰写一次行动的开场叙述文本。",
            "语言：中文\n行动类型：{action_zh}\n事件名称：{zh_label}",
        ),
        "en": (
            "Write the opening narrative text for an action in {game_en}.",
            "Language: English\nAction: {action_en}\nEvent name: {en_label}",
        ),
    },
    ("recipes", "description"): {
        "zh": (
            "为《{game_zh}》撰写一次行动结算后的叙述文本。",
            "语言：中文\n行动类型：{action_zh}\n事件名称：{zh_label}",
        ),
        "en": (
            "Write the narrative text shown after an action resolves in {game_en}.",
            "Language: English\nAction: {action_en}\nEvent name: {en_label}",
        ),
    },
    ("endings", "description"): {
        "zh": (
            "撰写《{game_zh}》的一个结局正文。",
            "语言：中文\n结局名称：{zh_label}",
        ),
        "en": (
            "Write the main text of an ending in {game_en}.",
            "Language: English\nEnding name: {en_label}",
        ),
    },
    ("endings", "flavour"): {
        "zh": (
            "撰写《{game_zh}》结局的收尾余韵短句（简洁、克制、有余味）。",
            "语言：中文\n结局名称：{zh_label}",
        ),
        "en": (
            "Write the closing flavour line of an ending in {game_en} (terse, restrained, resonant).",
            "Language: English\nEnding name: {en_label}",
        ),
    },
    ("achievements", "descriptionunlocked"): {
        "zh": (
            "撰写《{game_zh}》中成就解锁时的说明文本。",
            "语言：中文\n成就名称：{zh_label}",
        ),
        "en": (
            "Write the description shown when an achievement unlocks in {game_en}.",
            "Language: English\nAchievement name: {en_label}",
        ),
    },
    ("legacies", "description"): {
        "zh": (
            "撰写《{game_zh}》中一个「传承」的开场叙述（描述新角色接手的局面）。",
            "语言：中文\n传承名称：{zh_label}",
        ),
        "en": (
            "Write the opening narrative for a legacy in {game_en}.",
            "Language: English\nLegacy name: {en_label}",
        ),
    },
    ("legacies", "startdescription"): {
        "zh": (
            "撰写《{game_zh}》中一个「传承」启程时的提示文本。",
            "语言：中文\n传承名称：{zh_label}",
        ),
        "en": (
            "Write the starting prompt text for a legacy in {game_en}.",
            "Language: English\nLegacy name: {en_label}",
        ),
    },
    ("decks", "description"): {
        "zh": (
            "撰写《{game_zh}》中一个事件牌堆的说明文本。",
            "语言：中文\n牌堆名称：{zh_label}",
        ),
        "en": (
            "Write the description text for an event deck in {game_en}.",
            "Language: English\nDeck name: {en_label}",
        ),
    },
    ("verbs", "description"): {
        "zh": (
            "撰写《{game_zh}》中一个行动（verb）的说明文本。",
            "语言：中文\n行动名称：{zh_label}",
        ),
        "en": (
            "Write the description text for a verb in {game_en}.",
            "Language: English\nVerb name: {en_label}",
        ),
    },
}

ACTION_ZH = {
    "study": "研究",
    "dream": "入梦",
    "talk": "交谈",
    "explore": "探索",
    "work": "工作",
    "time": "时间流逝",
    "rite": "仪式",
}

MIN_ZH_CJK = 5
MIN_EN_CHARS = 10
MIN_CONTINUE_CJK = 40
# 单条样本允许的最大字符数（输入+输出）。超出者丢弃：
# 它们会被 cutoff_len 截断，导致模型学到被腰斩的文本。
DEFAULT_MAX_CHARS = 1600


def clean(text: str) -> str:
    return WS_RE.sub(" ", text or "").strip()


def strip_tags(text: str) -> str:
    """剥掉富文本标记，仅用于中英对照不自洽的翻译样本。"""
    return WS_RE.sub(" ", RT_TAG_RE.sub("", text or "")).strip()


def has_tags(text: str) -> bool:
    return bool(RT_TAG_RE.search(text or ""))


def cjk_len(text: str) -> int:
    return len(CJK_RE.findall(text or ""))


def aspect_names(unit: dict, labels: dict) -> list[str]:
    return [labels.get(a, {}).get("zh", "") or a for a in unit["ctx"].get("aspects", [])]


def describe_kind(unit: dict, labels: dict) -> tuple[str, str]:
    """返回 (中文类别串, 英文类别串)，形如 ("\\n类别：文献", "\\nCategory: Text")。"""
    aspects = unit["ctx"].get("aspects", [])
    kind = next((a for a in CARD_KINDS if a in aspects), None)
    if not kind:
        return "", ""
    zh = labels.get(kind, {}).get("zh", "") or kind
    en = labels.get(kind, {}).get("en", "") or kind
    return f"\n类别：{zh}", f"\nCategory: {en}"


def describe_theme(unit: dict, labels: dict) -> tuple[str, str]:
    aspects = unit["ctx"].get("aspects", [])
    hits = [a for a in PRINCIPLES if a in aspects]
    if not hits:
        return "", ""
    zh = "、".join(labels.get(a, {}).get("zh", "") or a for a in hits)
    en = ", ".join(labels.get(a, {}).get("en", "") or a for a in hits)
    return f"\n主题：{zh}", f"\nTheme: {en}"


def make(instruction: str, input_: str, output: str, task: str, lang: str, game: str) -> dict:
    return {
        "instruction": instruction,
        "input": input_.strip(),
        "output": output.strip(),
        "system": system_prompt(game),
        "_task": task,
        "_lang": lang,
    }


def render_values(unit: dict, labels: dict) -> dict:
    """计算模板需要的全部上下文占位值。"""
    game = unit["game"]
    kind_zh, kind_en = describe_kind(unit, labels)
    theme_zh, theme_en = describe_theme(unit, labels)
    # CS 的 actionid 是 study/dream/talk…（有中文映射）；BH 是 salon.*/library.* 等，
    # 没有现成映射时回退用原始 id，至少提示词里有个可读的占位而不是「未知」
    actionid = unit["ctx"].get("actionid") or ""
    action_zh = ACTION_ZH.get(actionid) or actionid or "未知"
    action_en = unit["ctx"].get("actionid_en") or actionid or "unknown"
    return {
        "game_zh": GAME_ZH[game],
        "game_en": GAME_EN[game],
        "zh_label": unit["zh_label"] or unit["id"],
        "en_label": unit["en_label"] or unit["zh_label"] or unit["id"],
        "kind_zh": kind_zh,
        "kind_en": kind_en,
        "theme_zh": theme_zh,
        "theme_en": theme_en,
        "action_zh": action_zh,
        "action_en": action_en,
    }


def render(template: str, unit: dict, labels: dict) -> str:
    return template.format(**render_values(unit, labels))


def build_for_unit(unit: dict, labels: dict) -> list[dict]:
    """把一个文本单元派生成多个训练样本。"""
    out: list[dict] = []
    game = unit["game"]
    zh, en = clean(unit["zh"]), clean(unit["en"])
    key = (unit["etype"], unit["field"])
    zh_ok = cjk_len(zh) >= MIN_ZH_CJK
    en_ok = len(en) >= MIN_EN_CHARS and en.lower() != zh.lower() and en != zh
    values = render_values(unit, labels)

    # ---- 1. 字段驱动的双向生成 ----
    prompts = FIELD_PROMPTS.get(key)
    if prompts:
        instr, tpl = prompts["zh"]
        if prompts["zh"] and zh_ok:
            out.append(make(instr.format(**values), render(tpl, unit, labels), zh, f"gen.{key[0]}.{key[1]}.zh", "zh", game))
        instr_en, tpl_en = prompts["en"]
        if en_ok:
            out.append(make(instr_en.format(**values), render(tpl_en, unit, labels), en, f"gen.{key[0]}.{key[1]}.en", "en", game))

    # ---- 2. 主题创作：以准则为主题写卡牌描述 ----
    if key == ("elements", "description"):
        theme_zh, theme_en = describe_theme(unit, labels)
        if theme_zh and zh_ok:
            out.append(make(
                f"以《{GAME_ZH[game]}》的文风，围绕给定主题写一段卡牌描述文本。",
                f"语言：中文{theme_zh}",
                zh, "theme.elements.zh", "zh", game))
        if theme_en and en_ok:
            out.append(make(
                f"In the style of {GAME_EN[game]}, write a card description on the given theme.",
                f"Language: English{theme_en}",
                en, "theme.elements.en", "en", game))

    # ---- 3. 卡牌命名：由主题推想卡牌名称 ----
    if key == ("elements", "label") and PRINCIPLES:
        theme_zh, theme_en = describe_theme(unit, labels)
        kind_zh, _ = describe_kind(unit, labels)
        if zh_ok and len(zh) <= 30:
            out.append(make(
                f"为《{GAME_ZH[game]}》设计一张新卡牌的名称。",
                f"语言：中文{kind_zh}{theme_zh}",
                zh, "name.elements.zh", "zh", game))

    # ---- 4. 文本续写 ----
    if zh_ok and cjk_len(zh) >= MIN_CONTINUE_CJK:
        prefix, suffix = split_for_continuation(zh)
        if prefix and suffix and cjk_len(suffix) >= MIN_ZH_CJK:
            out.append(make(
                f"下面是一段《{GAME_ZH[game]}》文本的开头，请以原文的风格与语气把它续写完整。",
                prefix, suffix, "continue.zh", "zh", game))

    # ---- 5. 中英互译 ----
    # 坑：中文侧常带 <b> 标注重点，英文原文大多没有对应标记。若一律要求「保留标记」，
    # 参考译文里标记却是没有的 —— 指令与答案自相矛盾，等于在教模型丢掉标记
    # （实测：训到 19% 时翻译输出已完全丢 <b>）。因此只在两侧都带标记时要求保留，
    # 其余样本统一把标记剥掉，让每条翻译样本的指令与参考答案自洽。
    if zh_ok and en_ok:
        keep_tags = has_tags(zh) and has_tags(en)
        src_zh, src_en = (zh, en) if keep_tags else (strip_tags(zh), strip_tags(en))
        if cjk_len(src_zh) >= MIN_ZH_CJK and len(src_en) >= MIN_EN_CHARS:
            tail = "保留 <b>、<i> 等富文本标记与换行结构。" if keep_tags else ""
            out.append(make(
                f"把下面的《{GAME_ZH[game]}》中文文本翻译成英文。{tail}",
                src_zh, src_en, "trans.zh2en", "en", game))
            out.append(make(
                f"把下面的《{GAME_ZH[game]}》英文文本翻译成中文。{tail}",
                src_en, src_zh, "trans.en2zh", "zh", game))
    return out


def split_for_continuation(text: str) -> tuple[str, str]:
    """在句子边界处切分，让前缀约占 35%~60%。"""
    sentences = [s for s in SENT_SPLIT_RE.split(text) if s]
    if len(sentences) < 2:
        return "", ""
    total = len(text)
    acc = 0
    for i, sent in enumerate(sentences[:-1], start=1):
        acc += len(sent)
        ratio = acc / total
        if 0.35 <= ratio <= 0.6:
            return "".join(sentences[:i]), "".join(sentences[i:])
    # 没落在区间内时取最接近 0.45 的切点
    best, best_i, acc = None, 0, 0
    for i, sent in enumerate(sentences[:-1], start=1):
        acc += len(sent)
        diff = abs(acc / total - 0.45)
        if best is None or diff < best:
            best, best_i = diff, i
    return "".join(sentences[:best_i]), "".join(sentences[best_i:])


def stable_split(etype: str, eid: str, eval_ratio: float) -> bool:
    """按 id 稳定划分训练/验证集，保证同一实体的样本不会跨集。"""
    digest = hashlib.md5(f"{etype}:{eid}".encode("utf-8")).hexdigest()
    return (int(digest[:8], 16) / 0xFFFFFFFF) < eval_ratio


def main() -> None:
    ap = argparse.ArgumentParser(description="构造《密教模拟器》多任务指令数据集")
    ap.add_argument("--corpus", default=os.path.join(HERE, "corpus"), help="extract.py 的输出目录")
    ap.add_argument("--out", default=os.path.join(HERE, "dataset"), help="数据集输出目录")
    ap.add_argument("--eval-ratio", type=float, default=0.03, help="验证集比例（按实体 id 划分）")
    ap.add_argument("--no-eval", action="store_true", help="不划分验证集，全部用于训练")
    ap.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS, help="单条样本输入+输出的最大字符数")
    args = ap.parse_args()

    units_path = os.path.join(args.corpus, "text_units.jsonl")
    with open(units_path, encoding="utf-8") as fh:
        units = [json.loads(line) for line in fh if line.strip()]
    with open(os.path.join(args.corpus, "aspect_labels.json"), encoding="utf-8") as fh:
        labels = json.load(fh)

    eval_ratio = 0.0 if args.no_eval else args.eval_ratio

    train: list[dict] = []
    evalset: list[dict] = []
    task_counts: collections.Counter = collections.Counter()
    over_units = 0
    too_long = 0

    for unit in units:
        # 中文侧必须含汉字；否则是未翻译的内部标记条目（如 "DO NOT TRANSLATE"）
        if not CJK_RE.search(unit["zh"]):
            continue
        is_eval = stable_split(unit["etype"], unit["id"], eval_ratio)
        samples = build_for_unit(unit, labels)
        if not samples:
            continue
        over_units += 1
        for s in samples:
            if len(s["input"]) + len(s["output"]) > args.max_chars:
                too_long += 1
                continue
            task_counts[s["_task"]] += 1
            (evalset if is_eval else train).append(s)

    # 样本级去重
    def dedupe(rows: list[dict]) -> tuple[list[dict], int]:
        seen: set[tuple[str, str, str]] = set()
        kept: list[dict] = []
        dropped = 0
        for r in rows:
            sig = (r["instruction"], r["input"], r["output"])
            if sig in seen:
                dropped += 1
                continue
            seen.add(sig)
            kept.append(r)
        return kept, dropped

    train, drop_train = dedupe(train)
    evalset, drop_eval = dedupe(evalset)

    os.makedirs(args.out, exist_ok=True)
    write_alpaca(os.path.join(args.out, "train.json"), train)
    if evalset:
        write_alpaca(os.path.join(args.out, "eval.json"), evalset)
    write_dataset_info(args.out, bool(evalset))

    print("===== 数据集构建完成 =====")
    print(f"覆盖文本单元    : {over_units} / {len(units)}")
    print(f"超长丢弃        : {too_long}  (输入+输出 > {args.max_chars} 字符)")
    print(f"训练样本        : {len(train):,}  (去重丢弃 {drop_train})")
    print(f"验证样本        : {len(evalset):,}  (去重丢弃 {drop_eval})")
    print(f"输出目录        : {args.out}")
    print("\n按任务分布：")
    for task, count in task_counts.most_common():
        print(f"  {task:34s} {count:6d}")
    print("\n按语言分布：")
    lang_counts = collections.Counter(
        r["_lang"] for r in train + evalset
    )
    for lang, count in lang_counts.most_common():
        print(f"  {lang:6s} {count:6d}")


def write_alpaca(path: str, rows: list[dict]) -> None:
    """写出 alpaca 格式；去掉内部以 _ 开头的辅助字段。"""
    payload = [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)


def write_dataset_info(out_dir: str, has_eval: bool) -> None:
    """生成 LLaMA-Factory 的 dataset_info.json。"""
    columns = {
        "prompt": "instruction",
        "query": "input",
        "response": "output",
        "system": "system",
    }
    info = {
        "cultist_cs": {
            "file_name": "train.json",
            "formatting": "alpaca",
            "columns": columns,
        }
    }
    if has_eval:
        info["cultist_cs_eval"] = {
            "file_name": "eval.json",
            "formatting": "alpaca",
            "columns": columns,
        }
    with open(os.path.join(out_dir, "dataset_info.json"), "w", encoding="utf-8") as fh:
        json.dump(info, fh, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()