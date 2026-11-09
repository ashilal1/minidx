"""言い換え対応表（2026-11-09 論点6。10/5 第2回MTG §3.3 の先方要望）。

社員の言葉（例: 泊まり）を規程の言葉（例: 宿泊）に置き換えてから検索する。
AI による質問の書き換えは使わない。遅く、書き換え自体を誤ることがあり、誰の判断かも残らないため。
対応表は管理者が登録し、置き換えたことは社員の画面と質問ログに必ず残す。

**安全上の位置づけ**: 置き換えは検索とゲート1の入力を変える。ゲート2（数値・表の行の照合）は変えない。
誤った対応（例: 在宅勤務 → 出張）を登録すると、本来棄却すべき質問がゲート1を通りうる。
対応表の登録は「この言葉はこの意味とみなす」という管理者の判断であり、その責任は登録者にある。
"""
from __future__ import annotations
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Synonym:
    term: str             # 社員が使う言葉
    replacement: str      # 規程の言葉
    note: str = ""


@dataclass
class Rewrite:
    """置き換えの結果。applied は画面とログに出す「泊まり→宿泊」の一覧。"""
    text: str
    applied: list[str]


def apply(question: str, synonyms: list[Synonym]) -> Rewrite:
    """長い言葉から順に、重ならないように置き換える（「泊まり出張」と「泊まり」が両方あれば前者を優先）。"""
    table = {s.term: s.replacement for s in synonyms if s.term and s.replacement and s.term != s.replacement}
    if not table:
        return Rewrite(question, [])
    pattern = re.compile("|".join(re.escape(t) for t in sorted(table, key=len, reverse=True)))
    applied: list[str] = []

    def repl(m: re.Match) -> str:
        label = f"{m.group()}→{table[m.group()]}"
        if label not in applied:
            applied.append(label)
        return table[m.group()]

    return Rewrite(pattern.sub(repl, question), applied)


def validate(rows: list[Synonym]) -> list[str]:
    """管理画面で保存する前の検査。問題があれば理由の一覧を返す（空なら保存してよい）。"""
    problems: list[str] = []
    seen: set[str] = set()
    for i, s in enumerate(rows, start=1):
        if not s.term or not s.replacement:
            problems.append(f"{i}行目: 「社員が使う言葉」と「規程の言葉」の両方を入力してください")
        elif s.term == s.replacement:
            problems.append(f"{i}行目: 同じ言葉どうしは登録できません（{s.term}）")
        elif s.term in seen:
            problems.append(f"{i}行目: 「{s.term}」が重複しています")
        seen.add(s.term)
    return problems
