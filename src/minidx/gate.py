"""二段棄却ゲート。本システムの中核。

このモジュールは generate.py を import しない。
「ゲートの判定もLLMにやらせれば賢くなる」という改修を構造で抑止するため。
"""
from __future__ import annotations
import re
import unicodedata
from dataclasses import dataclass

from .models import Hit


def gate1(hits: list[Hit], oov_ratio: float, tau_cos: float,
          max_oov_ratio: float = 0.5) -> tuple[bool, str, str]:
    """LLMを呼ぶ前の判定。呼ばなければ幻覚は原理的に発生しない。

    戻り値: (通過したか, verdict, 理由)
    """
    if not hits:
        return False, "rejected_low_score", "検索結果が0件"

    # 語彙判定。コーパス語彙という硬い事実に基づくため、標本数に依存しない保証になる
    if oov_ratio > max_oov_ratio:
        return False, "rejected_no_vocab", \
            f"質問の内容語の {oov_ratio:.0%} が規程に存在しない（上限 {max_oov_ratio:.0%}）"

    cos_max = max(h.cosine for h in hits)
    if cos_max < tau_cos:
        return False, "rejected_low_score", f"最上位のコサイン類似度 {cos_max:.3f} < 閾値 {tau_cos}"
    return True, "", ""


def _norm(text: str) -> str:
    """照合用。NFKC＋空白除去。表記ゆれで正しい引用を誤って棄却しないため。"""
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", text))


# --- 数値照合（要件 §3.4 ゲート2 / 2026-11-09 論点1で金額から単位付きの数字すべてに拡大） ---

_KANJI_DIGIT = {"〇": 0, "零": 0, "一": 1, "二": 2, "三": 3, "四": 4,
                "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
_KANJI_SMALL = {"十": 10, "百": 100, "千": 1000}
_KANJI_NUM_RE = re.compile(r"[〇零一二三四五六七八九十百千万]+")
# 条番号・項番号・号番号は照合対象から外す。引用の実在性は citation_ids で保証しており、
# 回答に「第14条」と書かれても数量ではないため
_ARTICLE_NO_RE = re.compile(r"第\d+(?:条|項|号|章)(?:の\d+)?")


def _kanji_to_int(s: str) -> int:
    """「二十」「三百五十」「一万二千」程度の漢数字を整数にする。規程の日数・回数の表記を想定。"""
    total, section, digit = 0, 0, 0
    for ch in s:
        if ch in _KANJI_DIGIT:
            digit = _KANJI_DIGIT[ch]
        elif ch in _KANJI_SMALL:
            section += (digit or 1) * _KANJI_SMALL[ch]
            digit = 0
        elif ch == "万":
            total += (section + digit or 1) * 10000
            section, digit = 0, 0
    return total + section + digit


def _to_arabic(text: str) -> str:
    """NFKC で全角を半角に、漢数字を算用数字にそろえる。"""
    return _KANJI_NUM_RE.sub(lambda m: str(_kanji_to_int(m.group())), unicodedata.normalize("NFKC", text))


@dataclass(frozen=True)
class Quantity:
    """単位付きの数。value は区切りのカンマを除いた数字列、unit は表記をそろえた単位。"""
    value: str
    unit: str


@dataclass(frozen=True)
class QuantityMatch:
    quantity: Quantity
    start: int            # 正規化後テキスト中の開始位置（行ラベルとの前後関係の判定に使う）


class NumberMatcher:
    """回答と条文から単位付きの数を取り出す。単位の一覧は config.yaml の gate.number_units。"""

    def __init__(self, units: dict[str, list[str]]) -> None:
        # 表記 → 正規の単位。長い表記から照合する（「時間」を「時」より先に）
        self._canon = {alias: canon for canon, aliases in units.items() for alias in [canon, *aliases]}
        alias_re = "|".join(re.escape(a) for a in sorted(self._canon, key=len, reverse=True))
        num = r"\d+(?:,\d{3})*(?:\.\d+)?"
        # ¥9,999 / 9,999円 / 7日 / 3か月 / 50% など。3桁区切りだけの数（9,999）も金額として扱う
        self._re = re.compile(rf"(?P<yen>[¥￥])(?P<yv>{num})|(?P<v>{num})(?P<u>{alias_re})|(?P<bare>\d{{1,3}}(?:,\d{{3}})+)")

    def normalize(self, text: str) -> str:
        return _ARTICLE_NO_RE.sub(" ", re.sub(r"\s+", "", _to_arabic(text)))

    def find(self, text: str) -> list[QuantityMatch]:
        """text は normalize 済みであること（位置を行ラベルの探索と共有するため）。"""
        out = []
        for m in self._re.finditer(text):
            if m.group("yen"):
                q = Quantity(m.group("yv").replace(",", ""), self._canon.get("円", "円"))
            elif m.group("v"):
                q = Quantity(m.group("v").replace(",", ""), self._canon[m.group("u")])
            else:
                q = Quantity(m.group("bare").replace(",", ""), self._canon.get("円", "円"))
            out.append(QuantityMatch(q, m.start()))
        return out

    def quantities(self, text: str) -> set[Quantity]:
        return {m.quantity for m in self.find(self.normalize(text))}


# --- 表の行ラベル照合（2026-11-09 論点2） ---

@dataclass
class TableRow:
    labels: set[str]                # 行の中の文字のセル（職級名など）とその「・」区切りの各語
    quantities: set[Quantity]


def _table_rows(body: str, matcher: NumberMatcher, min_label_chars: int) -> list[TableRow]:
    """表チャンクの本文（1行1レコード、`列見出し: 値、列見出し: 値`）を行ごとに分解する。

    数を含まないセルの値を行ラベルとみなす（例: 「職級: 主任」の「主任」）。
    「社長・会長・役員」のように1セルに複数の名称がある場合は、各語もラベルにする。
    セル全体の値は長さによらずラベルにする。職級を漢字1文字で略記した表が実在するため
    （2026-11-09、旅費規程の表で2〜4行目が1文字。除外していたため照合が誤って棄却側に倒れた）。
    「・」で区切った一部分は min_label_chars 以上だけをラベルにする（偶然の一致を避けるため）。
    同じ行の文字のセルを順につないだものもラベルにする。職級名を2列に分けて書いた表
    （1列目「主」・2列目「任」で「主任」）が実在するため。結合セルで同じ値が続く場合は1つにまとめる。
    """
    rows = []
    for line in body.splitlines():
        labels: set[str] = set()
        texts: list[str] = []
        for cell in line.split("、"):
            _, _, value = cell.partition(": ")
            value = matcher.normalize(value or cell)
            if not value or matcher.find(value):
                continue
            if not texts or texts[-1] != value:
                texts.append(value)
            labels.add(value)
            for term in re.split(r"[・/／]", value):
                if len(term) >= min_label_chars:
                    labels.add(term)
        if len(texts) > 1:
            labels.add("".join(texts))
        rows.append(TableRow(labels=labels, quantities=matcher.quantities(line)))
    return rows


# 回答を「区分ごとの記述」に区切る文字。カンマは「数字,数字3桁」（9,999）のときだけ3桁区切りとみなし、
# それ以外は区切りとして扱う。小型モデルは全角カンマ「，」で区分を並べることがあり、正規化（NFKC）で「,」になるため
# （2026-11-09: 当初はカンマで区切らず、回答全体が1つの区切りになって誤って棄却側に倒れていた）
_SEGMENT_RE = re.compile(r"[、。;\n]|(?<!\d),|,(?!\d{3}(?!\d))")


def _labels_in(segment: str, all_labels: list[str]) -> set[str]:
    """区切りの中に現れる行ラベル。長いラベルから順に取り、その範囲に含まれる短いラベルは数えない
    （「社長」の中の「長」や、「主任技師」の中の「技師」を別の区分として拾わないため）。"""
    covered = [False] * len(segment)
    found: set[str] = set()
    for lb in all_labels:                       # all_labels は長い順
        start = segment.find(lb)
        while start >= 0:
            end = start + len(lb)
            if not any(covered[start:end]):
                found.add(lb)
                for i in range(start, end):
                    covered[i] = True
            start = segment.find(lb, start + 1)
    return found


def _check_table_rows(answer_norm: str, rows: list[TableRow], matcher: NumberMatcher) -> str:
    """回答を読点・句点で区切り、区切りの中の数と行ラベルが表の同じ行にあるかを確かめる。

    - 区切りの中のすべての行ラベルについて、その行にその数がなければ棄却する
      （「部長と課長は¥9,999」のようなまとめ書きは、片方の行に無ければ棄却側に倒れる）
    - 表のどの行にも無い数は対象外（条文側の数。数値照合で別に検証済み）
    - 区切りの中に行ラベルが無ければ対象外（行を特定する手がかりが無い）
    - 「¥9,999（主任）」のように数が先、ラベルが後に書かれても同じ区切りの中なら照合できる
      （2026-11-09: 当初は「数の直前のラベル」だけを見ており、この書き方を誤って棄却していた）
    戻り値: 問題があれば理由、無ければ空文字
    """
    table_qs = {q for r in rows for q in r.quantities}
    all_labels = sorted({lb for r in rows for lb in r.labels}, key=len, reverse=True)
    if not all_labels:
        return ""
    for segment in _SEGMENT_RE.split(answer_norm):
        qs = [m.quantity for m in matcher.find(segment) if m.quantity in table_qs]
        if not qs:
            continue
        for lb in _labels_in(segment, all_labels):
            for q in qs:
                if not any(lb in r.labels and q in r.quantities for r in rows):
                    return f"回答中の「{lb}」と {q.value}{q.unit} が表の同じ行にない"
    return ""


def gate2(llm_out: dict, hits: list[Hit], matcher: NumberMatcher,
          min_label_chars: int) -> tuple[bool, str, str, list[int]]:
    """生成後の検証。LLMの自己申告を最終判定に使わない。

    1. sufficient=false は棄却方向にのみ使う（true は何も保証しない）
    2. citation_ids が渡した集合の範囲内か（enum制約に加えて実装側でも必ず確認する。
       スキーマを信用して検証を省くと、モデルや実行基盤を替えたときに静かに穴が開く）
    3. 回答中の単位付きの数（円・日・か月・年・回・時間 など）が、引用した条文の中に実在するか
    4. 引用に表がある場合、回答を読点・句点で区切った各部分で、数と行ラベルが表の同じ行にあるか

    3・4が関連性の検証を担う。当初は逐語引用の部分文字列照合で検証する設計だったが、
    実測で qwen2.5:7b が逐語コピーをしないことが判明したため数値照合に切り替えた。
    2026-11-09: 照合対象を金額から単位付きの数すべてに広げ（論点1）、表の行の取り違えの検出を加えた（論点2）。
    **限界**: 数を含まない回答（「支給されない」「承認が必要」など）の内容は検証できない。
    """
    if not llm_out.get("sufficient", False):
        return False, "rejected_insufficient", "条文から判断できないとLLMが申告", []

    ids = llm_out.get("citation_ids") or []
    if not ids:
        return False, "rejected_bad_citation", "引用が空", []

    valid = range(1, len(hits) + 1)
    chunk_ids: list[int] = []
    for label in ids:
        if not isinstance(label, int) or label not in valid:
            return False, "rejected_bad_citation", f"存在しないラベル番号: {label}", []
        chunk_ids.append(hits[label - 1].chunk.id)

    cited = [hits[i - 1].chunk for i in ids]
    answer = llm_out.get("answer", "")

    # 3. 回答に単位付きの数が出ているなら、それは引用した条文の中に実在しなければならない
    cited_qs: set[Quantity] = set()
    for c in cited:
        cited_qs |= matcher.quantities(c.body)
    unsupported = matcher.quantities(answer) - cited_qs
    if unsupported:
        shown = sorted(f"{q.value}{q.unit}" for q in unsupported)
        return False, "rejected_bad_number", f"回答中の数 {shown} が引用条文に存在しない", []

    # 4. 表の行の取り違え（部長の日当に課長の金額を答える等）
    rows = [r for c in cited if c.kind == "table"
            for r in _table_rows(c.body, matcher, min_label_chars)]
    if rows:
        problem = _check_table_rows(matcher.normalize(answer), rows, matcher)
        if problem:
            return False, "rejected_table_row", problem, []

    return True, "answered", "", chunk_ids
