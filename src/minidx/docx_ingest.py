"""Word(.docx) → チャンク。

実受領ファイル（旅費規程 2025.11.19改訂）の実測に基づく実装。想定と実物の差:
  - 条文は `第 １ 条` のように全角数字とスペースが混在する → NFKC正規化＋空白除去してから判定
  - 条見出し `（目 的）` は条文行ではなく**直前の独立行**にある（`第N条（見出し）` 形式は0件）
  - 条文本文は複数段落に分断されている（文の途中で改行される）
  - 金額は本文ではなく**表**に入っている（日当表=第14条 / 基準宿泊額=第13条 / 交通費=第10条）
  - 画像はテキスト層を持たないため取り込めない → エラーではなく警告として一覧報告する
"""
from __future__ import annotations
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph

from .models import Chunk

# 正規化後の条文見出し。枝番（第8条の2）に対応
ARTICLE_RE = re.compile(r"^第(\d+)条(?:の(\d+))?")
# 章見出し。条文ではないので本文に混ぜない
CHAPTER_RE = re.compile(r"^第(\d+)章")
# 条見出し（独立行の「（...）」のみ）。句点を含む行は本文なので除外
HEADING_RE = re.compile(r"^[（(]([^）)]{1,30})[）)]$")
# 項番号（行頭の算用数字・全角数字）
PARAGRAPH_NO_RE = re.compile(r"^(\d+)[　 \t]")


def norm(text: str) -> str:
    """照合用の正規化。全角→半角、空白をすべて除去する。

    `第 １ 条` → `第1条`。この処理を挟まないと実物の条文は1件も検出できない。
    本文の保存には使わない（原文をそのまま見せる要件 §3.6 があるため）。
    """
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", text))


@dataclass
class IngestReport:
    """取込結果の検証レポート。崩れに気づくための唯一の手段。"""
    source: str
    chunks: int = 0
    articles: int = 0
    table_rows: int = 0   # 表チャンクの数
    images: int = 0                     # テキストとして取り込めなかった画像
    missing_article_nos: list[int] = None   # 欠番（幽霊条文・取りこぼしの兆候）
    duplicate_locators: list[str] = None

    def warnings(self) -> list[str]:
        w = []
        if self.images:
            w.append(
                f"画像 {self.images} 枚はテキストとして取り込めていません。"
                "画像内に条件や金額が書かれている場合、その内容は回答に反映されません。"
            )
        if self.missing_article_nos:
            w.append(f"条番号に欠番があります: {self.missing_article_nos}（分割の取りこぼしの可能性）")
        if self.duplicate_locators:
            w.append(f"位置表示が重複しています: {self.duplicate_locators}（附則と本則の衝突の可能性）")
        return w


def _iter_body(doc: Document):
    """段落と表を本文の出現順に返す。表がどの条文に属するかを知るために必要。"""
    for child in doc.element.body.iterchildren():
        if child.tag.endswith("}p"):
            yield "p", Paragraph(child, doc)
        elif child.tag.endswith("}tbl"):
            yield "t", Table(child, doc)


def _clean(text: str) -> str:
    """セル内のタブ・改行・連続空白を1つの空白に潰す。表示にもBM25にも効く。"""
    return re.sub(r"\s+", " ", text).strip()


def _table_to_chunk(table: Table, locator_base: str, heading: str,
                    seq: int, context: str = "") -> Chunk | None:
    """表を1チャンクにまとめる。行ごとに `列見出し: 値` へ展開する。

    セルを素のテキストとして連結すると行見出しと列見出しの対応が失われ、
    「自転車の行を読んで自動車の金額を答える」類の誤りが起きる。
    行単位に分割する案もあるが、「日当はいくら」のような質問には表全体が必要なため、
    表を1チャンクとし、対応関係は各行の `列見出し: 値` 形式で保存する。
    """
    rows = table.rows
    if len(rows) < 2:
        return None
    header = [_clean(c.text) or f"列{i+1}" for i, c in enumerate(rows[0].cells)]
    lines = []
    for row in rows[1:]:
        pairs = [f"{h}: {v}" for h, cell in zip(header, row.cells)
                 if (v := _clean(cell.text))]
        if pairs:
            lines.append("、".join(pairs))
    if not lines:
        return None
    return Chunk(
        locator=f"{locator_base} 表{seq}",
        heading=heading,
        body="\n".join(lines),
        kind="table",
        context=f"{locator_base} {context}".strip(),
    )


def parse(path: Path, max_chars: int = 1200) -> tuple[list[Chunk], IngestReport]:
    doc = Document(path)
    report = IngestReport(source=path.name, missing_article_nos=[], duplicate_locators=[])
    report.images = sum(1 for r in doc.part.rels.values() if "image" in r.reltype)

    chunks: list[Chunk] = []
    pending_heading = ""
    cur_locator: str | None = None
    cur_heading = ""
    cur_body: list[str] = []
    table_seq: dict[str, int] = {}

    def flush() -> None:
        if cur_locator is None:
            return
        body = "\n".join(cur_body).strip()
        if not body:
            return
        for c in _split_long(cur_locator, cur_heading, body, max_chars):
            chunks.append(c)

    for kind, el in _iter_body(doc):
        if kind == "t":
            if cur_locator is None:
                continue  # 条文の前にある表（表紙等）は対象外
            table_seq[cur_locator] = table_seq.get(cur_locator, 0) + 1
            intro = _clean(" ".join(cur_body))[:80]
            tc = _table_to_chunk(el, cur_locator, cur_heading, table_seq[cur_locator], intro)
            if tc:
                chunks.append(tc)
            continue

        raw = el.text.strip()
        if not raw:
            continue
        n = norm(raw)

        if CHAPTER_RE.match(n):
            continue

        m = ARTICLE_RE.match(n)
        if m:
            flush()
            cur_locator = f"第{m.group(1)}条" + (f"の{m.group(2)}" if m.group(2) else "")
            cur_heading = pending_heading
            pending_heading = ""
            cur_body = [raw]
            continue

        h = HEADING_RE.match(n)
        if h and "。" not in raw:
            pending_heading = raw
            continue

        if cur_locator is not None:
            cur_body.append(raw)

    flush()

    # --- 検証 ---
    articles = [c for c in chunks if c.kind == "article"]
    report.chunks = len(chunks)
    report.articles = len(articles)
    report.table_rows = sum(1 for c in chunks if c.kind == "table")

    nos = sorted({int(re.match(r"第(\d+)条", c.locator).group(1))
                  for c in articles if re.match(r"第(\d+)条", c.locator)})
    if nos:
        report.missing_article_nos = [i for i in range(nos[0], nos[-1] + 1) if i not in nos]
    seen: set[str] = set()
    for c in chunks:
        if c.locator in seen and c.kind == "article":
            report.duplicate_locators.append(c.locator)
        seen.add(c.locator)
    return chunks, report


def _split_long(locator: str, heading: str, body: str, max_chars: int) -> list[Chunk]:
    """長い条文を項単位に再分割する。位置表示は `第10条第2項` のように引き継ぐ。"""
    if len(body) <= max_chars:
        return [Chunk(locator=locator, heading=heading, body=body, kind="article")]
    out, cur, cur_no = [], [], 1
    for line in body.split("\n"):
        m = PARAGRAPH_NO_RE.match(norm(line))
        if m and cur:
            out.append(Chunk(locator=f"{locator}第{cur_no}項", heading=heading,
                             body="\n".join(cur), kind="article"))
            cur, cur_no = [], int(m.group(1))
        cur.append(line)
    if cur:
        out.append(Chunk(locator=f"{locator}第{cur_no}項" if out else locator,
                         heading=heading, body="\n".join(cur), kind="article"))
    return out
