"""原本PDFの中で、取り込んだ条文・表がどこにあるかを特定し、該当ページを描画する。

条文テキストは Word から取り込み（docx_ingest.py）、PDF は利用者に原本を見せるためだけに使う。
PDF からのテキスト抽出は表の行と列が混ざり、存在しない条文が混入するため、取込の正本にしない（9/24ログ）。

位置特定の手順:
1. PDF 全ページの文字を NFKC 正規化＋空白除去して1本の文字列にし、各文字の (ページ, 矩形, 行頭か) を覚える
2. 各条文の開始位置を「行頭にある、その条文の冒頭テキスト（第N条＋本文の書き出し）」で探す。
   「就業規則第17条に基づく」のような本文中の参照を条文の開始と取り違えないため。
   条文は Word の出現順に前から順に探し、次の条文の開始位置までをその条文の範囲とする
3. 範囲の中でチャンクの本文（表なら各セルの値）を順に探し、一致した文字の矩形を色付けに使う

特定できなかったときは None を返す。違うページを「原本」として見せるより、見せない方が安全なため。
"""
from __future__ import annotations
import io
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import pypdfium2 as pdfium
from PIL import Image, ImageDraw

from .models import Box, Chunk, PdfLocation

ARTICLE_KEY_RE = re.compile(r"第\d+条(?:の\d+)?")


def _norm(text: str) -> str:
    """照合用。docx_ingest.norm と同じ規則（NFKC＋空白除去）にそろえる。"""
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", text))


@dataclass
class _Index:
    """PDF 全体を正規化した文字列と、各文字の位置。"""
    text: str
    page: list[int]
    box: list[Box]
    line_start: list[bool]


def build_index(pdf_path: Path) -> _Index:
    chars: list[str] = []
    page: list[int] = []
    box: list[Box] = []
    line_start: list[bool] = []
    pdf = pdfium.PdfDocument(pdf_path)
    try:
        for pno in range(len(pdf)):
            tp = pdf[pno].get_textpage()
            at_start = True
            for i in range(tp.count_chars()):
                u = pdfium.raw.FPDFText_GetUnicode(tp.raw, i)
                if u in (0x0D, 0x0A):
                    at_start = True
                    continue
                if not u or 0xD800 <= u <= 0xDFFF:
                    continue   # サロゲート（BMP外の文字）は捨てる。その行は分割照合に回る
                for c in unicodedata.normalize("NFKC", chr(u)):
                    if c.isspace():
                        continue
                    chars.append(c)
                    page.append(pno)
                    box.append(tuple(tp.get_charbox(i)))
                    line_start.append(at_start)
                    at_start = False
            tp.close()
    finally:
        pdf.close()
    return _Index("".join(chars), page, box, line_start)


def _article_key(locator: str) -> str:
    m = ARTICLE_KEY_RE.match(locator)
    return m.group(0) if m else locator


def _find_line_start(idx: _Index, needle: str, start: int) -> int:
    """行頭から始まる needle の最初の出現位置。無ければ -1。"""
    pos = idx.text.find(needle, start)
    while pos != -1 and not idx.line_start[pos]:
        pos = idx.text.find(needle, pos + 1)
    return pos


def _needles(chunk: Chunk) -> list[str]:
    """チャンクの中で、PDF 上に原文どおり現れるはずの文字列。

    表は取込時に `列見出し: 値、列見出し: 値` へ展開しているため、値だけを取り出す。
    結合セルは python-docx が同じ値を繰り返すので、行内で連続する重複は落とす。
    """
    if chunk.kind != "table":
        return [n for line in chunk.body.split("\n") if (n := _norm(line))]
    out: list[str] = []
    for line in chunk.body.split("\n"):
        prev = None
        for pair in line.split("、"):
            v = _norm(pair.partition(": ")[2])
            if v and v != prev:
                out.append(v)
            prev = v
    return out


def _merge(idx: _Index, positions: list[int]) -> dict[int, list[Box]]:
    """一致した文字の矩形を、同じ行の中で右へ続く限り1つの矩形にまとめる（表は1行が1つの帯になる）。"""
    pages: dict[int, list[Box]] = {}
    cur: list[float] | None = None
    cur_page = -1
    for p in positions:
        l, b, r, t = idx.box[p]
        # 「・」「「」は文字の矩形が行の高さより小さいので、下端の一致ではなく縦方向の重なりで同じ行とみなす
        same_line = False
        if cur is not None and idx.page[p] == cur_page and l >= cur[0]:
            overlap = min(t, cur[3]) - max(b, cur[1])
            same_line = overlap >= min(t - b, cur[3] - cur[1]) * 0.5
        if same_line:
            cur[1], cur[2], cur[3] = min(cur[1], b), max(cur[2], r), max(cur[3], t)
            continue
        if cur is not None:
            pages.setdefault(cur_page, []).append(tuple(cur))
        cur, cur_page = [l, b, r, t], idx.page[p]
    if cur is not None:
        pages.setdefault(cur_page, []).append(tuple(cur))
    return pages


def locate(idx: _Index, chunks: list[Chunk], *, key_chars: int, segment_chars: int,
           min_match_ratio: float) -> list[PdfLocation | None]:
    """chunks と同じ順・同じ長さで位置を返す。"""
    # 1. 条文ごとの冒頭テキスト（その条文で最初に現れる article チャンクの1行目）
    heads: dict[str, str] = {}
    for c in chunks:
        k = _article_key(c.locator)
        if c.kind == "article" and k not in heads:
            heads[k] = _norm(c.body.split("\n", 1)[0])[:key_chars]

    # 2. Word の出現順に前から探し、条文の範囲 [開始, 次の条文の開始) を決める
    starts: dict[str, int] = {}
    cursor = 0
    for k, head in heads.items():
        pos = _find_line_start(idx, head, cursor)
        if pos != -1:
            starts[k] = pos
            cursor = pos + 1
    ordered = sorted(starts.values())
    ranges = {k: (s, next((o for o in ordered if o > s), len(idx.text))) for k, s in starts.items()}

    # 3. 範囲の中で本文を順に探す
    out: list[PdfLocation | None] = []
    for c in chunks:
        rng = ranges.get(_article_key(c.locator))
        if rng is None:
            out.append(None)
            continue
        s, e = rng
        positions: list[int] = []
        total = 0
        cur = s
        for n in _needles(c):
            total += len(n)
            pos = idx.text.find(n, cur, e)
            if pos != -1:
                positions += range(pos, pos + len(n))
                cur = pos + len(n)
                continue
            # 行が丸ごと一致しないとき（ページ境界にページ番号が挟まる等）は短く区切って探す
            for i in range(0, len(n), segment_chars):
                seg = n[i:i + segment_chars]
                pos = idx.text.find(seg, cur, e)
                if pos != -1:
                    positions += range(pos, pos + len(seg))
                    cur = pos + len(seg)
        if total and len(positions) / total >= min_match_ratio:
            out.append(PdfLocation(pages=_merge(idx, positions), highlighted=True))
        else:
            # 条文の範囲は特定できたが本文の一致が少ない。条文の開始ページだけを色付けなしで示す
            out.append(PdfLocation(pages={idx.page[s]: []}, highlighted=False))
    return out


def render_page(pdf_path: Path, page_no: int, boxes: list[Box], scale: float) -> bytes:
    """原本の1ページを PNG にし、該当箇所に半透明の色を重ねる。"""
    pdf = pdfium.PdfDocument(pdf_path)
    try:
        page = pdf[page_no]
        height = page.get_height()
        img = page.render(scale=scale).to_pil().convert("RGBA")
    finally:
        pdf.close()
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    pad = 2.0
    for l, b, r, t in boxes:
        # PDF 座標は左下原点、画像は左上原点
        xy = ((l - pad) * scale, (height - t - pad) * scale, (r + pad) * scale, (height - b + pad) * scale)
        draw.rectangle(xy, fill=(255, 214, 0, 80), outline=(230, 120, 0, 255), width=2)
    buf = io.BytesIO()
    Image.alpha_composite(img, overlay).convert("RGB").save(buf, format="PNG")
    return buf.getvalue()
