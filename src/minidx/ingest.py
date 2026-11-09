"""取込CLI: regulations/*.docx → 中間Markdown → チャンク → 埋め込み → SQLite。

2026-11-09 論点5: **regulations/ の中身＝現行版**。取込のたびに DB の規程を全件作り直す。
旧版のファイルを残したまま新しい版を置くと、同じ規程が2つになるのでエラーで止める（DB には触れない）。

同じ名前の PDF（例: 07_旅費規程.docx と 07_旅費規程.pdf、または 07_旅費規程.docx.pdf）があれば、原本表示用として各チャンクの
PDF 上の位置を特定して保存する。PDF からテキストは取り込まない（取込の正本は Word）。

中間Markdownを必ず残すのは、抽出の崩れを人が目視修正して再取込できるようにするため
（議事録の「Markdown形式への変換を推奨」への対応）。2日で完璧なパーサは書けない以上、
これが最大の保険になる。
"""
from __future__ import annotations
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

from .config import REGULATIONS_DIR, ROOT, load
from .db import clear_documents, connect, insert_chunks, insert_document
from .docx_ingest import IngestReport, parse
from .models import Chunk
from .ollama import Ollama
from .pdf_locate import build_index, locate

SUPPORTED = {".docx"}
REJECTED = {".xlsx": "数表は行と列の取り違えを検証できないため、MVPでは意図的に対象外です",
            ".xls": "数表は行と列の取り違えを検証できないため、MVPでは意図的に対象外です"}
PDF = ".pdf"   # 原本表示用。同じ名前の .docx と組にして扱う


# ファイル名「07_旅費規程（2025.11.19改訂）」→ 番号・規程名・改訂表記。番号と括弧は省略可
_FILENAME_RE = re.compile(r"^(?:(?P<no>\d+)_)?(?P<name>.+?)\s*(?:[（(](?P<version>[^（）()]+)[）)])?\s*$")
_DATE_RE = re.compile(r"(\d{4})[./年-](\d{1,2})[./月-](\d{1,2})")


@dataclass
class RegulationFile:
    """ファイル名から読み取った規程の素性。name は改訂日を含まない（同じ規程の判定に使う）。"""
    name: str
    version: str = ""          # 「2025.11.19改訂」。表示と記録に使う
    effective_date: str = ""   # 「2025-11-19」。改訂表記に日付があれば


def parse_filename(stem: str) -> RegulationFile:
    m = _FILENAME_RE.match(stem)
    name = (m.group("name") if m else stem).strip()
    version = (m.group("version") or "").strip() if m else ""
    d = _DATE_RE.search(version)
    effective = f"{int(d[1]):04d}-{int(d[2]):02d}-{int(d[3]):02d}" if d else ""
    return RegulationFile(name=name, version=version, effective_date=effective)


def find_pdf(docx: Path) -> Path | None:
    """原本表示用の PDF。Word から「名前を付けて保存」すると「◯◯.docx.pdf」になることがあるため両方を探す。"""
    for cand in (docx.with_suffix(PDF), docx.with_name(docx.name + PDF)):
        if cand.exists():
            return cand
    return None


def write_markdown(path: Path, name: str, chunks) -> None:
    """人が目視修正できる中間成果物。ここを直して再取込できる。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# {name}", ""]
    for c in chunks:
        lines += [f"## {c.locator} {c.heading}".rstrip(), "", c.body, ""]
    path.write_text("\n".join(lines), encoding="utf-8")


@dataclass
class Parsed:
    path: Path
    meta: RegulationFile
    chunks: list[Chunk]
    report: IngestReport
    pdf: Path | None
    vecs: list = field(default_factory=list)


def main(argv: list[str] | None = None) -> int:
    cfg = load()
    src_dir = REGULATIONS_DIR
    files = sorted(p for p in src_dir.glob("*") if p.is_file() and not p.name.startswith("~$"))
    if not files:
        print(f"取込対象がありません: {src_dir}", file=sys.stderr)
        return 1

    # --- 1. 解析と検査（DB には触れない） ---
    words = [p for p in files if p.suffix.lower() in SUPPORTED]
    paired_pdfs = {pdf for w in words if (pdf := find_pdf(w))}
    for path in files:
        ext = path.suffix.lower()
        if ext == PDF and path not in paired_pdfs:
            print(f"[対象外] {path.name}: 同じ名前の Word がありません。"
                  "PDF は原本の表示用で、条文は Word から取り込みます")
        elif ext in REJECTED:
            print(f"[対象外] {path.name}: {REJECTED[ext]}")
        elif ext not in SUPPORTED and ext != PDF:
            print(f"[エラー] {path.name}: 未対応の形式です（{ext}）")

    by_name: dict[str, list[Path]] = {}
    for w in words:
        by_name.setdefault(parse_filename(w.stem).name, []).append(w)
    dups = {n: ps for n, ps in by_name.items() if len(ps) > 1}
    if dups:
        for n, ps in dups.items():
            print(f"[エラー] 同じ規程「{n}」のファイルが複数あります: {[p.name for p in ps]}", file=sys.stderr)
        print("regulations/ には現行版だけを置いてください（旧版は別の場所へ移す）。DB は変更していません。",
              file=sys.stderr)
        return 1

    parsed: list[Parsed] = []
    for path in words:
        meta = parse_filename(path.stem)
        chunks, report = parse(path, max_chars=cfg["split"]["max_chars"])
        label = f"{meta.name}（{meta.version}）" if meta.version else meta.name
        print(f"\n[解析] {label}")
        print(f"  チャンク {report.chunks}（条文 {report.articles} / 表の行 {report.table_rows}）")
        for w in report.warnings():
            print(f"  [警告] {w}")
        if not chunks:
            print("[エラー] 条文を検出できませんでした。中間Markdownを確認してください。DB は変更していません。",
                  file=sys.stderr)
            return 1
        for c in chunks:
            c.doc_name = meta.name
        pdf = find_pdf(path)
        if pdf:
            sv = cfg["source_view"]
            locs = locate(build_index(pdf), chunks, key_chars=sv["key_chars"],
                          segment_chars=sv["segment_chars"], min_match_ratio=sv["min_match_ratio"])
            for c, loc in zip(chunks, locs):
                c.pdf_loc = loc
            found = [c for c in chunks if c.pdf_loc]
            print(f"  原本PDF {pdf.name}: 位置を特定 {len(found)}/{len(chunks)}"
                  f"（色付けまで {sum(c.pdf_loc.highlighted for c in found)}）")
            missing = [c.locator for c in chunks if not c.pdf_loc]
            if missing:
                print(f"  [警告] 原本PDFで位置を特定できませんでした: {missing}"
                      "（画面では原本を表示せず、取り込んだテキストのみを示します。PDF と Word の版違いの可能性）")
        else:
            print(f"  [警告] 原本PDF（{path.stem}.pdf）がありません。画面では取り込んだテキストのみを示します")
        parsed.append(Parsed(path=path, meta=meta, chunks=chunks, report=report, pdf=pdf))

    # --- 2. 埋め込み（失敗しても DB はそのまま） ---
    client = Ollama(cfg)
    for p in parsed:
        p.vecs = client.embed([c.search_text for c in p.chunks])

    # --- 3. 全件の入れ替え（1トランザクション。途中で失敗すれば元の状態に戻る） ---
    conn = connect(ROOT / "data" / "minidx.db")
    with conn:
        clear_documents(conn)
        for p in parsed:
            doc_id = insert_document(conn, name=p.meta.name, doc_type="regulation",
                                     source_file=p.path.name, source_format=p.path.suffix.lstrip("."),
                                     image_count=p.report.images,
                                     source_pdf=p.pdf.name if p.pdf else "",
                                     version=p.meta.version, effective_date=p.meta.effective_date)
            insert_chunks(conn, doc_id, p.chunks, p.vecs)

    # 中間Markdownも現行版だけにする（フォルダから消した規程の写しを残さない）
    md_dir = ROOT / "data" / "markdown"
    for old in md_dir.glob("*.md"):
        old.unlink()
    for p in parsed:
        write_markdown(md_dir / f"{p.meta.name}.md", p.meta.name, p.chunks)

    total = sum(len(p.chunks) for p in parsed)
    print(f"\n{len(parsed)}規程・合計 {total} チャンクで DB を作り直しました → data/minidx.db")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
