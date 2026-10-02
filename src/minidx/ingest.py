"""取込CLI: regulations/*.docx → 中間Markdown → チャンク → 埋め込み → SQLite。

中間Markdownを必ず残すのは、抽出の崩れを人が目視修正して再取込できるようにするため
（議事録の「Markdown形式への変換を推奨」への対応）。2日で完璧なパーサは書けない以上、
これが最大の保険になる。
"""
from __future__ import annotations
import sys
from pathlib import Path

from .config import ROOT, load
from .db import connect, insert_chunks, replace_document
from .docx_ingest import parse
from .ollama import Ollama

SUPPORTED = {".docx"}
REJECTED = {".xlsx": "数表は行と列の取り違えを検証できないため、MVPでは意図的に対象外です",
            ".xls": "数表は行と列の取り違えを検証できないため、MVPでは意図的に対象外です",
            ".pdf": "MVPでは .docx のみを扱います（Word原本のご提供を依頼中）"}


def write_markdown(path: Path, name: str, chunks) -> None:
    """人が目視修正できる中間成果物。ここを直して再取込できる。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# {name}", ""]
    for c in chunks:
        lines += [f"## {c.locator} {c.heading}".rstrip(), "", c.body, ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    cfg = load()
    src_dir = ROOT / "regulations"
    files = sorted(p for p in src_dir.glob("*") if p.is_file() and not p.name.startswith("~$"))
    if not files:
        print(f"取込対象がありません: {src_dir}", file=sys.stderr)
        return 1

    conn = connect(ROOT / "data" / "minidx.db")
    client = Ollama(cfg)
    total = 0
    for path in files:
        ext = path.suffix.lower()
        if ext in REJECTED:
            print(f"[対象外] {path.name}: {REJECTED[ext]}")
            continue
        if ext not in SUPPORTED:
            print(f"[エラー] {path.name}: 未対応の形式です（{ext}）")
            continue

        chunks, report = parse(path, max_chars=cfg["split"]["max_chars"])
        name = path.stem.split("_", 1)[-1]
        print(f"\n[取込] {name}")
        print(f"  チャンク {report.chunks}（条文 {report.articles} / 表の行 {report.table_rows}）")
        for w in report.warnings():
            print(f"  [警告] {w}")
        if not chunks:
            print("  条文を検出できませんでした。中間Markdownを確認してください")
            continue

        write_markdown(ROOT / "data" / "markdown" / f"{name}.md", name, chunks)
        vecs = client.embed([c.search_text for c in chunks])
        doc_id = replace_document(conn, name=name, doc_type="regulation",
                                  source_file=path.name, source_format=ext.lstrip("."),
                                  image_count=report.images)
        insert_chunks(conn, doc_id, chunks, vecs)
        conn.commit()
        total += len(chunks)
        print(f"  登録しました（埋め込み {len(vecs)}件）")

    print(f"\n合計 {total} チャンクを登録しました → data/minidx.db")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
