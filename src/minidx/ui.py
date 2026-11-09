"""社員画面（app.py）と管理画面（admin.py）で共有する部品。

2026-11-09 論点7: 管理画面を別アプリ（別 URL）に分けた。質問ログには聞きづらい質問も残るため、
社員が使う LAN の URL からは管理画面を開けないようにする。
"""
from __future__ import annotations
import csv
import io
import re

import streamlit as st

from . import db
from .config import ROOT, load
from .pipeline import Pipeline

cfg = load()
DB_PATH = ROOT / "data" / "minidx.db"

# 判定の表示名。棄却理由は統合せずに分けて見せる（要件 §3.8）
VERDICT_LABELS = {
    "answered": "回答",
    "rejected_low_score": "棄却（類似度不足）",
    "rejected_no_vocab": "棄却（規程に無い語）",
    "rejected_insufficient": "棄却（LLMが根拠不足と申告）",
    "rejected_bad_citation": "棄却（引用の検証に失敗）",
    "rejected_bad_number": "棄却（回答の数が条文に無い）",
    "rejected_table_row": "棄却（表の行の取り違え）",
    "rejected_generation_error": "棄却（生成の失敗・時間切れ）",
}

_MD_SPECIAL = re.compile(r"([\\`*_{}\[\]()#+\-.!|~$<>])")


def plain(text: str) -> str:
    """LLM の出力や条文を Markdown として解釈させずに表示する。

    `~` が取り消し線に、`$` が数式になるなど、原文と違う見た目になるのを防ぐ。改行は保持する。
    """
    return _MD_SPECIAL.sub(r"\\\1", text).replace("\n", "  \n")


def to_csv(rows: list[dict]) -> bytes:
    """Excel でそのまま開けるよう BOM 付き UTF-8 にする。"""
    if not rows:
        return b""
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue().encode("utf-8-sig")


def corpus_signature() -> str:
    conn = db.connect(DB_PATH)
    try:
        return db.corpus_signature(conn)
    finally:
        conn.close()


def citation_title(p: Pipeline, chunk_id: int) -> str:
    """再取込で条文IDが振り直されると、過去ログの引用先は消える。その場合も落とさずに表示する。"""
    if chunk_id not in p.chunks_by_id:
        return f"（再取込で削除された条文 #{chunk_id}）"
    return p.citation_label(chunk_id)


def is_local_only() -> bool:
    """このアプリが設置 PC の上（ループバック）だけで待ち受けているか。"""
    return (st.get_option("server.address") or "") in {"localhost", "127.0.0.1", "::1"}
