"""SQLite。納品時にファイル1つをコピーするだけで済む構成を保つ。

スキーマは条文以外の文書も入る形に一般化してある（`documents` / `chunks(locator)`）。
ただし **locator は人に見せる位置表示であり、引用照合には使わない**。
照合は chunk.id（整数）のみで行う。locator を照合キーにすると表記ゆれで両方向に壊れる。
"""
from __future__ import annotations
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .models import Chunk, PdfLocation

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
  id INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE, doc_type TEXT NOT NULL,
  version TEXT, effective_date TEXT, source_file TEXT, source_format TEXT,
  image_count INTEGER DEFAULT 0, ingested_at TEXT DEFAULT CURRENT_TIMESTAMP,
  source_pdf TEXT);                -- 原本表示用のPDF（取込の正本は source_file）

CREATE TABLE IF NOT EXISTS chunks (
  id INTEGER PRIMARY KEY,
  document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  locator TEXT NOT NULL, heading TEXT, body TEXT NOT NULL, kind TEXT NOT NULL,
  context TEXT DEFAULT '', embedding BLOB,
  pdf_loc TEXT);                   -- 原本PDF上の位置（PdfLocation の JSON）。特定できなければ NULL
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(document_id);

CREATE TABLE IF NOT EXISTS query_log (
  id INTEGER PRIMARY KEY, asked_at TEXT DEFAULT CURRENT_TIMESTAMP,
  question TEXT NOT NULL,
  verdict TEXT NOT NULL,          -- answered / rejected_* を分けて持つ（統合しない）
  top_cos REAL, top_bm25 REAL, vocab_hit INTEGER,
  retrieved_chunk_ids TEXT, answer TEXT, citations TEXT,
  latency_ms INTEGER, tau_cos REAL);   -- その時点の閾値も残す。後から再判定できる
"""


def connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Streamlit はセッションごとに別スレッドでスクリプトを走らせる。
    # 接続を共有するため同一スレッド制約を外し、書き込みの直列化は呼び出し側（app.py のロック）が担う
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    # 既定では外部キー制約が無効で、ON DELETE CASCADE が効かず再取込のたびに古いチャンクが残る
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA)
    _migrate(conn)
    return conn


# 後から足した列。CREATE TABLE IF NOT EXISTS は既存のテーブルに列を足さないため、ここで補う
_ADDED_COLUMNS = [("documents", "source_pdf", "TEXT"), ("chunks", "pdf_loc", "TEXT")]


def _migrate(conn: sqlite3.Connection) -> None:
    for table, col, typ in _ADDED_COLUMNS:
        cols = {r["name"] for r in conn.execute(f"PRAGMA table_info({table})")}
        if col not in cols:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {typ}")
    conn.commit()


def to_blob(vec: np.ndarray) -> bytes:
    """L2正規化してから float32 で保存する。正規化済みならコサイン類似度が内積になる。"""
    v = np.asarray(vec, dtype=np.float32)
    n = np.linalg.norm(v)
    if n > 0:
        v = v / n
    return v.tobytes()


def from_blob(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


def replace_document(conn: sqlite3.Connection, *, name: str, doc_type: str,
                     source_file: str, source_format: str, image_count: int,
                     source_pdf: str = "", version: str = "", effective_date: str = "") -> int:
    conn.execute("DELETE FROM documents WHERE name = ?", (name,))
    cur = conn.execute(
        "INSERT INTO documents(name, doc_type, version, effective_date,"
        " source_file, source_format, image_count, source_pdf) VALUES (?,?,?,?,?,?,?,?)",
        (name, doc_type, version, effective_date, source_file, source_format, image_count,
         source_pdf or None))
    return int(cur.lastrowid)


def insert_chunks(conn: sqlite3.Connection, document_id: int,
                  chunks: list[Chunk], embeddings: list[np.ndarray]) -> None:
    conn.executemany(
        "INSERT INTO chunks(document_id, locator, heading, body, kind, context, embedding, pdf_loc)"
        " VALUES (?,?,?,?,?,?,?,?)",
        [(document_id, c.locator, c.heading, c.body, c.kind, c.context, to_blob(e),
          c.pdf_loc.to_json() if c.pdf_loc else None)
         for c, e in zip(chunks, embeddings)])


def load_all(conn: sqlite3.Connection) -> tuple[list[Chunk], np.ndarray, dict[int, str]]:
    """全チャンクと埋め込み行列を返す。2,000件×1024次元でも約8MB。起動時に丸ごと載せる。"""
    rows = conn.execute(
        "SELECT c.id, c.document_id, c.locator, c.heading, c.body, c.kind, c.context, c.embedding,"
        " c.pdf_loc, d.name AS doc_name FROM chunks c JOIN documents d ON d.id = c.document_id"
        " ORDER BY c.id").fetchall()
    chunks = [Chunk(id=r["id"], document_id=r["document_id"], locator=r["locator"],
                    heading=r["heading"] or "", body=r["body"], kind=r["kind"], context=r["context"] or "",
                    pdf_loc=PdfLocation.from_json(r["pdf_loc"]) if r["pdf_loc"] else None)
              for r in rows]
    matrix = (np.vstack([from_blob(r["embedding"]) for r in rows])
              if rows else np.zeros((0, 1024), dtype=np.float32))
    doc_names = {r["id"]: r["doc_name"] for r in rows}
    return chunks, matrix, doc_names


def document_pdfs(conn: sqlite3.Connection) -> dict[int, str]:
    """document_id → 原本PDFのファイル名。PDF が無い規程は含まない。"""
    rows = conn.execute("SELECT id, source_pdf FROM documents WHERE source_pdf IS NOT NULL").fetchall()
    return {r["id"]: r["source_pdf"] for r in rows}


def log_query(conn: sqlite3.Connection, v, tau_cos: float) -> None:
    conn.execute(
        "INSERT INTO query_log(question, verdict, top_cos, top_bm25, vocab_hit,"
        " retrieved_chunk_ids, answer, citations, latency_ms, tau_cos)"
        " VALUES (?,?,?,?,?,?,?,?,?,?)",
        (v.question, v.verdict, v.top_cos,
         max((h.bm25 for h in v.hits), default=0.0),
         1 if v.verdict != "rejected_no_vocab" else 0,
         json.dumps([h.chunk.id for h in v.hits]), v.answer,
         json.dumps(v.citations), v.latency_ms, tau_cos))
    conn.commit()



# --- 管理者画面用（docs/MVP要件定義.md §3.7） ---
# 時刻は DB に UTC（CURRENT_TIMESTAMP）で入っているため、表示用に localtime へ変換して返す

@dataclass
class DocumentRow:
    name: str
    source_file: str
    effective_date: str
    image_count: int
    ingested_at: str
    articles: int
    tables: int
    source_pdf: str
    located: int          # 原本PDF上で位置を特定できたチャンク数


@dataclass
class UnansweredRow:
    question: str
    count: int
    last_asked: str
    verdicts: str         # その質問が受けた棄却判定（カンマ区切り）


@dataclass
class AnsweredRow:
    asked_at: str
    question: str
    answer: str
    citations: list[int]  # chunk.id。表示側で条文名に変換する
    latency_ms: int


def documents_summary(conn: sqlite3.Connection) -> list[DocumentRow]:
    """取込状況。登録済み規程ごとのチャンク数と取込日時。"""
    rows = conn.execute(
        "SELECT d.name, d.source_file, COALESCE(d.effective_date, '') AS effective_date,"
        " d.image_count, datetime(d.ingested_at, 'localtime') AS ingested_at,"
        " COALESCE(SUM(c.kind = 'article'), 0) AS articles,"
        " COALESCE(SUM(c.kind = 'table'), 0) AS tables,"
        " COALESCE(d.source_pdf, '') AS source_pdf,"
        " COALESCE(SUM(c.pdf_loc IS NOT NULL), 0) AS located"
        " FROM documents d LEFT JOIN chunks c ON c.document_id = d.id"
        " GROUP BY d.id ORDER BY d.name").fetchall()
    return [DocumentRow(**dict(r)) for r in rows]


def verdict_counts(conn: sqlite3.Connection) -> dict[str, int]:
    """判定ごとの件数。棄却理由を分けて見せる（閾値調整の根拠になるため統合しない）。"""
    rows = conn.execute("SELECT verdict, COUNT(*) AS n FROM query_log GROUP BY verdict").fetchall()
    return {r["verdict"]: r["n"] for r in rows}


def unanswered_summary(conn: sqlite3.Connection) -> list[UnansweredRow]:
    """未整備質問一覧。同じ質問文をまとめ、発生回数の多い順に並べる。"""
    rows = conn.execute(
        "SELECT question, COUNT(*) AS count,"
        " datetime(MAX(asked_at), 'localtime') AS last_asked,"
        " GROUP_CONCAT(DISTINCT verdict) AS verdicts"
        " FROM query_log WHERE verdict != 'answered'"
        " GROUP BY question ORDER BY count DESC, last_asked DESC").fetchall()
    return [UnansweredRow(**dict(r)) for r in rows]


def answered_list(conn: sqlite3.Connection) -> list[AnsweredRow]:
    """回答済み質問一覧。新しい順。"""
    rows = conn.execute(
        "SELECT datetime(asked_at, 'localtime') AS asked_at, question, answer, citations, latency_ms"
        " FROM query_log WHERE verdict = 'answered' ORDER BY asked_at DESC").fetchall()
    return [AnsweredRow(asked_at=r["asked_at"], question=r["question"], answer=r["answer"] or "",
                        citations=json.loads(r["citations"] or "[]"), latency_ms=r["latency_ms"] or 0)
            for r in rows]
