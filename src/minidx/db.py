"""SQLite。納品時にファイル1つをコピーするだけで済む構成を保つ。

スキーマは条文以外の文書も入る形に一般化してある（`documents` / `chunks(locator)`）。
ただし **locator は人に見せる位置表示であり、引用照合には使わない**。
照合は chunk.id（整数）のみで行う。locator を照合キーにすると表記ゆれで両方向に壊れる。
"""
from __future__ import annotations
import json
import sqlite3
from pathlib import Path

import numpy as np

from .models import Chunk

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
  id INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE, doc_type TEXT NOT NULL,
  version TEXT, effective_date TEXT, source_file TEXT, source_format TEXT,
  image_count INTEGER DEFAULT 0, ingested_at TEXT DEFAULT CURRENT_TIMESTAMP);

CREATE TABLE IF NOT EXISTS chunks (
  id INTEGER PRIMARY KEY,
  document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  locator TEXT NOT NULL, heading TEXT, body TEXT NOT NULL, kind TEXT NOT NULL,
  context TEXT DEFAULT '', embedding BLOB);
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
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


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
                     version: str = "", effective_date: str = "") -> int:
    conn.execute("DELETE FROM documents WHERE name = ?", (name,))
    cur = conn.execute(
        "INSERT INTO documents(name, doc_type, version, effective_date,"
        " source_file, source_format, image_count) VALUES (?,?,?,?,?,?,?)",
        (name, doc_type, version, effective_date, source_file, source_format, image_count))
    return int(cur.lastrowid)


def insert_chunks(conn: sqlite3.Connection, document_id: int,
                  chunks: list[Chunk], embeddings: list[np.ndarray]) -> None:
    conn.executemany(
        "INSERT INTO chunks(document_id, locator, heading, body, kind, context, embedding)"
        " VALUES (?,?,?,?,?,?,?)",
        [(document_id, c.locator, c.heading, c.body, c.kind, c.context, to_blob(e))
         for c, e in zip(chunks, embeddings)])


def load_all(conn: sqlite3.Connection) -> tuple[list[Chunk], np.ndarray, dict[int, str]]:
    """全チャンクと埋め込み行列を返す。2,000件×1024次元でも約8MB。起動時に丸ごと載せる。"""
    rows = conn.execute(
        "SELECT c.id, c.document_id, c.locator, c.heading, c.body, c.kind, c.context, c.embedding,"
        " d.name AS doc_name FROM chunks c JOIN documents d ON d.id = c.document_id"
        " ORDER BY c.id").fetchall()
    chunks = [Chunk(id=r["id"], document_id=r["document_id"], locator=r["locator"],
                    heading=r["heading"] or "", body=r["body"], kind=r["kind"], context=r["context"] or "") for r in rows]
    matrix = (np.vstack([from_blob(r["embedding"]) for r in rows])
              if rows else np.zeros((0, 1024), dtype=np.float32))
    doc_names = {r["id"]: r["doc_name"] for r in rows}
    return chunks, matrix, doc_names


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
