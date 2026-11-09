"""1問を処理する本体。全経路が query_log に残る。"""
from __future__ import annotations
import re
import time
from dataclasses import replace
from collections.abc import Callable
from pathlib import Path

from . import db
from .config import REGULATIONS_DIR, ROOT, load
from .gate import NumberMatcher, gate1, gate2
from .generate import generate
from .models import Verdict
from .ollama import Ollama
from .retrieve import Retriever
from .synonyms import apply as apply_synonyms
from .models import Chunk, Hit

# 項に分けた条の位置表示（「第10条第2項」）から条の部分（「第10条」）を取り出す
_ARTICLE_BASE_RE = re.compile(r"^(第\d+条(?:の\d+)?)")


def article_groups(chunks: list[Chunk]) -> dict[int, list[Chunk]]:
    """chunk.id → 同じ条に属する条文チャンク（項に分けたもの）の一覧。表は対象外。

    2026-11-09 論点②: 検索は項単位のまま、AI に渡すのと画面に出すのは条全体にする。
    「ただし」の例外の項だけを AI に渡すと、原則と逆の回答になりうるため（記事調査 §2 A）。
    """
    by_key: dict[tuple[int, str], list[Chunk]] = {}
    for c in chunks:
        m = _ARTICLE_BASE_RE.match(c.locator)
        if c.kind == "article" and m:
            by_key.setdefault((c.document_id, m.group(1)), []).append(c)
    return {c.id: sorted(group, key=lambda x: x.id) for group in by_key.values() for c in group}


def merge_split_articles(hits: list[Hit], groups: dict[int, list[Chunk]]) -> list[Hit]:
    """項に分けた条のヒットを条全体に広げる。同じ条が複数ヒットしたら最初の1件にまとめる。

    広げた Hit の chunk.id は、その条で最初にヒットした項の id のまま（引用の記録とゲート2の照合に使う）。
    """
    out: list[Hit] = []
    seen: set[tuple[int, str]] = set()
    for h in hits:
        group = groups.get(h.chunk.id, [h.chunk])
        if len(group) == 1:
            out.append(h)
            continue
        key = (h.chunk.document_id, _ARTICLE_BASE_RE.match(h.chunk.locator).group(1))
        if key in seen:
            continue
        seen.add(key)
        whole = replace(h.chunk, locator=key[1], body="\n".join(c.body for c in group))
        out.append(replace(h, chunk=whole))
    return out


NOT_FOUND_MESSAGE = (
    "社内規程を確認しましたが、ご質問に該当する条文は見つかりませんでした。\n"
    "この内容は未整備の可能性があります。総務ご担当者へ直接ご確認ください。\n"
    "（このご質問は未整備候補として記録しました）"
)


class Pipeline:
    def __init__(self, db_path: Path | None = None) -> None:
        self.cfg = load()
        self.db_path = db_path or ROOT / "data" / "minidx.db"
        self.conn = db.connect(self.db_path)
        chunks, matrix, self.doc_names = db.load_all(self.conn)
        self.chunks_by_id = {c.id: c for c in chunks}
        self.doc_pdfs = db.document_pdfs(self.conn)
        self.retriever = Retriever(
            chunks, matrix,
            rrf_k=self.cfg["retrieve"]["rrf_k"],
            max_per_article=self.cfg["retrieve"]["max_chunks_per_article"])
        self.client = Ollama(self.cfg)
        self.article_groups = article_groups(chunks)
        self.matcher = NumberMatcher(self.cfg["gate"]["number_units"])

    def ask(self, question: str, *, log: bool = True,
            on_step: Callable[[str], None] | None = None) -> Verdict:
        """on_step には進捗の段階（"search" / "generate" / "verify"）が通知される。画面の待ち表示用。"""
        t0 = time.perf_counter()
        g = self.cfg["gate"]
        step = on_step or (lambda _s: None)
        step("search")
        # 論点6: 言い換え対応表で規程の言葉に置き換えてから検索する（質問ログには元の質問を残す）
        rw = apply_synonyms(question, db.load_synonyms(self.conn))
        query = rw.text
        vec = self.client.embed([query])[0]
        hits = self.retriever.search(query, vec, self.cfg["retrieve"]["top_k"])
        top_cos = max((h.cosine for h in hits), default=0.0)

        # --- ゲート1: ここを通らなければ LLM は一度も呼ばれない ---
        passed, verdict, reason = gate1(
            hits, self.retriever.oov_ratio(query), g["tau_cos"], g["max_oov_ratio"])
        if not passed:
            v = Verdict(verdict=verdict, question=question, hits=hits,
                        top_cos=top_cos, reason=reason, synonyms_applied=rw.applied)
            return self._finish(v, t0, log)

        # --- 生成 ---
        step("generate")
        gen_hits = merge_split_articles(hits, self.article_groups)   # 論点②: AI には条全体を渡す
        try:
            out, _stats = generate(self.client, query, gen_hits, self.cfg["prompt"]["system"])
        except Exception as e:  # 時間切れ・JSON の破損など。論点4: 待たせるより断る（未検証の条文は見せない）
            v = Verdict(verdict="rejected_generation_error", question=question, hits=hits,
                        top_cos=top_cos, reason=f"生成に失敗: {type(e).__name__}",
                        synonyms_applied=rw.applied)
            return self._finish(v, t0, log)

        # --- ゲート2: LLMの自己申告を最終判定に使わない ---
        step("verify")
        ok, verdict, reason, chunk_ids = gate2(
            out, gen_hits, self.matcher, self.cfg["gate"]["table_label_min_chars"])
        v = Verdict(verdict=verdict, question=question, hits=hits, top_cos=top_cos,
                    reason=reason,
                    answer=out.get("answer", "") if ok else "",
                    citations=chunk_ids if ok else [],
                    citation_labels=[self.citation_label(c) for c in chunk_ids] if ok else [],
                    synonyms_applied=rw.applied)
        return self._finish(v, t0, log)

    def _finish(self, v: Verdict, t0: float, log: bool) -> Verdict:
        v.latency_ms = int((time.perf_counter() - t0) * 1000)
        if log:
            db.log_query(self.conn, v, self.cfg["gate"]["tau_cos"])
        return v

    def citation_label(self, chunk_id: int) -> str:
        c = self.chunks_by_id[chunk_id]
        return f"{self.doc_names.get(chunk_id, '')} {c.locator} {c.heading}".strip()

    def display_body(self, chunk_id: int) -> str:
        """画面に出す条文。項に分けた条は条全体を返す（論点②）。"""
        return "\n".join(c.body for c in self.article_groups.get(chunk_id, [self.chunks_by_id[chunk_id]]))

    def source_pdf(self, chunk_id: int) -> Path | None:
        """そのチャンクの原本PDF。未登録、またはファイルが消えていれば None。"""
        name = self.doc_pdfs.get(self.chunks_by_id[chunk_id].document_id)
        path = REGULATIONS_DIR / name if name else None
        return path if path and path.exists() else None
