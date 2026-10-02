"""1問を処理する本体。全経路が query_log に残る。"""
from __future__ import annotations
import time
from collections.abc import Callable
from pathlib import Path

from . import db
from .config import REGULATIONS_DIR, ROOT, load
from .gate import gate1, gate2
from .generate import generate
from .models import Verdict
from .ollama import Ollama
from .retrieve import Retriever

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

    def ask(self, question: str, *, log: bool = True,
            on_step: Callable[[str], None] | None = None) -> Verdict:
        """on_step には進捗の段階（"search" / "generate" / "verify"）が通知される。画面の待ち表示用。"""
        t0 = time.perf_counter()
        g = self.cfg["gate"]
        step = on_step or (lambda _s: None)
        step("search")
        vec = self.client.embed([question])[0]
        hits = self.retriever.search(question, vec, self.cfg["retrieve"]["top_k"])
        top_cos = max((h.cosine for h in hits), default=0.0)

        # --- ゲート1: ここを通らなければ LLM は一度も呼ばれない ---
        passed, verdict, reason = gate1(
            hits, self.retriever.oov_ratio(question), g["tau_cos"], g["max_oov_ratio"])
        if not passed:
            v = Verdict(verdict=verdict, question=question, hits=hits,
                        top_cos=top_cos, reason=reason)
            return self._finish(v, t0, log)

        # --- 生成 ---
        step("generate")
        out, _stats = generate(self.client, question, hits)

        # --- ゲート2: LLMの自己申告を最終判定に使わない ---
        step("verify")
        ok, verdict, reason, chunk_ids = gate2(out, hits)
        v = Verdict(verdict=verdict, question=question, hits=hits, top_cos=top_cos,
                    reason=reason,
                    answer=out.get("answer", "") if ok else "",
                    citations=chunk_ids if ok else [])
        return self._finish(v, t0, log)

    def _finish(self, v: Verdict, t0: float, log: bool) -> Verdict:
        v.latency_ms = int((time.perf_counter() - t0) * 1000)
        if log:
            db.log_query(self.conn, v, self.cfg["gate"]["tau_cos"])
        return v

    def citation_label(self, chunk_id: int) -> str:
        c = self.chunks_by_id[chunk_id]
        return f"{self.doc_names.get(chunk_id, '')} {c.locator} {c.heading}".strip()

    def source_pdf(self, chunk_id: int) -> Path | None:
        """そのチャンクの原本PDF。未登録、またはファイルが消えていれば None。"""
        name = self.doc_pdfs.get(self.chunks_by_id[chunk_id].document_id)
        path = REGULATIONS_DIR / name if name else None
        return path if path and path.exists() else None
