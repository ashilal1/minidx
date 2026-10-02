"""Ollama クライアント。推論は localhost で完結させる（外部AI APIは呼ばない）。"""
from __future__ import annotations
from typing import Any

import httpx
import numpy as np

from .config import load


class Ollama:
    def __init__(self, cfg: dict[str, Any] | None = None) -> None:
        c = (cfg or load())["ollama"]
        self.host = c["host"]
        self.llm = c["llm"]
        self.embed_model = c["embed"]
        self.num_ctx = c["num_ctx"]
        self.temperature = c["temperature"]
        self.seed = c["seed"]
        self.timeout = c["timeout_sec"]

    def embed(self, texts: list[str]) -> list[np.ndarray]:
        """bge-m3 による埋め込み。クエリ側のプレフィックスは不要なモデル。"""
        r = httpx.post(f"{self.host}/api/embed",
                       json={"model": self.embed_model, "input": texts},
                       timeout=self.timeout)
        r.raise_for_status()
        return [np.asarray(v, dtype=np.float32) for v in r.json()["embeddings"]]

    def generate_json(self, prompt: str, schema: dict[str, Any]) -> tuple[dict[str, Any], dict[str, int]]:
        """JSON Schema を強制した生成。

        num_ctx を明示するのは、既定値に依存するとプロンプト先頭（システムプロンプトと
        最初の根拠条文）が警告なく切り捨てられ、最も危険な幻覚経路になるため。
        """
        r = httpx.post(f"{self.host}/api/generate", timeout=self.timeout, json={
            "model": self.llm, "prompt": prompt, "stream": False, "format": schema,
            "options": {"temperature": self.temperature, "seed": self.seed,
                        "num_ctx": self.num_ctx},
        })
        r.raise_for_status()
        data = r.json()
        import json as _json
        stats = {k: data.get(k, 0) for k in
                 ("prompt_eval_count", "eval_count", "total_duration")}
        return _json.loads(data["response"]), stats
