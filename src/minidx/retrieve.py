"""ハイブリッド検索。BM25（語彙一致）＋ bge-m3（意味一致）。

重要: RRF融合スコアは**並べ替えにのみ**使う。RRFは順位だけを入力に取るため、
最上位スコアは関連性の高低によらず 1/(60+1)+1/(60+1)=0.0328 に張り付く。
関連性が完璧でも全件無関係でも同値になるので、棄却の閾値判定には使えない。
判定に使えるのは絶対スケールを持つコサイン類似度のみ（gate.py）。
"""
from __future__ import annotations
import re
import unicodedata

import numpy as np
from fugashi import Tagger
from rank_bm25 import BM25Okapi

from .models import Chunk, Hit

# 内容語のみを見る。助詞・助動詞・記号は「識別力のある語か」の判定を濁らせる
CONTENT_POS = {"名詞", "動詞", "形容詞", "副詞"}
_tagger = Tagger()


def normalize(text: str) -> str:
    """検索用の正規化。全角英数・記号を統一する。本文の保存には使わない。"""
    return unicodedata.normalize("NFKC", text)


def tokenize(text: str) -> list[str]:
    """BM25用の分かち書き。日本語は空白で区切られないため必須。"""
    return [w.surface for w in _tagger(normalize(text)) if w.surface.strip()]


def content_words(text: str) -> list[str]:
    """内容語だけを取り出す。語彙カバレッジ判定に使う。"""
    out = []
    for w in _tagger(normalize(text)):
        pos = w.feature.pos1
        if pos in CONTENT_POS and len(w.surface) > 1 and not re.fullmatch(r"[\d,.]+", w.surface):
            out.append(w.surface)
    return out


class Retriever:
    def __init__(self, chunks: list[Chunk], matrix: np.ndarray, rrf_k: int = 60,
                 max_per_article: int = 2) -> None:
        self.chunks = chunks
        self.matrix = matrix                      # (N, 1024) L2正規化済み
        self.rrf_k = rrf_k
        self.max_per_article = max_per_article
        self.tokens = [tokenize(c.search_text) for c in chunks]
        self.bm25 = BM25Okapi(self.tokens) if chunks else None
        # コーパス語彙。「質問の語が規程に存在するか」という硬い事実
        self.vocab: set[str] = {t for toks in self.tokens for t in toks}

    def oov(self, question: str) -> tuple[list[str], list[str]]:
        """質問の内容語を、コーパスに「ある語」と「ない語」に分ける。

        頻度による識別力判定は使わない。規程1本のコーパスでは主題語（出張・日当）が
        最頻出になり、最も重要な語が「汎用語」と誤判定されるため。
        コーパスに存在するか否かという二値の事実だけを使う。
        """
        words = content_words(question)
        known = [w for w in words if w in self.vocab]
        unknown = [w for w in words if w not in self.vocab]
        return known, unknown

    def oov_ratio(self, question: str) -> float:
        """未知語の割合。1.0 に近いほど「規程が扱っていない話題」を意味する。"""
        known, unknown = self.oov(question)
        total = len(known) + len(unknown)
        return len(unknown) / total if total else 1.0

    def search(self, question: str, query_vec: np.ndarray, top_k: int = 5) -> list[Hit]:
        if not self.chunks:
            return []
        q = np.asarray(query_vec, dtype=np.float32)
        n = np.linalg.norm(q)
        if n > 0:
            q = q / n
        cos = self.matrix @ q                                  # 正規化済みなので内積＝コサイン
        bm = np.asarray(self.bm25.get_scores(tokenize(question)), dtype=np.float32)

        rank_cos = {i: r for r, i in enumerate(np.argsort(-cos), start=1)}
        rank_bm = {i: r for r, i in enumerate(np.argsort(-bm), start=1)}
        rrf = {i: 1 / (self.rrf_k + rank_cos[i]) + 1 / (self.rrf_k + rank_bm[i])
               for i in range(len(self.chunks))}

        order = sorted(rrf, key=lambda i: -rrf[i])
        hits, per_article = [], {}
        for i in order:
            key = _article_key(self.chunks[i].locator)
            if per_article.get(key, 0) >= self.max_per_article:
                continue      # 同一条文の断片が上位を食い潰すと文脈の多様性が失われる
            per_article[key] = per_article.get(key, 0) + 1
            hits.append(Hit(chunk=self.chunks[i], cosine=float(cos[i]),
                            bm25=float(bm[i]), rrf=rrf[i]))
            if len(hits) >= top_k:
                break
        return hits


def _article_key(locator: str) -> str:
    m = re.match(r"(第\d+条(?:の\d+)?)", locator)
    return m.group(1) if m else locator
