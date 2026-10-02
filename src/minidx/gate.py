"""二段棄却ゲート。本システムの中核。

このモジュールは generate.py を import しない。
「ゲートの判定もLLMにやらせれば賢くなる」という改修を構造で抑止するため。
"""
from __future__ import annotations
import re
import unicodedata

from .models import Hit


def gate1(hits: list[Hit], oov_ratio: float, tau_cos: float,
          max_oov_ratio: float = 0.5) -> tuple[bool, str, str]:
    """LLMを呼ぶ前の判定。呼ばなければ幻覚は原理的に発生しない。

    戻り値: (通過したか, verdict, 理由)
    """
    if not hits:
        return False, "rejected_low_score", "検索結果が0件"

    # 語彙判定。コーパス語彙という硬い事実に基づくため、標本数に依存しない保証になる
    if oov_ratio > max_oov_ratio:
        return False, "rejected_no_vocab", \
            f"質問の内容語の {oov_ratio:.0%} が規程に存在しない（上限 {max_oov_ratio:.0%}）"

    cos_max = max(h.cosine for h in hits)
    if cos_max < tau_cos:
        return False, "rejected_low_score", f"最上位のコサイン類似度 {cos_max:.3f} < 閾値 {tau_cos}"
    return True, "", ""


def _norm(text: str) -> str:
    """照合用。NFKC＋空白除去。表記ゆれで正しい引用を誤って棄却しないため。"""
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", text))


# 金額らしい表現のみを対象にする。条番号の数字で誤爆させないため、
# 通貨記号か「円」が付くもの、または3桁区切りのカンマを含むものに限る。
MONEY_RE = re.compile(r"[¥￥]\s*[\d,]+|[\d,]{2,}\s*円|\d{1,3}(?:,\d{3})+")


def _monies(text: str) -> set[str]:
    return {re.sub(r"[^\d]", "", m) for m in MONEY_RE.findall(_norm(text))}


def gate2(llm_out: dict, hits: list[Hit]) -> tuple[bool, str, str, list[int]]:
    """生成後の検証。LLMの自己申告を最終判定に使わない。

    1. sufficient=false は棄却方向にのみ使う（true は何も保証しない）
    2. citation_ids が渡した集合の範囲内か（enum制約に加えて実装側でも必ず確認する。
       スキーマを信用して検証を省くと、モデルや実行基盤を替えたときに静かに穴が開く）
    3. 回答中の金額が、引用した条文の中に実在するか

    3が関連性の検証を担う。当初は逐語引用の部分文字列照合で検証する設計だったが、
    実測で qwen2.5:7b が逐語コピーをしないことが判明したため金額照合に切り替えた。
    **限界**: 同一の表の中で行を取り違えた場合（部長の欄ではなくアルバイトの欄を読む等）は、
    どちらの金額も表の中に実在するため検出できない。この盲点は明示的に残っている。
    """
    if not llm_out.get("sufficient", False):
        return False, "rejected_insufficient", "条文から判断できないとLLMが申告", []

    ids = llm_out.get("citation_ids") or []
    if not ids:
        return False, "rejected_bad_citation", "引用が空", []

    valid = range(1, len(hits) + 1)
    chunk_ids: list[int] = []
    for label in ids:
        if not isinstance(label, int) or label not in valid:
            return False, "rejected_bad_citation", f"存在しないラベル番号: {label}", []
        chunk_ids.append(hits[label - 1].chunk.id)

    # 回答に金額が出ているなら、それは引用した条文の中に実在しなければならない
    cited_text = " ".join(hits[i - 1].chunk.body for i in ids)
    unsupported = _monies(llm_out.get("answer", "")) - _monies(cited_text)
    if unsupported:
        return False, "rejected_bad_citation", \
            f"回答中の金額 {sorted(unsupported)} が引用条文に存在しない", []

    return True, "answered", "", chunk_ids
