"""回答生成。LLMの役割は「渡した条文の範囲で日本語を整えること」だけ。

答えてよいかの判断はこのモジュールの外（gate.py）にある。
"""
from __future__ import annotations

from .models import Hit
from .ollama import Ollama

# 出力契約。id は表記ゆれを持たない参照、quote は関連性の実質的な検証材料。
# 条番号を文字列で書かせない（「第八条」「第8条」の揺れで照合が両方向に壊れるため）。
# フィールドの宣言順が生成順になる（文法制約付きデコード）。
# sufficient を先頭に置くと、回答を組み立てる前に「答えられるか」を判断してしまい、
# 条文が手元にあるのに false を返す挙動が出る。
# 引用 → 回答 → 判定 の順にして、根拠を書き出してから判断させる。
def build_schema(n: int) -> dict:
    """引用できる条文IDを enum で列挙し、**存在しない引用を文法的に生成不可能**にする。

    当初は `{id, quote}` 形式で逐語引用を必須化し、部分文字列照合で関連性を検証する設計にしたが、
    実測で qwen2.5:7b は逐語コピーをしないことが分かった（離れた箇所を「…」で連結する、
    条番号「第4条」をIDと取り違える等。3件中2件が不一致）。
    このため quote を棄却条件にすると、正しく答えられた質問まで棄却され過剰棄却枠を使い切る。

    enum 制約なら、引用の実在性はスキーマレベルで保証され、出力トークンも約6割減る。
    関連性の検証は「回答中の金額が引用条文に実在するか」（gate.py）が担う。
    """
    return {
        "type": "object",
        "properties": {
            # 宣言順が生成順になる。引用 → 回答 → 判定 の順で根拠を先に書かせる
            "citation_ids": {"type": "array",
                             "items": {"type": "integer", "enum": list(range(1, n + 1))}},
            "answer": {"type": "string"},
            "sufficient": {"type": "boolean"},
        },
        "required": ["citation_ids", "answer", "sufficient"],
    }


# プロンプト本文は config.yaml の prompt.system（AGENTS.md §5）


def build_prompt(question: str, hits: list[Hit], template: str) -> str:
    lines = []
    for i, h in enumerate(hits, start=1):
        head = f"[{i}] {h.chunk.locator}"
        if h.chunk.heading:
            head += f" {h.chunk.heading}"
        lines.append(f"{head}\n{h.chunk.body}")
    return template.format(context="\n\n".join(lines), question=question)


def generate(client: Ollama, question: str, hits: list[Hit],
             template: str) -> tuple[dict, dict[str, int]]:
    return client.generate_json(build_prompt(question, hits, template), build_schema(len(hits)))
