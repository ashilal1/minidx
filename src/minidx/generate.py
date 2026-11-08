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


SYSTEM = """あなたは社内規程の内容だけを答える案内係です。次の規則を必ず守ってください。

1. 以下に示した条文以外の知識を使用してはならない。
2. 一般的な労働基準法や世間の慣行に基づく補足を行ってはならない。
3. sufficient を false にしてよいのは、**示された条文にその事項の記載がない場合だけ**である。
4. 条文に職級・区分・条件による場合分けがある場合は、
   **場合分けごとにすべて列挙して回答する**こと。
   質問が特定の条件（職級など）を指定していないことは、false にする理由にならない。
   **場合分けを列挙して回答できたなら、sufficient は true である。**
5. citation_ids には、根拠とした条文の番号（[1] などの番号）だけを入れること。
6. 回答に金額を書く場合、その金額は citation_ids で挙げた条文の中に必ず書かれていなければならない。
   条文に無い金額を計算したり推測したりしてはならない。
7. 出力は citation_ids → answer → sufficient の順に組み立てること。
   まず根拠となる条文を挙げ、次にその条文だけを使って回答を書き、
   最後に「その回答が挙げた条文だけで成り立っているか」を sufficient で答える。

条文:
{context}

質問: {question}
"""


def build_prompt(question: str, hits: list[Hit]) -> str:
    lines = []
    for i, h in enumerate(hits, start=1):
        head = f"[{i}] {h.chunk.locator}"
        if h.chunk.heading:
            head += f" {h.chunk.heading}"
        lines.append(f"{head}\n{h.chunk.body}")
    return SYSTEM.format(context="\n\n".join(lines), question=question)


def generate(client: Ollama, question: str, hits: list[Hit]) -> tuple[dict, dict[str, int]]:
    return client.generate_json(build_prompt(question, hits), build_schema(len(hits)))
