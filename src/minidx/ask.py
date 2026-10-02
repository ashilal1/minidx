"""CLI: uv run python -m minidx.ask "質問文" """
from __future__ import annotations
import sys

from .pipeline import NOT_FOUND_MESSAGE, Pipeline


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if not args:
        print('使い方: uv run python -m minidx.ask "出張の日当はいくらですか"', file=sys.stderr)
        return 1
    question = " ".join(args)
    p = Pipeline()
    v = p.ask(question)

    print(f"\n質問: {question}")
    print(f"判定: {v.verdict}  （最上位cos {v.top_cos:.3f} / {v.latency_ms}ms）")
    if v.reason:
        print(f"理由: {v.reason}")
    print("-" * 60)
    if v.answered:
        print(v.answer)
        print("\n根拠:")
        for cid in v.citations:
            c = p.chunks_by_id[cid]
            print(f"  ■ {p.citation_label(cid)}")
            print(f"    {c.body[:120]}")
    else:
        # 未整備判定では参考条文を出さない。近いものを見せると誤読を誘発するため
        print(NOT_FOUND_MESSAGE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
