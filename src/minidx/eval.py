"""評価セットの実行。uv run python -m minidx.eval [--split calib|holdout] [--llm MODEL]

要件定義 §5.2 の指標を、回答可能質問と回答不能質問の両方で測る。
- 誤回答率   … 回答不能質問に answered を返した割合（絶対条件 0%）
- 過剰棄却率 … 回答可能質問を棄却した割合（許容 20% 以下）
- 検索再現率 … 回答可能質問で、正解条文が検索上位 k 件に入った割合
- 回答正確性 … answered になった回答可能質問のうち、引用に正解条文を含む割合
- 応答時間   … 1問あたりの中央値と最大値

質問ログ（query_log）には記録しない（log=False）。評価の質問が未整備一覧に混ざらないようにするため。
1問ごとの結果は eval/results/ に JSON Lines で保存する。回答文（規程の内容）を含むので git 管理外。
"""
from __future__ import annotations
import argparse
import copy
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import yaml

from .config import ROOT
from .ollama import Ollama
from .pipeline import Pipeline

EVAL_DIR = ROOT / "eval"
# ゲート1の棄却判定。LLM を呼ばずに終わる経路（gate.py の gate1 が返す判定名）
GATE1_VERDICTS = {"rejected_low_score", "rejected_no_vocab"}
RESULTS_DIR = EVAL_DIR / "results"


@dataclass
class Expected:
    """正解条文。document は規程名の一部（改訂日の表記ゆれを吸収するため部分一致）。"""
    document: str
    locator: str          # "第14条"。"第14条 表1" や "第14条第2項" も正解として扱う


@dataclass
class Case:
    id: str
    kind: str             # "positive" | "negative"
    split: str            # "calib" | "holdout"
    question: str
    expected: list[Expected] = field(default_factory=list)   # negative では空


@dataclass
class Result:
    id: str
    kind: str
    split: str
    question: str
    verdict: str          # pipeline の判定。例外で止まった場合は "error"
    top_cos: float
    latency_ms: int
    retrieved: list[str]  # 検索上位の「規程名 条番号」
    cited: list[str]      # 引用した「規程名 条番号」
    hit_expected: bool | None    # 正解条文が検索上位に入ったか（negative では None）
    cited_expected: bool | None  # 引用に正解条文を含むか（negative / 未回答では None）
    reason: str = ""
    answer: str = ""


def locator_matches(actual: str, expected: str) -> bool:
    """"第1条" が "第14条" に前方一致しないよう、区切り（空白・項）まで見る。"""
    return actual == expected or actual.startswith(expected + " ") or actual.startswith(expected + "第")


def load_cases(split: str) -> list[Case]:
    cases: list[Case] = []
    neg = yaml.safe_load((EVAL_DIR / "negative" / "themes.yaml").read_text(encoding="utf-8"))
    for t in neg["themes"]:
        if t["split"] != split:
            continue
        for i, q in enumerate(t["questions"], start=1):
            cases.append(Case(id=f"{t['id']}-{i}", kind="negative", split=t["split"], question=q))
    pos_path = EVAL_DIR / "positive" / "questions.yaml"
    pos = yaml.safe_load(pos_path.read_text(encoding="utf-8")) or {}
    for q in pos.get("questions") or []:
        if q["split"] != split:
            continue
        cases.append(Case(id=q["id"], kind="positive", split=q["split"], question=q["question"],
                          expected=[Expected(**e) for e in q["expected"]]))
    return cases


def run_case(p: Pipeline, case: Case) -> Result:
    try:
        v = p.ask(case.question, log=False)
    except Exception as e:  # 時間切れなど。棄却として数えるが、件数は分けて報告する
        return Result(id=case.id, kind=case.kind, split=case.split, question=case.question,
                      verdict="error", top_cos=0.0, latency_ms=0, retrieved=[], cited=[],
                      hit_expected=None if case.kind == "negative" else False,
                      cited_expected=None, reason=f"{type(e).__name__}: {e}")

    def label(cid: int) -> tuple[str, str]:
        return p.doc_names.get(cid, ""), p.chunks_by_id[cid].locator

    def matches(cid: int) -> bool:
        doc, loc = label(cid)
        return any(e.document in doc and locator_matches(loc, e.locator) for e in case.expected)

    retrieved_ids = [h.chunk.id for h in v.hits]
    hit_expected = cited_expected = None
    if case.kind == "positive":
        hit_expected = any(matches(c) for c in retrieved_ids)
        if v.answered:
            cited_expected = any(matches(c) for c in v.citations)
    return Result(id=case.id, kind=case.kind, split=case.split, question=case.question,
                  verdict=v.verdict, top_cos=round(v.top_cos, 4), latency_ms=v.latency_ms,
                  retrieved=[" ".join(label(c)) for c in retrieved_ids],
                  cited=[" ".join(label(c)) for c in v.citations],
                  hit_expected=hit_expected, cited_expected=cited_expected,
                  reason=v.reason, answer=v.answer)


def rate(num: int, den: int) -> str:
    return f"{num}/{den} = {num / den:.1%}" if den else "対象なし"


def summarize(results: list[Result]) -> str:
    neg = [r for r in results if r.kind == "negative"]
    pos = [r for r in results if r.kind == "positive"]
    pos_answered = [r for r in pos if r.verdict == "answered"]
    # ゲート1で棄却した質問は LLM を呼ばず0秒前後で終わる。混ぜると中央値が実態より速く見えるため、
    # LLM を呼んだ質問（ゲート1を通過したもの）だけで集計する
    lat = [r.latency_ms for r in results if r.verdict not in GATE1_VERDICTS | {"error"}]
    lines = [
        f"誤回答率（回答不能に回答）   : {rate(sum(r.verdict == 'answered' for r in neg), len(neg))}   ← 絶対条件 0%",
        f"過剰棄却率（回答可能を棄却） : {rate(sum(r.verdict != 'answered' for r in pos), len(pos))}   ← 許容 20% 以下",
        f"検索再現率（正解が上位k件）  : {rate(sum(bool(r.hit_expected) for r in pos), len(pos))}   ← 目標 80% 以上",
        f"回答正確性（引用に正解）     : {rate(sum(bool(r.cited_expected) for r in pos_answered), len(pos_answered))}   ← 目標 90% 以上",
    ]
    if lat:
        lines.append(f"応答時間（LLMを呼んだ{len(lat)}問）: 中央値 {statistics.median(lat) / 1000:.1f}秒 / 最大 {max(lat) / 1000:.1f}秒   ← 目標 20秒以内")
    errors = [r for r in results if r.verdict == "error"]
    if errors:
        lines.append(f"例外で止まった質問           : {len(errors)}件（棄却として数えた）")
    for kind in ("negative", "positive"):
        counts: dict[str, int] = {}
        for r in results:
            if r.kind == kind:
                counts[r.verdict] = counts.get(r.verdict, 0) + 1
        if counts:
            lines.append(f"判定の内訳（{'回答不能' if kind == 'negative' else '回答可能'}）: "
                         + ", ".join(f"{k} {n}" for k, n in sorted(counts.items())))
    wrong = [r for r in neg if r.verdict == "answered"]
    if wrong:
        lines.append("★ 回答不能質問に回答したもの（誤回答）:")
        lines += [f"    {r.id}: {r.question}  （引用: {', '.join(r.cited)}）" for r in wrong]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="評価セットを実行する")
    ap.add_argument("--split", choices=["calib", "holdout"], default="calib")
    ap.add_argument("--llm", help="config.yaml の ollama.llm をこの実行だけ差し替える（モデル比較用）")
    ap.add_argument("--negative-only", action="store_true", help="回答不能質問だけを実行する")
    args = ap.parse_args(argv)

    if args.split == "holdout":
        # eval/README.md: holdout は一度だけ。見た後に閾値を触ったら burned
        print("★ holdout は一度だけ測る取り決めです（eval/README.md）。"
              "この結果を見た後に閾値・プロンプトを変えたら、このホールドアウトは使えなくなります。",
              file=sys.stderr)

    cases = load_cases(args.split)
    if args.negative_only:
        cases = [c for c in cases if c.kind == "negative"]
    if not cases:
        print("対象の質問がありません。", file=sys.stderr)
        return 1

    p = Pipeline()
    if args.llm:
        cfg = copy.deepcopy(p.cfg)
        cfg["ollama"]["llm"] = args.llm
        p.client = Ollama(cfg)
    model = p.client.llm
    # 画面と同じく、計測の前にモデルを読み込んでおく。読み込み時間（約90秒）が最初の1問に混ざるのを防ぐ
    p.client.warmup()

    n_pos = sum(c.kind == "positive" for c in cases)
    print(f"評価: split={args.split} / LLM={model} / 回答不能 {len(cases) - n_pos}問・回答可能 {n_pos}問")
    results: list[Result] = []
    t0 = time.perf_counter()
    for i, case in enumerate(cases, start=1):
        r = run_case(p, case)
        results.append(r)
        mark = "OK" if (r.verdict == "answered") == (case.kind == "positive") else "NG"
        print(f"  [{i:>3}/{len(cases)}] {mark} {case.kind[:3]} {r.verdict:<22} cos={r.top_cos:.3f} "
              f"{r.latency_ms / 1000:>5.1f}s  {case.id}", flush=True)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = RESULTS_DIR / f"{stamp}_{args.split}_{model.replace(':', '-')}.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")

    print("\n" + "=" * 70)
    print(f"LLM={model} / tau_cos={p.cfg['gate']['tau_cos']} / 所要 {time.perf_counter() - t0:.0f}秒")
    print(summarize(results))
    print(f"\n1問ごとの結果: {out.relative_to(ROOT)}（回答文を含むため git 管理外）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
