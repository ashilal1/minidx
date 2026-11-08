"""横断的に使うデータ構造。dict を返り値にしないのは、キー名の揺れで後段が静かに壊るため。"""
from __future__ import annotations
import json
from dataclasses import dataclass, field

# PDF 座標の矩形 (left, bottom, right, top)。単位は pt、原点はページ左下
Box = tuple[float, float, float, float]


@dataclass
class PdfLocation:
    """原本PDFの中での位置（pdf_locate.py が取込時に特定する）。"""
    pages: dict[int, list[Box]]   # ページ番号（0始まり）→ 色付けする矩形
    highlighted: bool             # False: 条文の範囲は特定できたが、本文の位置までは特定できなかった

    def to_json(self) -> str:
        return json.dumps({"pages": {str(k): v for k, v in self.pages.items()},
                           "highlighted": self.highlighted})

    @classmethod
    def from_json(cls, s: str) -> PdfLocation:
        d = json.loads(s)
        return cls(pages={int(k): [tuple(b) for b in v] for k, v in d["pages"].items()},
                   highlighted=d["highlighted"])


@dataclass
class Chunk:
    """検索と引用の単位。locator は人に見せる位置表示であり、照合には使わない（照合は chunk_id）。"""
    locator: str          # "第14条" / "第14条 表1 3行目"
    heading: str          # "（日当）" 条見出し。無ければ空
    body: str
    kind: str             # "article" | "table"
    context: str = ""     # 所属条文の導入文。表の行に文脈を与えるために使う
    document_id: int = 0
    id: int = 0
    pdf_loc: PdfLocation | None = None   # 原本PDFが無い、または特定できなかったときは None

    @property
    def search_text(self) -> str:
        """検索用テキスト。

        表の行（「職級: 部長、日当: ¥2,300」）は単体では質問と意味的に遠い。
        所属条文の見出しと導入文を足さないと、金額の入った行が検索に上がってこない。
        表示は body のみなので、文脈の付与が原文表示を汚すことはない。
        """
        return f"{self.heading} {self.context} {self.body}".strip()


@dataclass
class Hit:
    chunk: Chunk
    cosine: float         # ゲート1の主判定に使う唯一のスコア（0〜1、クエリ横断で比較可能）
    bm25: float           # 並べ替えの材料。クエリ間で比較できないため閾値には使わない
    rrf: float


@dataclass
class Verdict:
    """1問の処理結果。全経路を query_log に残すため、棄却理由を分けて持つ。"""
    verdict: str          # answered / rejected_low_score / rejected_no_vocab
                          # / rejected_insufficient / rejected_bad_citation
    question: str
    answer: str = ""
    citations: list[int] = field(default_factory=list)   # chunk.id
    hits: list[Hit] = field(default_factory=list)
    top_cos: float = 0.0
    reason: str = ""
    latency_ms: int = 0

    @property
    def answered(self) -> bool:
        return self.verdict == "answered"
