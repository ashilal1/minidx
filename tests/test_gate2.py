"""ゲート2（生成後の機械的検証）のテスト。

条文・金額はすべて架空。実規程の本文はテストに書かない（AGENTS.md §6）。
"""
from __future__ import annotations

import pytest

from minidx.config import load
from minidx.gate import NumberMatcher, gate2
from minidx.models import Chunk, Hit

CFG = load()["gate"]
MATCHER = NumberMatcher(CFG["number_units"])
MIN_LABEL = CFG["table_label_min_chars"]

# 架空の規程
ARTICLE = Chunk(id=1, locator="第5条", heading="（精算）", kind="article",
                body="旅費は、帰着後七日以内に精算しなければならない。長期の場合は毎月25日に精算する。")
TABLE = Chunk(id=2, locator="第8条 表1", heading="（手当）", kind="table",
              body="区分: 甲地、手当: ¥3,300\n区分: 乙地、手当: ¥2,200\n区分: 丙地・丁地、手当: ¥1,100")
TABLE_NAMED = Chunk(id=3, locator="第9条 表1", heading="（宿泊）", kind="table",
                    body="職級: 主任技師、上限: 9,100円\n職級: 技師、上限: 8,100円")


def hits(*chunks: Chunk) -> list[Hit]:
    return [Hit(chunk=c, cosine=0.9, bm25=1.0, rrf=0.1) for c in chunks]


def run(answer: str, ids: list[int], *chunks: Chunk, sufficient: bool = True):
    out = {"citation_ids": ids, "answer": answer, "sufficient": sufficient}
    return gate2(out, hits(*chunks), MATCHER, MIN_LABEL)


# --- 従来からの検証 ---

def test_sufficient_false_is_rejected():
    assert run("分かりません", [1], ARTICLE, sufficient=False)[1] == "rejected_insufficient"


def test_label_out_of_range_is_rejected():
    assert run("7日以内です", [3], ARTICLE)[1] == "rejected_bad_citation"


# --- 論点1: 単位付きの数の照合 ---

@pytest.mark.parametrize("answer", [
    "帰着後7日以内に精算します。",       # 条文は漢数字「七日」
    "帰着後７日以内に精算します。",       # 全角数字
    "長期の場合は毎月25日です。",
])
def test_number_present_in_cited_article_passes(answer):
    assert run(answer, [1], ARTICLE)[0] is True


@pytest.mark.parametrize("answer", [
    "帰着後10日以内に精算します。",       # 条文に無い日数
    "帰着後1週間以内に精算します。",      # 言い換えも条文に無い数として棄却（許容する過剰棄却）
    "精算は3か月以内です。",
])
def test_number_absent_from_cited_article_is_rejected(answer):
    assert run(answer, [1], ARTICLE)[1] == "rejected_bad_number"


def test_money_notations_are_equivalent():
    # 条文は「¥2,200」、回答は「2,200円」
    assert run("乙地は2,200円です。", [1], TABLE)[0] is True


def test_money_not_in_table_is_rejected():
    assert run("乙地は2,700円です。", [1], TABLE)[1] == "rejected_bad_number"


def test_article_number_is_not_treated_as_quantity():
    assert run("第5条により帰着後7日以内です。", [1], ARTICLE)[0] is True


def test_one_char_label_is_ignored():
    # 1文字の語は行ラベルにしない（table_label_min_chars）。誤一致を避けるため
    one = Chunk(id=4, locator="第7条 表1", heading="", kind="table",
                body="区分: 甲、手当: ¥3,300\n区分: 乙、手当: ¥2,200")
    assert run("甲は¥2,200です。", [1], one)[0] is True


def test_number_must_be_in_cited_chunk_not_just_retrieved():
    # 表の金額でも、表を引用していなければ棄却
    assert run("乙地は¥2,200です。", [1], ARTICLE, TABLE)[1] == "rejected_bad_number"


# --- 論点2: 表の行ラベル照合 ---

def test_correct_row_passes():
    assert run("甲地は¥3,300、乙地は¥2,200です。", [1], TABLE)[0] is True


def test_swapped_row_is_rejected():
    v = run("甲地は¥2,200です。", [1], TABLE)
    assert v[1] == "rejected_table_row"


def test_label_inside_combined_cell_passes():
    # 「丙地・丁地」のセルの一方だけを書いた場合
    assert run("丁地は¥1,100です。", [1], TABLE)[0] is True


def test_combined_listing_falls_to_reject_side():
    # 「甲と乙地は¥3,300」: 直前の「乙」で照合され棄却側に倒れる（許容する過剰棄却）
    assert run("甲と乙地は¥3,300です。", [1], TABLE)[1] == "rejected_table_row"


def test_longer_label_wins_over_contained_label():
    # 「主任技師」の中の「技師」を拾って誤判定しない
    assert run("主任技師は9,100円です。", [1], TABLE_NAMED)[0] is True
    assert run("主任技師は8,100円です。", [1], TABLE_NAMED)[1] == "rejected_table_row"


def test_number_without_label_is_checked_only_by_presence():
    assert run("手当は¥1,100〜¥3,300です。", [1], TABLE)[0] is True


def test_label_in_previous_sentence_does_not_bind():
    assert run("甲地の手当があります。金額は¥2,200の場合もあります。", [1], TABLE)[0] is True
