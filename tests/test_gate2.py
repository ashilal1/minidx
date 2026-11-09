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


def test_one_char_cell_label_is_used():
    # 職級を漢字1文字で略記した表（セル全体が1文字）。回答の「甲種」の中の「甲」と照合する
    one = Chunk(id=4, locator="第7条 表1", heading="", kind="table",
                body="区分: 甲、手当: ¥3,300\n区分: 乙、手当: ¥2,200")
    assert run("甲種は¥3,300、乙種は¥2,200です。", [1], one)[0] is True
    assert run("甲種は¥2,200です。", [1], one)[1] == "rejected_table_row"


def test_one_char_split_part_is_not_a_label():
    # 「・」で区切った1文字の部分はラベルにしない（偶然の一致を避ける）。セル全体「丙・丁」はラベルになる
    t = Chunk(id=5, locator="第7条 表2", heading="", kind="table",
              body="区分: 丙・丁、手当: ¥1,100\n区分: 戊地、手当: ¥900")
    assert run("丁の場合も¥900です。", [1], t)[0] is True   # 「丁」単独はラベルではないので照合しない


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


def test_label_split_across_two_columns():
    # 職級名を2列に分けて書いた表（「主」+「任」=「主任」、「係」+「任」=「係任」）。2列目の共通の字に引きずられない
    t = Chunk(id=6, locator="第7条 表3", heading="", kind="table",
              body="職級: 主、職級: 任、手当: ¥3,300\n職級: 係、職級: 任、手当: ¥2,200\n"
                   "職級: 一般職、職級: 一般職、手当: ¥1,100")
    assert run("主任は¥3,300、係任は¥2,200、一般職は¥1,100です。", [1], t)[0] is True
    assert run("主任は¥2,200です。", [1], t)[1] == "rejected_table_row"


def test_amount_before_label_in_same_segment():
    # 数が先、区分名が後の書き方
    assert run("¥3,300（甲地）、¥2,200（乙地）です。", [1], TABLE)[0] is True


def test_neighbor_label_does_not_rescue_wrong_amount():
    # 「甲地: ¥2,200、乙地: ¥1,100」はどちらも1行ずれている。隣の区分名と合っても通さない
    assert run("甲地: ¥2,200、乙地: ¥1,100", [1], TABLE)[1] == "rejected_table_row"


def test_short_label_inside_longer_label_is_ignored():
    # 「主任技師」の中の「技師」を別の区分として数えない
    assert run("主任技師: 9,100円", [1], TABLE_NAMED)[0] is True


def test_fullwidth_comma_separates_items_but_not_thousands():
    # 全角カンマで区分を並べた回答。「¥3,300」の3桁区切りでは区切らない
    assert run("甲地: ¥3,300，乙地: ¥2,200，丙地・丁地: ¥1,100", [1], TABLE)[0] is True
    assert run("甲地: ¥2,200，乙地: ¥3,300", [1], TABLE)[1] == "rejected_table_row"
