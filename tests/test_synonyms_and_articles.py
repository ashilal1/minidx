"""言い換え対応表（論点6）と、項に分けた条を条全体に広げる処理（論点②）のテスト。データはすべて架空。"""
from __future__ import annotations

from minidx.models import Chunk, Hit
from minidx.pipeline import article_groups, merge_split_articles
from minidx.synonyms import Synonym, apply, validate


# --- 言い換え対応表 ---

def test_apply_replaces_and_reports():
    rw = apply("泊まりの出張の手当は？", [Synonym("泊まり", "宿泊")])
    assert rw.text == "宿泊の出張の手当は？"
    assert rw.applied == ["泊まり→宿泊"]


def test_longer_term_wins():
    rw = apply("泊まり出張の手当", [Synonym("泊まり", "宿泊"), Synonym("泊まり出張", "宿泊を伴う出張")])
    assert rw.text == "宿泊を伴う出張の手当"
    assert rw.applied == ["泊まり出張→宿泊を伴う出張"]


def test_no_match_keeps_question():
    rw = apply("日当はいくら", [Synonym("泊まり", "宿泊")])
    assert rw.text == "日当はいくら" and rw.applied == []


def test_validate_rejects_bad_rows():
    problems = validate([Synonym("泊まり", ""), Synonym("宿泊", "宿泊"),
                         Synonym("ホテル代", "宿泊料"), Synonym("ホテル代", "宿泊費")])
    assert len(problems) == 3


# --- 項に分けた条を条全体に広げる ---

def chunk(id_, locator, body, kind="article", doc=1):
    return Chunk(id=id_, document_id=doc, locator=locator, heading="", body=body, kind=kind)


CHUNKS = [
    chunk(1, "第3条", "原則として支給する。"),
    chunk(2, "第3条第2項", "ただし日帰りの場合は支給しない。"),
    chunk(3, "第4条", "別の条。"),
    chunk(4, "第3条 表1", "区分: 甲地、手当: ¥1,100", kind="table"),
    chunk(5, "第3条", "他の規程の第3条。", doc=2),
]


def hit(c):
    return Hit(chunk=c, cosine=0.9, bm25=1.0, rrf=0.1)


def test_exception_paragraph_expands_to_whole_article():
    groups = article_groups(CHUNKS)
    out = merge_split_articles([hit(CHUNKS[1])], groups)
    assert len(out) == 1
    assert out[0].chunk.locator == "第3条"
    assert "原則として支給する。" in out[0].chunk.body and "ただし" in out[0].chunk.body
    assert out[0].chunk.id == 2          # 引用の記録はヒットした項の id のまま


def test_siblings_are_merged_once_and_others_untouched():
    groups = article_groups(CHUNKS)
    out = merge_split_articles([hit(c) for c in CHUNKS], groups)
    assert [h.chunk.id for h in out] == [1, 3, 4, 5]   # 第3条の2つの項は1件に。表と別規程はそのまま
    assert out[2].chunk.body == CHUNKS[3].body          # 表は広げない
    assert out[3].chunk.body == "他の規程の第3条。"      # 別の規程の同じ条番号は混ぜない
