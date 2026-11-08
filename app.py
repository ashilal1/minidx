"""Streamlit 画面。社員タブ（質問する）と管理者タブ（質問ログ・取込状況）。

起動: uv run streamlit run app.py

★認証は未実装。2026-10-02 のユーザー判断で、ローカル動作の間はログインなしで画面を出す。
  トンネルで公開する前に docs/MVP要件定義.md §7.3 のパスワードゲート（社員用／管理者用を分ける）を
  必ず入れること。それまでは .streamlit/config.toml で localhost のみにバインドしている。
"""
from __future__ import annotations
import csv
import io
import logging
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import streamlit as st

from minidx import db
from minidx.config import load
from minidx.models import Box, PdfLocation
from minidx.pdf_locate import render_page
from minidx.pipeline import NOT_FOUND_MESSAGE, Pipeline

log = logging.getLogger(__name__)
cfg = load()

# 待ち表示の文言。Pipeline.ask の on_step が通知する段階に対応する
STEP_LABELS = {
    "search": "条文を検索しています…",
    "generate": "条文をもとに回答を作成しています…",
    "verify": "回答の根拠を検証しています…",
}

# 管理者画面の判定表示。棄却理由は統合せずに分けて見せる（要件 §3.8）
VERDICT_LABELS = {
    "answered": "回答",
    "rejected_low_score": "棄却（類似度不足）",
    "rejected_no_vocab": "棄却（規程に無い語）",
    "rejected_insufficient": "棄却（LLMが根拠不足と申告）",
    "rejected_bad_citation": "棄却（引用の検証に失敗）",
}


@dataclass
class Card:
    """根拠条文カード1枚。原本PDFがあれば該当ページを、無ければ取り込んだテキストを見せる。"""
    title: str                     # 規程名 第○条 条見出し
    body: str                      # 取り込んだテキスト
    pdf: Path | None = None
    loc: PdfLocation | None = None


@dataclass
class Shown:
    """社員タブに表示する1問ぶんの結果。再描画のたびに session_state から復元する。"""
    question: str
    answered: bool = False
    answer: str = ""
    cards: list[Card] = field(default_factory=list)
    error: bool = False


@st.cache_resource
def get_pipeline() -> Pipeline:
    """条文・埋め込み・BM25 索引はプロセスで1つだけ持つ。再取込後は管理者タブから読み直す。"""
    return Pipeline()


@st.cache_resource
def get_lock() -> threading.Lock:
    """質問の処理を直列化する。8GB 機では LLM の同時実行でメモリが溢れ、SQLite 接続も共有しているため。"""
    return threading.Lock()


_MD_SPECIAL = re.compile(r"([\\`*_{}\[\]()#+\-.!|~$<>])")


def plain(text: str) -> str:
    """LLM の出力や条文を Markdown として解釈させずに表示する。

    `~` が取り消し線に、`$` が数式になるなど、原文と違う見た目になるのを防ぐ。改行は保持する。
    """
    return _MD_SPECIAL.sub(r"\\\1", text).replace("\n", "  \n")


def to_csv(rows: list[dict]) -> bytes:
    """Excel でそのまま開けるよう BOM 付き UTF-8 にする。"""
    if not rows:
        return b""
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue().encode("utf-8-sig")


@st.cache_data(max_entries=64)
def page_image(pdf: str, mtime: float, page_no: int, boxes: tuple[Box, ...]) -> bytes:
    """原本の1ページを描画する。mtime をキーに含め、PDF を差し替えたら描き直す。"""
    return render_page(Path(pdf), page_no, list(boxes), cfg["source_view"]["render_scale"])


def citation_title(p: Pipeline, chunk_id: int) -> str:
    """再取込で条文IDが振り直されると、過去ログの引用先は消える。その場合も落とさずに表示する。"""
    if chunk_id not in p.chunks_by_id:
        return f"（再取込で削除された条文 #{chunk_id}）"
    return p.citation_label(chunk_id)


# --- 社員タブ ---

def set_question(q: str) -> None:
    st.session_state.question = q


def ask(question: str) -> Shown:
    p = get_pipeline()
    lock = get_lock()
    with st.status("確認しています…", expanded=True) as status:
        def on_step(step: str) -> None:
            status.update(label=STEP_LABELS[step])
            status.write(STEP_LABELS[step])

        if not lock.acquire(blocking=False):
            status.write("他の方のご質問を処理しています。順番にお待ちください…")
            lock.acquire()
        try:
            v = p.ask(question, on_step=on_step)
        except (httpx.HTTPError, ValueError):
            # Ollama 停止・タイムアウト・JSON 不正。回答は出さず、原因はサーバ側のログに残す
            log.exception("質問の処理に失敗: %s", question)
            status.update(label="処理に失敗しました", state="error", expanded=False)
            return Shown(question=question, error=True)
        finally:
            lock.release()
        status.update(label="確認しました", state="complete", expanded=False)

    if not v.answered:
        # 未整備判定では参考条文を出さない（要件 §3.6）
        return Shown(question=question)
    cards = [Card(title=citation_title(p, cid), body=p.chunks_by_id[cid].body,
                  pdf=p.source_pdf(cid), loc=p.chunks_by_id[cid].pdf_loc)
             for cid in v.citations]
    return Shown(question=question, answered=True, answer=v.answer, cards=cards)


def render_result(r: Shown) -> None:
    st.divider()
    st.markdown(f"**ご質問**　{plain(r.question)}")
    if r.error:
        st.error("回答システムに接続できませんでした。時間をおいて、もう一度お試しください。")
        return
    if not r.answered:
        st.info(plain(NOT_FOUND_MESSAGE))
        return
    st.subheader("回答")
    st.markdown(plain(r.answer))
    st.subheader("根拠条文")
    for i, card in enumerate(r.cards):
        with st.expander(card.title, expanded=i == 0):
            render_card(card)
    st.caption("回答は条文をもとに要約したものです。正式な内容は根拠条文の原本でご確認ください。")


def render_card(card: Card) -> None:
    if card.pdf is None or card.loc is None:
        st.markdown(plain(card.body))
        st.caption("原本PDFが未登録のため、取り込んだテキストを表示しています。" if card.pdf is None
                   else "原本PDFの中で該当箇所を特定できなかったため、取り込んだテキストを表示しています。")
        return
    tab_src, tab_text = st.tabs(["原本", "取り込んだテキスト"])
    with tab_src:
        mtime = card.pdf.stat().st_mtime
        for page_no, boxes in sorted(card.loc.pages.items()):
            st.image(page_image(str(card.pdf), mtime, page_no, tuple(boxes)),
                     caption=f"{card.pdf.name}　{page_no + 1} ページ目（PDF のページ番号）")
        if card.loc.highlighted:
            st.caption("色付きの部分が回答の根拠です。画像の右上のボタンで拡大できます。")
        else:
            st.caption("該当条文が始まるページです。本文の位置までは特定できなかったため、色付けしていません。")
    with tab_text:
        st.markdown(plain(card.body))


def employee_tab() -> None:
    st.caption("登録済みの社内規程の条文だけを根拠に回答します。規程に記載がない事項には回答しません。")

    examples = cfg["ui"]["examples"]
    st.markdown("**質問の例**")
    for col, ex in zip(st.columns(len(examples)), examples):
        col.button(ex, on_click=set_question, args=(ex,), width="stretch")

    with st.form("ask"):
        st.text_area("ご質問", key="question", height=90,
                     placeholder="例: 出張の日当はいくらですか")
        submitted = st.form_submit_button("確認する", type="primary")

    if submitted:
        q = st.session_state.question.strip()
        if not q:
            st.warning("ご質問を入力してください。")
        else:
            st.session_state.result = ask(q)

    if r := st.session_state.get("result"):
        render_result(r)


# --- 管理者タブ（要件 §3.7） ---

def admin_tab() -> None:
    p = get_pipeline()
    # 質問処理中でも待たされないよう、読み取りは別接続で行う
    conn = db.connect(p.db_path)
    try:
        counts = db.verdict_counts(conn)
        unanswered = db.unanswered_summary(conn)
        answered = db.answered_list(conn)
        documents = db.documents_summary(conn)
    finally:
        conn.close()

    total = sum(counts.values())
    rejected = total - counts.get("answered", 0)
    c1, c2, c3 = st.columns(3)
    c1.metric("質問数", total)
    c2.metric("回答", counts.get("answered", 0))
    c3.metric("棄却", rejected)
    if rejected:
        st.caption("棄却の内訳: " + " ／ ".join(
            f"{VERDICT_LABELS.get(k, k)} {n}件" for k, n in sorted(counts.items()) if k != "answered"))

    st.subheader("未整備質問一覧")
    st.caption("規程に該当する条文が見つからなかった質問。発生回数の多い順。")
    rows = [{"質問": r.question, "発生回数": r.count, "直近日時": r.last_asked,
             "判定": "、".join(VERDICT_LABELS.get(v, v) for v in r.verdicts.split(","))}
            for r in unanswered]
    if rows:
        st.dataframe(rows, hide_index=True, width="stretch")
        st.download_button("CSV をダウンロード", to_csv(rows), "未整備質問一覧.csv",
                           "text/csv", key="dl_unanswered")
    else:
        st.write("まだありません。")

    st.subheader("回答済み質問一覧")
    st.caption("回答の妥当性を総務ご担当者が確認するための一覧。新しい順。")
    rows = [{"日時": r.asked_at, "質問": r.question, "回答": r.answer,
             "引用条文": " ／ ".join(citation_title(p, c) for c in r.citations),
             "応答時間(秒)": round(r.latency_ms / 1000, 1)}
            for r in answered]
    if rows:
        st.dataframe(rows, hide_index=True, width="stretch")
        st.download_button("CSV をダウンロード", to_csv(rows), "回答済み質問一覧.csv",
                           "text/csv", key="dl_answered")
    else:
        st.write("まだありません。")

    st.subheader("取込状況")
    rows = [{"規程名": d.name, "条文": d.articles, "表": d.tables, "施行日": d.effective_date,
             "取込日時": d.ingested_at, "取り込めなかった画像": d.image_count,
             "元ファイル": d.source_file, "原本PDF": d.source_pdf or "未登録",
             "原本で位置を特定": f"{d.located}/{d.articles + d.tables}" if d.source_pdf else "-"}
            for d in documents]
    if rows:
        st.dataframe(rows, hide_index=True, width="stretch")
    else:
        st.write("登録済みの規程がありません。`uv run python -m minidx.ingest` で取り込んでください。")
    st.caption("規程を取り込み直したら、下のボタンで画面側の条文データを読み直してください。")
    if st.button("条文データを再読み込み"):
        get_pipeline.clear()
        st.rerun()


st.set_page_config(page_title=cfg["ui"]["title"], layout="centered")
st.title(cfg["ui"]["title"])
tab_employee, tab_admin = st.tabs(["質問する", "管理者"])
with tab_employee:
    employee_tab()
with tab_admin:
    admin_tab()
