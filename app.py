"""Streamlit 社員画面（質問する）。管理画面は admin.py（2026-11-09 論点7で別アプリに分離）。

起動: uv run streamlit run app.py
  社内 LAN で使うときは --server.address 0.0.0.0 を付ける（既定は .streamlit/config.toml の localhost）。

★認証は未実装。2026-10-02 のユーザー判断で、ローカル動作の間はログインなしで画面を出す。
  トンネルで公開する前に docs/MVP要件定義.md §7.3 のパスワードゲート（社員用／管理者用を分ける）を
  必ず入れること。それまでは .streamlit/config.toml で localhost のみにバインドしている。
"""
from __future__ import annotations
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import streamlit as st

from minidx.models import Box, PdfLocation
from minidx.pdf_locate import render_page
from minidx.pipeline import NOT_FOUND_MESSAGE, Pipeline
from minidx.ui import cfg, citation_title, corpus_signature, plain

log = logging.getLogger(__name__)

# 待ち表示の文言。Pipeline.ask の on_step が通知する段階に対応する
STEP_LABELS = {
    "search": "条文を検索しています…",
    "generate": "条文をもとに回答を作成しています…",
    "verify": "回答の根拠を検証しています…",
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
    synonyms_applied: list[str] = field(default_factory=list)


@st.cache_resource(max_entries=1)
def _pipeline(signature: str) -> Pipeline:
    """条文・埋め込み・BM25 索引はプロセスで1つだけ持つ。

    signature（登録済み規程の件数と最新の取込日時）が変わったら作り直す。管理画面を別アプリに分けたため、
    再取込を管理画面のボタンで知らせる代わりに、社員画面が DB の変化を見て自動で読み直す（論点7）。
    """
    p = Pipeline()
    try:
        p.client.warmup()   # 論点4: 社員の最初の質問で読み込みを待たせない
    except Exception as e:  # 読み込めなくても画面は出す。質問時に生成が失敗すれば棄却として扱われる
        st.warning(f"AI モデルの事前読み込みに失敗しました（{type(e).__name__}）。Ollama が起動しているか確認してください。")
    return p


def get_pipeline() -> Pipeline:
    return _pipeline(corpus_signature())


@st.cache_resource
def get_lock() -> threading.Lock:
    """質問の処理を直列化する。8GB 機では LLM の同時実行でメモリが溢れ、SQLite 接続も共有しているため。"""
    return threading.Lock()


@st.cache_data(max_entries=64)
def page_image(pdf: str, mtime: float, page_no: int, boxes: tuple[Box, ...]) -> bytes:
    """原本の1ページを描画する。mtime をキーに含め、PDF を差し替えたら描き直す。"""
    return render_page(Path(pdf), page_no, list(boxes), cfg["source_view"]["render_scale"])


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
        return Shown(question=question, synonyms_applied=v.synonyms_applied)
    cards = [Card(title=citation_title(p, cid), body=p.display_body(cid),   # 論点②: 項に分けた条は条全体
                  pdf=p.source_pdf(cid), loc=p.chunks_by_id[cid].pdf_loc)
             for cid in v.citations]
    return Shown(question=question, answered=True, answer=v.answer, cards=cards,
                 synonyms_applied=v.synonyms_applied)


def render_result(r: Shown) -> None:
    st.divider()
    st.markdown(f"**ご質問**　{plain(r.question)}")
    if r.synonyms_applied:
        # 論点6: 言い換え対応表で置き換えたことを必ず見せる
        st.caption("言い換え対応表により、" + "、".join(
            f"「{a.split('→')[0]}」を「{a.split('→')[1]}」" for a in r.synonyms_applied) + "として検索しました。")
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


st.set_page_config(page_title=cfg["ui"]["title"], layout="centered")
st.title(cfg["ui"]["title"])
employee_tab()
