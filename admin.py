"""Streamlit 管理画面（要件 §3.7）。2026-11-09 論点7で社員画面から分離した。

起動: uv run streamlit run admin.py --server.port 8502 --server.address 127.0.0.1

設置 PC の上（127.0.0.1）でだけ開く。質問ログには聞きづらい質問も残るため（10/5 第2回MTG §3.4）、
社員が使う LAN の URL からは開けないようにする。ループバック以外で起動された場合は何も表示しない
（未認証時は UI もデータも描画しない、AGENTS.md §6 と同じ考え方）。
"""
from __future__ import annotations

import streamlit as st

from minidx import db
from minidx.pipeline import Pipeline
from minidx.synonyms import Synonym, validate
from minidx.ui import DB_PATH, VERDICT_LABELS, cfg, citation_title, corpus_signature, is_local_only, to_csv

st.set_page_config(page_title=f"{cfg['ui']['title']}（管理）", layout="wide")

if not is_local_only():
    st.error("管理画面は設置 PC の上でだけ開けます。--server.address 127.0.0.1 を付けて起動してください。")
    st.stop()


@st.cache_resource(max_entries=1)
def _pipeline(signature: str) -> Pipeline:
    """引用の表示名（規程名・条番号）を引くためだけに使う。LLM は呼ばないので事前読み込みはしない。"""
    return Pipeline()


def section_summary(p: Pipeline, conn) -> None:
    counts = db.verdict_counts(conn)
    total = sum(counts.values())
    rejected = total - counts.get("answered", 0)
    c1, c2, c3 = st.columns(3)
    c1.metric("質問数", total)
    c2.metric("回答", counts.get("answered", 0))
    c3.metric("棄却", rejected)
    if rejected:
        st.caption("棄却の内訳: " + " ／ ".join(
            f"{VERDICT_LABELS.get(k, k)} {n}件" for k, n in sorted(counts.items()) if k != "answered"))


def section_unanswered(conn) -> None:
    st.subheader("未整備質問一覧")
    st.caption("規程に該当する条文が見つからなかった質問。発生回数の多い順。"
               "規程に追記すべき事項か、言い換えの問題（下の対応表で解決できる）かを判断してください。")
    rows = [{"質問": r.question, "発生回数": r.count, "直近日時": r.last_asked,
             "判定": "、".join(VERDICT_LABELS.get(v, v) for v in r.verdicts.split(","))}
            for r in db.unanswered_summary(conn)]
    if rows:
        st.dataframe(rows, hide_index=True, width="stretch")
        st.download_button("CSV をダウンロード", to_csv(rows), "未整備質問一覧.csv",
                           "text/csv", key="dl_unanswered")
    else:
        st.write("まだありません。")


def section_answered(p: Pipeline, conn) -> None:
    st.subheader("回答済み質問一覧")
    st.caption("回答の妥当性を総務ご担当者が確認するための一覧。新しい順。")
    rows = [{"日時": r.asked_at, "質問": r.question, "回答": r.answer,
             "引用条文": " ／ ".join(citation_title(p, c) for c in r.citations),
             "応答時間(秒)": round(r.latency_ms / 1000, 1)}
            for r in db.answered_list(conn)]
    if rows:
        st.dataframe(rows, hide_index=True, width="stretch")
        st.download_button("CSV をダウンロード", to_csv(rows), "回答済み質問一覧.csv",
                           "text/csv", key="dl_answered")
    else:
        st.write("まだありません。")


def section_synonyms(conn) -> None:
    """論点6: 言い換え対応表。表の内容を保存すると、次の質問から社員画面に反映される。"""
    st.subheader("言い換え対応表")
    st.caption("社員が使う言葉を、規程の言葉に置き換えてから検索します（例: 泊まり → 宿泊）。"
               "置き換えたことは社員の画面と質問ログに表示されます。")
    st.warning("対応表は「この言葉はこの意味とみなす」という判断です。意味の違う言葉どうし（例: 在宅勤務 → 出張）を"
               "登録すると、本来は答えるべきでない質問に回答するおそれがあります。登録は担当者だけが行ってください。")
    current = [{"社員が使う言葉": s.term, "規程の言葉": s.replacement, "メモ": s.note}
               for s in db.load_synonyms(conn)]
    edited = st.data_editor(current or [{"社員が使う言葉": "", "規程の言葉": "", "メモ": ""}],
                            num_rows="dynamic", hide_index=True, width="stretch", key="synonyms_editor")
    if st.button("対応表を保存", type="primary"):
        rows = [Synonym(term=(r.get("社員が使う言葉") or "").strip(),
                        replacement=(r.get("規程の言葉") or "").strip(),
                        note=(r.get("メモ") or "").strip())
                for r in edited]
        rows = [r for r in rows if r.term or r.replacement]   # 完全に空の行は無視する
        problems = validate(rows)
        if problems:
            for msg in problems:
                st.error(msg)
        else:
            db.save_synonyms(conn, rows)
            st.success(f"{len(rows)}件を保存しました。次の質問から反映されます。")


def section_documents(conn) -> None:
    st.subheader("取込状況")
    rows = [{"規程名": d.name, "条文": d.articles, "表": d.tables, "施行日": d.effective_date,
             "取込日時": d.ingested_at, "取り込めなかった画像": d.image_count,
             "元ファイル": d.source_file, "原本PDF": d.source_pdf or "未登録",
             "原本で位置を特定": f"{d.located}/{d.articles + d.tables}" if d.source_pdf else "-"}
            for d in db.documents_summary(conn)]
    if rows:
        st.dataframe(rows, hide_index=True, width="stretch")
    else:
        st.write("登録済みの規程がありません。`uv run python -m minidx.ingest` で取り込んでください。")
    st.caption("規程を取り込み直すと、社員画面は次の質問のときに自動で新しい条文を読み直します。")


st.title(f"{cfg['ui']['title']}（管理）")
pipeline = _pipeline(corpus_signature())
connection = db.connect(DB_PATH)   # 社員画面とは別プロセス。読み書きはこの接続で行う
try:
    section_summary(pipeline, connection)
    section_unanswered(connection)
    section_answered(pipeline, connection)
    section_synonyms(connection)
    section_documents(connection)
finally:
    connection.close()
