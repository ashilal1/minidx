@AGENTS.md

## Claude Code 固有の運用

- プロジェクト指示の正本は `AGENTS.md`（上の import で読み込まれる）。他の AI CLI と共有するため、**指示の追記は `AGENTS.md` 側に書く**。このファイルには Claude Code 固有の設定だけを置く。
- ファイル種別ごとの詳細規約は `.claude/rules/` に分割してある（該当ファイルを開いたときだけ読み込まれる）。
  - `.claude/rules/python.md` — Python コーディング規約（`**/*.py`）
  - `.claude/rules/gate.md` — 棄却ゲート・検索・生成の実装規律（`src/minidx/{retrieve,gate,generate}.py`, `eval/**`）
  - `.claude/rules/streamlit.md` — 画面と認証（`app.py`, `app/**`）
- 権限は `.claude/settings.json` にコミット済み。`curl` / `wget` / `cloudflared` / `git push` は毎回確認を挟む（規程テキストの外部送信と意図しない公開を防ぐため）。個人用の例外は `.claude/settings.local.json` に書く。
- 実装は未着手の部分が多い。`AGENTS.md` §3 のディレクトリ構成と §4 のコマンドは**これから作る際の取り決め**であり、既存の実測ではない。新規作成時はこの構成に合わせる。
