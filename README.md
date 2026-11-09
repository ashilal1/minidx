# minidx — 社内規程AI確認システム

株式会社大和田測量設計「ミニDX化支援業務」の開発リポジトリ（会津大学チーム）。
社内規程を取り込み、社員の自然文の質問に対して**登録済みの条文だけを根拠に**短く回答する。

> **規程に書かれていない質問に、一度も「それらしい回答」を出さないこと。**
>
> 誤回答率（回答不能な質問に回答してしまう率）は 0% が条件。過剰棄却は 20% まで許容する。
> 回答範囲を広げることと棄却を確実にすることが衝突したら、常に棄却側を採る。

設計の正本は [docs/MVP要件定義.md](docs/MVP要件定義.md)（2026-11 の改訂は §10）、開発規約は [AGENTS.md](AGENTS.md)。
改訂の経緯と変更点の一覧は [docs/改訂設計_2026-11.md](docs/改訂設計_2026-11.md)。

## 現在の状態（2026-11-09）

| 機能 | 状態 |
|---|---|
| 取込（`.docx` → 条文・表チャンク → 埋め込み → SQLite。毎回全件作り直し） | 実装済み |
| 原本PDFの該当箇所の表示（同名の PDF に色付け） | 実装済み。旅費規程で 28/28 か所を確認 |
| ハイブリッド検索（bge-m3 + BM25、RRF 融合） | 実装済み |
| 二段棄却ゲート（数値照合・表の行照合を含む） / 1文回答の生成 / 質問ログ | 実装済み |
| 言い換え対応表（管理画面で編集） | 実装済み |
| 社員画面 `app.py` / 管理画面 `admin.py`（別アプリ） | 実装済み |
| 評価スクリプト `minidx.eval` | 実装済み。回答可能質問は2問のみ（人が追加中） |
| テスト `tests/` | 42件 |
| ログイン（パスワードゲート） | 未実装。社内 LAN 利用では作らない合意（10/5）。インターネット公開時は必須 |

- 誤回答率: calib で 0/48（測定値ではない。holdout は未実行）。詳しくは [docs/改訂設計_2026-11.md](docs/改訂設計_2026-11.md) §7
- LLM は `qwen2.5:3b` を仮採用。最終決定は回答可能質問がそろってから評価で行う

## 仕組み

```
[取込]  regulations/*.docx（＋同名 PDF）
          → 検査（同じ規程が2つあれば止める）→ 条・項・表の行に分割
          → 規程名・条番号を付けて bge-m3 で埋め込み → PDF 上の位置を特定
          → 1トランザクションで DB の規程を全件入れ替え

[質問]  質問文 → 言い換え対応表で規程の言葉に置き換え
          → ハイブリッド検索（上位 k=5）
          → ゲート1（LLM 呼び出し前）── 不合格 → 固定文言で棄却（LLM は呼ばない）
          → 項に分けた条は条全体に広げて LLM へ
          → qwen2.5 が JSON Schema 付きで1文の回答を生成（時間切れは棄却）
          → ゲート2（生成後の機械検証）── 不合格 → 棄却
          → 回答 + 根拠条文（原本 PDF の色付け）   ※全経路を query_log に記録
```

**ゲート1**（[src/minidx/gate.py](src/minidx/gate.py)）は LLM を呼ぶ前に判定する。ここで棄却すれば幻覚は起こりえない。
- 質問の内容語が規程の語彙に1語も無ければ棄却（`rejected_no_vocab`）
- 最上位のコサイン類似度が `tau_cos` 未満なら棄却（`rejected_low_score`）

**ゲート2** は LLM の自己申告（`sufficient`）を最終判定に使わない。
- `sufficient=false` は棄却に使う。`true` は合格の保証にならない（`rejected_insufficient`）
- 引用 ID は JSON Schema の `enum` で検索結果の番号に制限し、実装側でも範囲を確認する（`rejected_bad_citation`）
- 回答中の単位付きの数（円・日・か月・年・回・時間 など）が引用条文に無ければ棄却（`rejected_bad_number`）
- 表を引用した場合、回答の各区切りの中の区分名と数が表の同じ行に無ければ棄却（`rejected_table_row`）
- 生成の時間切れ・出力の破損は棄却（`rejected_generation_error`）

**既知の限界**
- 数を含まない回答（「支給しない」「承認が必要」など）の内容は検証できない
- 小型モデルが表の回答を崩れた形（区分名と金額の対応が読み取れない形）で出すと、行照合で棄却される（安全側）
- 言い換え対応表に意味の違う語どうしを登録すると、本来棄却すべき質問がゲート1を通りうる（登録は担当者の責任）
- 画像の中の文字は取り込めない（取込時に枚数を警告する）

## セットアップ

前提: [uv](https://docs.astral.sh/uv/) と [Ollama](https://ollama.com/)。Python は uv が 3.12 を用意する。

```bash
ollama pull qwen2.5:3b
ollama pull bge-m3
uv sync
```

規程ファイルを `regulations/` に置く。このディレクトリは顧客の機密情報なので **git 管理外**。
**このフォルダの中身が現行版**として扱われる。旧版は別の場所へ移すこと（同じ規程が2つあると取込が止まる）。

- `07_旅費規程（2025.11.19改訂）.docx` … 取込の正本。ファイル名の括弧内は改訂表記として保存される
- `07_旅費規程（2025.11.19改訂）.pdf` または `….docx.pdf` … 同じ Word から PDF として保存したもの（任意）。原本表示にだけ使う

## 使い方

```bash
uv run python -m minidx.ingest                       # regulations/ を取り込む（DB の規程を全件作り直す）
uv run streamlit run app.py                          # 社員画面 → http://localhost:8501
uv run streamlit run admin.py --server.port 8502 --server.address 127.0.0.1   # 管理画面（設置 PC の上でだけ）
uv run python -m minidx.ask "出張の日当はいくらですか"   # CLI（ゲートの挙動確認用）
uv run python -m minidx.eval                         # 評価（calib）。--llm でモデルを差し替えて比較
uv run pytest                                        # テスト
```

社内 LAN で社員画面を使うときは `--server.address 0.0.0.0` を付ける（既定は localhost のみ）。
管理画面はループバック以外で起動すると何も表示しない。

### 社員画面（app.py）

質問を入力して「確認する」を押す。回答できたときは1文の回答と根拠条文（原本 PDF の該当ページに色付け）を表示する。
回答できないときは固定文言だけを表示し、参考条文も出さない。言い換え対応表で置き換えた場合は、その旨を表示する。
質問は1問ずつ独立して判定され、前の質問の内容は引き継がない。

### 管理画面（admin.py）

質問数と棄却の内訳、未整備質問一覧（発生回数順）、回答済み質問一覧、言い換え対応表の編集、取込状況。一覧は CSV でダウンロードできる。
規程を取り込み直すと、社員画面は次の質問のときに自動で新しい条文を読み直す。

## ディレクトリ

```
src/minidx/
  docx_ingest.py   Word → チャンク（条文の境界判定、表を「列見出し: 値」に展開、項単位の再分割）
  ingest.py        取込 CLI（検査 → 埋め込み → 全件入れ替え）
  retrieve.py      ハイブリッド検索・語彙カバレッジの計算
  gate.py          二段棄却ゲート（generate.py を import しない）
  generate.py      JSON Schema と、プロンプトへの条文の差し込み
  pipeline.py      1問ぶんの処理本体（言い換え、条全体への拡張を含む）
  synonyms.py      言い換え対応表
  eval.py          評価 CLI
  ask.py           質疑応答 CLI
  pdf_locate.py    原本PDF上の位置の特定と、該当ページの描画（色付け）
  db.py            SQLite（documents / chunks / query_log / synonyms）
  ollama.py        localhost の Ollama クライアント
  ui.py            社員画面と管理画面の共通部品
  config.py        config.yaml の読み込み
app.py             社員画面
admin.py           管理画面
config.yaml        閾値・k・モデル名・プロンプト（コードに直書きしない）
eval/              評価セット（回答不能テーマ・回答可能質問）。運用ルールは eval/README.md
tests/             pytest（データはすべて架空）
docs/              要件定義・客先受領資料・調査・作業ログ
regulations/       規程ファイル（git 管理外）
data/              中間 Markdown・SQLite（git 管理外）
```

## 設定

すべて [config.yaml](config.yaml) で変える。主な項目:

| キー | 意味 |
|---|---|
| `ollama.llm` / `ollama.embed` | 生成モデル / 埋め込みモデル |
| `ollama.keep_alive` / `ollama.timeout_sec` / `ollama.num_predict` | モデルの保持時間 / 時間切れ（棄却）/ 生成の上限 |
| `retrieve.top_k` | LLM に渡す条文の数 |
| `gate.tau_cos` / `gate.max_oov_ratio` | ゲート1の閾値 |
| `gate.number_units` / `gate.table_label_min_chars` | ゲート2の数値照合の単位 / 表の行ラベルの最短文字数 |
| `prompt.system` | 生成のプロンプト |

検索・ゲート・生成を変更したら `uv run python -m minidx.eval` を回し、誤回答率 0% を確認してから完了とする。

## 機密情報の扱い

- `regulations/`・`data/`・`eval/results/` はコミットしない
- 推論は localhost の Ollama で完結させ、外部の AI API は呼ばない
- 規程本文と質問ログを外部サービスに送らない。開発支援の AI ツールに規程本文を読ませない
- コード・テスト・設定・文書に規程の金額や本文を書かない（テストは架空のデータ）
- 詳細は [AGENTS.md](AGENTS.md) §6

## 開発ルール

- 作業ブランチは `feat/revision`（develop から分岐）。コミットメッセージは日本語で、`feat:` / `fix:` / `docs:` を付ける
- 作業記録は `docs/log/YYYY-MM-DD.md` に追記する（書式は [docs/log/README.md](docs/log/README.md)）
- 客先受領資料（`docs/社内規程AI確認システムの開発仕様.md`、`docs/2026-09-03_キックオフMTG議事録.md`）は編集しない
