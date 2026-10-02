# minidx — 社内規程AI確認システム

株式会社大和田測量設計「ミニDX化支援業務」の MVP（会津大学チーム）。
社内規程を取り込み、社員の自然文の質問に対して**登録済みの条文だけを根拠に**回答する。

> **規程に書かれていない質問に、一度も「それらしい回答」を出さないこと。**
>
> 誤回答率（回答不能な質問に回答してしまう率）は 0% が条件。過剰棄却は 20% まで許容する。
> 回答範囲を広げることと棄却を確実にすることが衝突したら、常に棄却側を採る。

設計の正本は [docs/MVP要件定義.md](docs/MVP要件定義.md)、開発規約は [AGENTS.md](AGENTS.md)。

## 現在の状態

| 機能 | 状態 |
|---|---|
| 取込（`.docx` → 条文・表チャンク → 埋め込み → SQLite） | 実装済み |
| 原本PDFの該当箇所の表示（同名 PDF のページを描画して色付け） | 実装済み（実物の PDF では未検証） |
| ハイブリッド検索（bge-m3 + BM25、RRF 融合） | 実装済み |
| 二段棄却ゲート / 回答生成 / 質問ログ | 実装済み |
| CLI での質疑応答（`minidx.ask`） | 実装済み |
| Streamlit 画面（`app.py`：社員タブ・管理者タブ） | 実装済み |
| ログイン（パスワードゲート） | **未実装**（公開前に必須。要件 §7.3） |
| 評価スクリプト（`minidx.eval`） | **未実装** |
| テスト（`tests/`） | **未実装** |

- 誤回答率: **未計測**（評価スクリプトと回答可能質問のセットがまだ無いため）
- `gate.tau_cos = 0.60` は暫定値。校正と測定に同じ質問セットを使っているため、測定値とは言えない
- 応答時間の実測値と経緯は [docs/log/](docs/log/) を参照

## 仕組み

```
[取込]  regulations/*.docx → 条文・表に分割 → data/markdown/（目視修正用）→ bge-m3 で埋め込み → data/minidx.db

[質問]  質問文
          ↓ 埋め込み + BM25
        ハイブリッド検索（上位 k=5）
          ↓
        ゲート1（LLM 呼び出し前）── 不合格 → 固定文言で棄却（LLM は呼ばない）
          ↓ 合格
        qwen2.5:7b が JSON Schema 付きで生成
          ↓
        ゲート2（生成後の機械検証）── 不合格 → 棄却
          ↓ 合格
        回答 + 根拠条文          ※全経路を query_log に記録
```

**ゲート1**（[src/minidx/gate.py](src/minidx/gate.py)）は LLM を呼ぶ前に判定する。ここで棄却すれば幻覚は起こりえない。
- 質問の内容語が規程の語彙に1語も無ければ棄却（`rejected_no_vocab`）
- 最上位のコサイン類似度が `tau_cos` 未満なら棄却（`rejected_low_score`）
- RRF の融合スコアは順位ベースで絶対的な基準を持たないため、並べ替えにだけ使い、閾値判定には使わない

**ゲート2** は LLM の自己申告（`sufficient`）を最終判定に使わない。
- `sufficient=false` は棄却に使う。`true` は合格の保証にならない（`rejected_insufficient`）
- 引用 ID は JSON Schema の `enum` で検索結果の番号に制限し、さらに実装側でも範囲を確認する（`rejected_bad_citation`）
- 回答に出てくる金額が、引用した条文の中に実在しなければ棄却する（`rejected_bad_citation`）

**既知の限界**
- 同じ表の中で行を取り違えた場合（別の職級の金額を答えるなど）は検出できない
- 金額を含まない回答は、ゲート1・`sufficient`・引用番号の範囲チェックしか通らない

## セットアップ

前提: [uv](https://docs.astral.sh/uv/) と [Ollama](https://ollama.com/)。Python は uv が 3.12 を用意する。

```bash
ollama pull qwen2.5:7b
ollama pull bge-m3
uv sync
```

規程ファイルを `regulations/` に置く。このディレクトリは顧客の機密情報なので **git 管理外**。

- `07_旅費規程.docx` … 取込の正本。条文はここから取り込む
- `07_旅費規程.pdf` … 同じ Word から「PDFとして保存」したもの（任意）。画面で原本の該当ページを見せるためだけに使う。
  Word と**同じ版**であること。版が違うと該当箇所を特定できず、原本は表示されない

## 使い方

```bash
uv run python -m minidx.ingest              # regulations/*.docx を取り込む（同名の規程は置き換え）
uv run streamlit run app.py                 # 画面を起動 → http://localhost:8501
uv run python -m minidx.ask "出張の日当はいくらですか"   # CLI（ゲートの挙動確認用）
```

### 画面

- **質問する**: 質問を入力して「確認する」を押す。待っている間は処理の段階（検索 → 回答作成 → 根拠の検証）が表示される。
  回答できたときは回答文と根拠条文を表示する。根拠条文は**原本 PDF の該当ページ**を描画し、根拠の部分に色を付ける（「取り込んだテキスト」タブで抽出結果も確認できる）。
  回答できないときは固定文言だけを表示する。
  質問例のボタンは `config.yaml` の `ui.examples` で変える。質問は1問ずつ独立して判定され、前の質問の内容は引き継がない。
- **管理者**: 質問数と棄却の内訳、未整備質問一覧（発生回数順）、回答済み質問一覧、取込状況。一覧は CSV でダウンロードできる。
  規程を取り込み直したら「条文データを再読み込み」を押す。

**ログインは未実装。** `.streamlit/config.toml` で localhost だけで待ち受ける設定にしている。
トンネルで公開する前に、要件 §7.3 のパスワードゲート（社員用と管理者用で分ける）を実装すること。

取込時には、条番号の欠番、位置表示の重複、取り込めなかった画像の枚数が警告として出る。
分割が崩れていたら `data/markdown/<規程名>.md` を確認する。

PDF があれば、取込時に各条文の PDF 上の位置を特定し、件数を表示する（例: `位置を特定 28/28`）。
特定できなかった条文は、画面で原本を出さずに取り込んだテキストだけを示す。違うページを原本として見せないためである。

同じ名前の Word がない PDF、`.xlsx`、スキャンした PDF は MVP では対象外にしている（取込時に理由を表示してスキップする）。

## ディレクトリ

```
src/minidx/
  docx_ingest.py   Word → チャンク（条文の境界判定、表を「列見出し: 値」に展開、項単位の再分割）
  ingest.py        取込 CLI
  retrieve.py      ハイブリッド検索・語彙カバレッジの計算
  gate.py          二段棄却ゲート（generate.py を import しない）
  generate.py      プロンプトと JSON Schema
  pipeline.py      1問ぶんの処理本体
  ask.py           質疑応答 CLI
  pdf_locate.py    原本PDF上の位置の特定と、該当ページの描画（色付け）
  db.py            SQLite（documents / chunks / query_log）
  ollama.py        localhost の Ollama クライアント
  config.py        config.yaml の読み込み
app.py             Streamlit 画面（社員タブ・管理者タブ）
.streamlit/        画面の設定（localhost のみで待ち受け）
config.yaml        閾値・k・モデル名（コードに直書きしない）
eval/              評価セット（回答不能テーマ）。運用ルールは eval/README.md
docs/              要件定義・客先受領資料・作業ログ
regulations/       規程ファイル（git 管理外）
data/              中間 Markdown・SQLite（git 管理外）
```

## 設定

すべて [config.yaml](config.yaml) で変える。主な項目:

| キー | 意味 |
|---|---|
| `ollama.llm` / `ollama.embed` | 生成モデル / 埋め込みモデル |
| `ollama.num_ctx` | コンテキスト長。明示しないとプロンプトが黙って切り捨てられる |
| `retrieve.top_k` | LLM に渡す条文の数 |
| `gate.tau_cos` | ゲート1のコサイン類似度の閾値 |
| `gate.max_oov_ratio` | 規程に無い語の割合の上限 |

検索・ゲート・生成を変更したら、評価を回して誤回答率 0% を確認してから完了とする（評価スクリプトは未実装）。

## 機密情報の扱い

- `regulations/` と `data/` はコミットしない
- 推論は localhost の Ollama で完結させ、外部の AI API は呼ばない
- 規程本文と質問ログを外部サービスに送らない
- 詳細は [AGENTS.md](AGENTS.md) §6

## 開発ルール

- ブランチは `feat/MVP`。コミットメッセージは日本語で、`feat:` / `fix:` / `docs:` を付ける
- 作業記録は `docs/log/YYYY-MM-DD.md` に追記する（書式は [docs/log/README.md](docs/log/README.md)）
- 客先受領資料（`docs/社内規程AI確認システムの開発仕様.md`、`docs/2026-09-03_キックオフMTG議事録.md`）は編集しない
