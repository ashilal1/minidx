#!/bin/bash
# Stop フック: 作業記録の付け忘れを検知して差し戻す。
#
# 判定: 「当日ログより新しく更新されたファイル」が存在するか。
#   → 存在する = コードを触ったのにログに書いていない状態。
# ループ防止: prompt_id ごとにマーカーを置き、1ユーザーターンにつき最大1回しか差し戻さない
#   （公式推奨の prompt_id によるリトライ追跡）。加えて stop_hook_active が true なら即通す。
set -u

input=$(cat)
jq_get() { printf '%s' "$input" | jq -r "$1" 2>/dev/null; }

[ "$(jq_get '.stop_hook_active // false')" = "true" ] && exit 0

project="${CLAUDE_PROJECT_DIR:-$(jq_get '.cwd // "."')}"
[ -d "$project/docs/log" ] || exit 0

prompt_id=$(jq_get '.prompt_id // empty')
[ -n "$prompt_id" ] || exit 0
marker_dir="${TMPDIR:-/tmp}/minidx-log-hook"
mkdir -p "$marker_dir" 2>/dev/null || exit 0
find "$marker_dir" -type f -mtime +1 -delete 2>/dev/null
marker="$marker_dir/${prompt_id//\//_}"
[ -e "$marker" ] && exit 0

today=$(date +%Y-%m-%d)
log="$project/docs/log/$today.md"

find_args=(
  "$project" -type f
  -not -path "*/.git/*" -not -path "*/docs/log/*" -not -path "*/data/*"
  -not -path "*/.venv/*" -not -path "*/__pycache__/*"
  -not -path "*/.pytest_cache/*" -not -path "*/.ruff_cache/*"
  -not -name ".DS_Store" -not -name "*.db"
)

if [ -f "$log" ]; then
  # 当日ログより後に更新されたファイル
  pending=$(find "${find_args[@]}" -newer "$log" 2>/dev/null | head -5)
else
  # 当日ログが無い場合は、今日更新されたファイルがあれば未記録とみなす
  pending=$(find "${find_args[@]}" -newermt "$today 00:00:00" 2>/dev/null | head -5)
fi

[ -z "$pending" ] && exit 0

: > "$marker"   # このターンでは二度と差し戻さない

{
  echo "作業記録が未記入です。ターンを終える前に $log へ追記してください（書式: docs/log/README.md）。"
  echo "当日ログより新しく更新されたファイル:"
  printf '%s\n' "$pending" | sed "s|^$project/|  - |"
  echo "追記したら、そのまま応答を終えて構いません（このフックは同じターンで再度止めません）。"
} >&2
exit 2
