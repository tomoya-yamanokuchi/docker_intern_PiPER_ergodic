#!/usr/bin/env bash
# PostToolUse gate. Formats the edited file, then re-checks it.
# Exit 2 hands the message back to Claude so it fixes the problem in the same
# turn instead of you finding it later. Exit 0 means the file is clean.
set -uo pipefail

f=$(jq -r '.tool_input.file_path // empty' 2>/dev/null)
[ -z "$f" ] && exit 0
[ -f "$f" ] || exit 0

case "$f" in
  *.py)
    command -v ruff >/dev/null || exit 0
    ruff format "$f" >/dev/null 2>&1
    if ! out=$(ruff check "$f" 2>&1); then
      printf 'ruff found problems in %s. Fix them now; do not suppress with noqa.\n%s\n' "$f" "$out" >&2
      exit 2
    fi
    ;;
  *.cpp|*.cc|*.cxx|*.hpp|*.hh|*.h)
    command -v clang-format >/dev/null && clang-format -i "$f" >/dev/null 2>&1
    if command -v clang-tidy >/dev/null && [ -f compile_commands.json ]; then
      if ! out=$(clang-tidy "$f" 2>/dev/null); then
        printf 'clang-tidy found problems in %s. Fix them now.\n%s\n' "$f" "$out" >&2
        exit 2
      fi
    fi
    ;;
esac
exit 0
