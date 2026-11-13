#!/usr/bin/env bash
# Runs every linter. Exits non-zero if a *blocking* check fails.
#
#   tools/lint.sh            report all checks; block on the ones listed in BLOCKING
#   tools/lint.sh --strict   block on every check
#
# Checks: format (clang-format), tidy (clang-tidy, needs build/release/
# compile_commands.json), ruff (Python), shellcheck (shell). Checks are added to
# BLOCKING as the refactor brings each one to zero findings
# (docs/refactoring_verification.md).
set -euo pipefail
cd "$(dirname "$0")/.."
BLOCKING="format"
[[ "${1:-}" == "--strict" ]] && BLOCKING="format tidy ruff shellcheck"

cxx_files() { git ls-files '*.h' '*.cpp' '*.mm'; }
py_files() { git ls-files '*.py'; }
sh_files() { git ls-files '*.sh'; }
tidy="$(command -v clang-tidy || true)"
[[ -z "$tidy" && -x "$(brew --prefix llvm 2>/dev/null)/bin/clang-tidy" ]] && tidy="$(brew --prefix llvm)/bin/clang-tidy"
py=".venv/bin"

declare -a failed=()
check() {  # name, count
  local name="$1" count="$2"
  printf '%-11s %s finding(s)\n' "$name" "$count"
  if [[ "$count" != "0" && " $BLOCKING " == *" $name "* ]]; then failed+=("$name"); fi
}

n=$(cxx_files | xargs clang-format --dry-run 2>&1 | grep -c 'warning:' || true)
check format "$n"

if [[ -n "$tidy" && -f build/release/compile_commands.json ]]; then
  n=$(git ls-files '*.cpp' '*.mm' | xargs "$tidy" -p build/release --quiet 2>/dev/null \
      | grep -cE 'warning:|error:' || true)
  check tidy "$n"
else
  echo "tidy        skipped (needs clang-tidy and build/release/compile_commands.json)"
fi

n=$(py_files | xargs "$py/ruff" check --quiet --output-format concise 2>/dev/null | grep -cE '^[^ ].*:[0-9]+:[0-9]+:' || true)
check ruff "$n"

n=$(sh_files | xargs shellcheck --format=gcc 2>/dev/null | grep -cE ':[0-9]+:[0-9]+:' || true)
check shellcheck "$n"

if [[ ${#failed[@]} -gt 0 ]]; then echo "blocking checks failed: ${failed[*]}"; exit 1; fi
echo "lint: blocking checks pass (${BLOCKING})"
