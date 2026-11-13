#!/usr/bin/env bash
# Builds paper/main.pdf: claim numbers -> figures -> LaTeX.
#
#   paper/build.sh [--session NAME] [--final]
#
# --session picks the session whose raw data draws the figures (orig = original M4-M6 runs, or s0, s1, ...);
# --final hides the TODO / VERIFY / claim-ID annotations.
set -euo pipefail
cd "$(dirname "$0")/.."
session=orig
final=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --session) session="$2"; shift 2 ;;
    --final) final=1; shift ;;
    *) echo "unknown argument $1" >&2; exit 2 ;;
  esac
done

.venv/bin/python benchmarks/analysis/confirm_claims.py >/dev/null
.venv/bin/python paper/make_figures.py --session "$session"
cd paper
if [[ $final -eq 1 ]]; then
  sed 's/^\\newif\\ifdraft\\drafttrue/\\newif\\ifdraft\\draftfalse/' main.tex > main-final.tex
  tectonic --keep-logs main-final.tex
else
  tectonic --keep-logs main.tex
fi
echo "built paper/$([[ $final -eq 1 ]] && echo main-final || echo main).pdf"
echo "open TODO/VERIFY markers in main.tex: $(grep -o '\\todo{\|\\verify{' main.tex | wc -l | tr -d ' ')"
