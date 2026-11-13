#!/bin/sh
# Raw-data integrity for benchmarks/results/.
#
#   results_manifest.sh check    verify every file in MANIFEST.sha256 is unchanged (CI / ctest)
#   results_manifest.sh add      append new result files to the manifest (existing entries are never rewritten)
#
# Policy (see benchmarks/results/README.md): committed result files are immutable.
# A wrong run is superseded by a new run, never edited or deleted.
set -eu
cd "$(dirname "$0")/../benchmarks/results"
MANIFEST=MANIFEST.sha256

case "${1:-check}" in
  check)
    [ -f "$MANIFEST" ] || { echo "no $MANIFEST"; exit 1; }
    shasum -a 256 -c --quiet "$MANIFEST"
    echo "results: $(wc -l < "$MANIFEST" | tr -d ' ') files verified"
    ;;
  add)
    touch "$MANIFEST"
    # Every regular file except the manifest and this README, in stable order.
    find . -type f ! -name "$MANIFEST" ! -name README.md ! -name .DS_Store | sed 's|^\./||' | LC_ALL=C sort |
      while IFS= read -r f; do
        if ! awk -v f="$f" 'substr($0, 67) == f { found = 1 } END { exit !found }' "$MANIFEST"; then
          shasum -a 256 "$f" >> "$MANIFEST"
          echo "added $f"
        fi
      done
    ;;
  *)
    echo "usage: $0 [check|add]"; exit 2 ;;
esac
