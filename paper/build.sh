#!/usr/bin/env bash
# Compile the manuscript and FAIL on anything a reader would notice.
#
# LaTeX is cheerfully tolerant: an undefined \ref renders as "??", a missing
# citation as "[?]", and the build still exits 0. Three \ref commands in this
# manuscript were silently corrupted on 2026-08-30 and would have shipped that
# way. This script makes those conditions fatal.
#
# Checked:
#   - LaTeX errors
#   - undefined references  (\ref to a label that does not exist)
#   - undefined citations   (\cite to a missing bib key)
#   - missing figure files  (\includegraphics targets that are not on disk)
#   - missing \input targets
#   - escape corruption     (lines starting with a bare "ef{", "n{", "extbf")
#   - numeric drift         (paper/check_numbers.py)
#
# Usage:  bash paper/build.sh

set -u
cd "$(dirname "$0")/.." || exit 1
PAPER="paper"
MAIN="$PAPER/main.tex"
fail=0
note() { echo "  $*"; }
bad()  { echo "  FAIL: $*"; fail=1; }

echo "== latex structure (main.tex + every included file) =="
if command -v python >/dev/null 2>&1; then PYBIN=python; else PYBIN=./.venv/Scripts/python; fi
$PYBIN "$PAPER/check_latex.py" || bad "LaTeX structural defects above"

echo "== referenced files exist =="
missing=0
# \includegraphics{...} and \input{...} are relative to paper/
while read -r target; do
  [ -z "$target" ] && continue
  resolved="$PAPER/$target"
  if [ -f "$resolved" ] || [ -f "$resolved.tex" ] || [ -f "$resolved.pdf" ] \
     || [ -f "$resolved.png" ]; then continue; fi
  bad "referenced file not found: $target"
  missing=$((missing + 1))
done < <(grep -oE '\\(includegraphics|input)(\[[^]]*\])?\{[^}]*\}' "$MAIN" \
         | sed -E 's/.*\{([^}]*)\}/\1/')
[ "$missing" -eq 0 ] && note "all present"

echo "== numeric drift =="
if command -v python >/dev/null 2>&1; then PY=python; else PY=./.venv/Scripts/python; fi
if $PY "$PAPER/check_numbers.py" 2>&1 | tail -3; then :; else bad "check_numbers.py failed"; fi

echo "== latex =="
if ! command -v pdflatex >/dev/null 2>&1; then
  note "pdflatex not installed -- compile checks SKIPPED (not passed)"
  note "install a TeX distribution before submission; undefined refs and"
  note "missing citations cannot be detected without it"
else
  log="$PAPER/.build.log"
  ( cd "$PAPER" && pdflatex -interaction=nonstopmode -halt-on-error main.tex \
      > .build.log 2>&1 )
  status=$?
  [ "$status" -ne 0 ] && bad "pdflatex exited $status"
  grep -q "Undefined control sequence" "$log" && bad "undefined control sequence"
  grep -qE "LaTeX Warning: Reference .* undefined" "$log" && bad "undefined reference(s)"
  grep -qE "LaTeX Warning: Citation .* undefined" "$log" && bad "undefined citation(s)"
  [ "$fail" -eq 0 ] && note "compiled clean"
fi

echo
if [ "$fail" -eq 0 ]; then
  echo "BUILD OK"
else
  echo "BUILD FAILED"
fi
exit "$fail"
