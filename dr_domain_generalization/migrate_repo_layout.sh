#!/usr/bin/env bash
# Flatten the repository: make D:/Research Code contain the project directly.
#
# The problem this fixes
# ----------------------
# The git root is D:/Research Code, and the project exists inside it TWICE:
#
#   D:/Research Code/                       403 tracked files, a stale snapshot
#   D:/Research Code/dr_domain_generalization/   435 tracked files, the live tree
#
# The two disagree on published numbers. The root copy still says the
# deployment cost is -0.147 QWK and +128% severe errors; the live tree says
# -0.1376 and +107% after the three-seed paired re-analysis. The root copy also
# has no Phase 8, no audit_consistency.py and no analyse_severe_error.py.
#
# Anyone cloning the repo reads the top level first, and the top level is wrong.
#
# What this does
# --------------
# Deletes the stale root copy and moves the live tree up one level, so there is
# exactly one copy of every file and the README renders on the repo landing
# page. Nothing is lost: the deleted copy stays in git history, and every file
# it contained also exists in the live tree (asserted below, not assumed).
#
# Safety
# ------
# * Refuses to run while training is active. The live tree holds ~15 GB of
#   checkpoints and an open log; on Windows, moving a directory with open
#   handles fails partway and leaves the repo half-migrated.
# * Refuses unless the live tree is a strict superset of the root copy.
# * Refuses if the working tree is dirty, so the migration is the only change
#   in the resulting commit and is trivial to revert.
# * Does not push. Verify, then push by hand.
#
# Usage:
#     bash migrate_repo_layout.sh --dry-run     # show what would happen
#     bash migrate_repo_layout.sh

set -euo pipefail

ROOT="/d/Research Code"
LIVE="$ROOT/dr_domain_generalization"
DRY=0
[ "${1:-}" = "--dry-run" ] && DRY=1

say() { echo "  $*"; }
die() { echo "ABORT: $*" >&2; exit 1; }

cd "$ROOT" || die "cannot enter $ROOT"

echo "== preflight =="

[ -d "$LIVE" ] || die "$LIVE does not exist; migration may already be done"
[ -d "$ROOT/.git" ] || die "$ROOT/.git not found; wrong directory"

# 1. Nothing may be training. A python process from the project venv means a
#    run is live and its file handles will break the move.
if ps -W 2>/dev/null | grep -q "dr_domain_generalization/.venv/Scripts/python"; then
  die "a training run is active (project venv python is running). Wait for it to finish."
fi
say "no training process running"

# 2. The working tree must be clean, so this commit contains only the move.
if [ -n "$(git status --porcelain)" ]; then
  git status --short | head -20
  die "working tree is dirty; commit or stash first"
fi
say "working tree clean"

# 3. Every tracked file at the root must also exist in the live tree, or
#    deleting the root copy would lose something. Checked, not assumed.
missing=0
while IFS= read -r f; do
  [ -z "$f" ] && continue
  if [ ! -e "$LIVE/$f" ]; then
    echo "    root-only file, would be LOST: $f"
    missing=$((missing + 1))
  fi
done < <(git ls-files | grep -v '^dr_domain_generalization/')
[ "$missing" -eq 0 ] || die "$missing file(s) exist only in the root copy"
say "live tree is a superset of the root copy (nothing would be lost)"

root_count=$(git ls-files | grep -vc '^dr_domain_generalization/' || true)
live_count=$(git ls-files | grep -c '^dr_domain_generalization/' || true)
say "root copy: $root_count tracked files | live tree: $live_count tracked files"

if [ "$DRY" -eq 1 ]; then
  echo
  echo "== dry run: would delete the root copy, then move the live tree up =="
  git ls-files | grep -v '^dr_domain_generalization/' | sed 's/^/    rm  /' | head -12
  echo "    ... ($root_count files)"
  echo "    mv  dr_domain_generalization/*  ->  ."
  exit 0
fi

echo
echo "== migrating =="

# Delete the stale root copy. Only entries that are not .git and not the live
# folder, so the repository and the tree being kept are never touched.
for entry in * .[!.]*; do
  [ "$entry" = "dr_domain_generalization" ] && continue
  [ "$entry" = ".git" ] && continue
  [ -e "$entry" ] || continue
  rm -rf -- "$entry"
done
say "stale root copy removed"

# Move the live tree up, including dotfiles.
shopt -s dotglob
mv -- "$LIVE"/* "$ROOT"/
shopt -u dotglob
rmdir -- "$LIVE"
say "live tree moved to the repository root"

git add -A
say "staged"

echo
echo "== result =="
git status --short | head -8
echo "  ..."
echo "  $(git status --porcelain | wc -l) path(s) changed"
echo
echo "Next, in this order:"
echo "  cd '$ROOT'"
echo "  ./.venv/Scripts/python.exe -m pytest tests/ -q -m 'not slow'"
echo "  ./.venv/Scripts/python.exe audit_consistency.py"
echo "  git commit -m 'Flatten repo layout: single copy of the project at the root'"
echo "  git push"
echo
echo "The venv moved too. pyvenv.cfg points at the base Python, so"
echo ".venv/Scripts/python.exe re-resolves its own prefix and keeps working;"
echo "console entry points such as pip.exe have the old path baked in, so use"
echo "'python -m pip' if you need to install anything."
