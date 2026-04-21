#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

cat <<EOF
[INFO] This helper does NOT modify code files.
[INFO] It only prints safe git workflow commands.

Run manually:

cd "$ROOT"

# 1) Initialize git repo (if not already)
git init

# 2) Check what will be staged
git add .
git status --short

# 3) (IMPORTANT) If any sensitive file appears, unstage it
# git restore --staged <path>

# 4) First commit
git commit -m "init: MemRefine-IAD public baseline"

# 5) Add remote and push
git remote add origin <your-github-repo-url>
git branch -M main
git push -u origin main
EOF
