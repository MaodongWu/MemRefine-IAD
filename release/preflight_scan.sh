#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT_DIR="$ROOT/release"
TS="$(date +%Y%m%d_%H%M%S)"
REPORT="$OUT_DIR/preflight_report_${TS}.md"

mkdir -p "$OUT_DIR"

{
  echo "# MemRefine-IAD Preflight Report"
  echo
  echo "- Time: $(date '+%F %T %Z')"
  echo "- Root: $ROOT"
  echo

  echo "## 1) Directory Size Overview"
  echo '```'
  du -sh "$ROOT"/* "$ROOT"/.[^.]* 2>/dev/null | sort -h || true
  echo '```'
  echo

  echo "## 2) Files Larger Than 20MB"
  echo '```'
  find "$ROOT" -type f -size +20M \
    ! -path "$ROOT/.git/*" \
    | sed "s#^$ROOT/##" | sort || true
  echo '```'
  echo

  echo "## 3) Sensitive Filename Check"
  echo '```'
  find "$ROOT" -type f \
    \( -name "*.key" -o -name "*.pem" -o -name "*.p12" -o -name "*.pfx" -o -name "*.secret" -o -name ".env" -o -name ".dashscope_key" \) \
    ! -path "$ROOT/.git/*" \
    | sed "s#^$ROOT/##" | sort || true
  echo '```'
  echo

  echo "## 4) Secret-like Pattern Scan (code/text only)"
  echo '```'
  rg -n --hidden -S \
    -g '!**/.git/**' \
    -g '!result/**' \
    -g '!logs/**' \
    -g '!data/**' \
    -g '!**/*.ipynb' \
    -g '*.py' -g '*.sh' -g '*.yaml' -g '*.yml' -g '*.md' -g '*.json' -g '*.toml' \
    '(api[_-]?key\s*[=:]|secret\s*[=:]|password\s*[=:]|token\s*[=:]|AKIA[0-9A-Z]{16}|sk-[A-Za-z0-9]{20,})' \
    "$ROOT" || true
  echo '```'
  echo

  echo "## 5) Gitignore Sanity"
  echo '```'
  if [ -f "$ROOT/.gitignore" ]; then
    sed -n '1,220p' "$ROOT/.gitignore"
  else
    echo '.gitignore NOT FOUND'
  fi
  echo '```'
  echo

  echo "## 6) Suggested Next Commands"
  echo '```bash'
  echo "cd $ROOT"
  echo "git init"
  echo "git add ."
  echo "git status --short"
  echo "# Verify nothing sensitive is staged before first commit"
  echo '```'
} > "$REPORT"

echo "[OK] Report generated: $REPORT"
