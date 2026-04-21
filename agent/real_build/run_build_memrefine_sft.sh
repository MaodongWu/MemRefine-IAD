#!/usr/bin/env bash
set -euo pipefail

PY=/mnt/nfs/wmd/conda_envs/vllm/bin/python
SCRIPT=/mnt/nfs/wmd/research/MemRefine-IAD/agent/real_build/build_memrefine_sft_via_api.py
OUTDIR=/mnt/nfs/wmd/research/MemRefine-IAD/agent/real_build
LOG=/mnt/nfs/wmd/research/MemRefine-IAD/agent/real_build/build_memrefine_sft_gpt54.log

mkdir -p "$OUTDIR"

$PY "$SCRIPT" \
  --mpdd_tsv /mnt/nfs/wmd/research/MemRefine-IAD/agent/sft_non_test_MPDD.tsv \
  --visa_tsv /mnt/nfs/wmd/research/MemRefine-IAD/agent/sft_non_test_VisA.tsv \
  --out_dir "$OUTDIR" \
  --api_key_file /mnt/nfs/wmd/research/MemRefine-IAD/.dashscope_key \
  --api_url https://api.vectorengine.ai/v1/chat/completions \
  --model gpt-5.4 \
  --n_mpdd 200 \
  --n_visa 800 \
  --concurrency 16 \
  --seed 42 \
  --temperature 0.2 \
  --top_p 1.0 \
  --max_tokens 700 \
  --timeout 120 \
  --max_retries 5 \
  --retry_sleep 2.0 \
  | tee "$LOG"
