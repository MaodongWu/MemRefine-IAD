# MemRefine-IAD Preflight Report

- Time: 2026-04-21 21:24:45 CST
- Root: /mnt/nfs/wmd/research/MemRefine-IAD

## 1) Directory Size Overview
```
4.0K	/mnt/nfs/wmd/research/MemRefine-IAD/assets
4.0K	/mnt/nfs/wmd/research/MemRefine-IAD/configs
4.0K	/mnt/nfs/wmd/research/MemRefine-IAD/.dashscope_key
4.0K	/mnt/nfs/wmd/research/MemRefine-IAD/.gitignore
4.0K	/mnt/nfs/wmd/research/MemRefine-IAD/README.md
4.0K	/mnt/nfs/wmd/research/MemRefine-IAD/requirements.txt
24K	/mnt/nfs/wmd/research/MemRefine-IAD/release
56K	/mnt/nfs/wmd/research/MemRefine-IAD/web
68K	/mnt/nfs/wmd/research/MemRefine-IAD/bash
500K	/mnt/nfs/wmd/research/MemRefine-IAD/scripts
13M	/mnt/nfs/wmd/research/MemRefine-IAD/agent
14M	/mnt/nfs/wmd/research/MemRefine-IAD/logs
17M	/mnt/nfs/wmd/research/MemRefine-IAD/data
157M	/mnt/nfs/wmd/research/MemRefine-IAD/result
```

## 2) Files Larger Than 20MB
```
result/MemRefine-IAD_qwen25vl3b_mem_v1/test_DAGM/trace_0_shot_IAD-R1(Qwen2.5-VL-Instruct-3B)_agentzoom_vllm.jsonl
```

## 3) Sensitive Filename Check
```
.dashscope_key
```

## 4) Secret-like Pattern Scan (code/text only)
```
```

## 5) Gitignore Sanity
```
__pycache__/
*.pyc
.env
.temp/
outputs/
logs/
*.log
.DS_Store

# Secrets / credentials
.dashscope_key
*.key
*.pem
*.p12
*.pfx
*.crt
*.secret
secrets/
**/secrets/

# Large experiment artifacts
result/
data/
wandb/
runs/
checkpoints/
**/checkpoint-*/
**/merged-checkpoint-*/
*.pt
*.pth
*.bin
*.safetensors
*.ckpt

# Python / tooling caches
.pytest_cache/
.mypy_cache/
.ruff_cache/
.coverage
coverage.xml
*.egg-info/
.venv/
venv/

# IDE / OS
.vscode/
.idea/
```

## 6) Suggested Next Commands
```bash
cd /mnt/nfs/wmd/research/MemRefine-IAD
git init
git add .
git status --short
# Verify nothing sensitive is staged before first commit
```
