# MemRefine-IAD Preflight Report

- Time: 2026-04-21 21:20:06 CST
- Root: /mnt/nfs/wmd/research/MemRefine-IAD

## 1) Directory Size Overview
```
4.0K	/mnt/nfs/wmd/research/MemRefine-IAD/.dashscope_key
4.0K	/mnt/nfs/wmd/research/MemRefine-IAD/.gitignore
4.0K	/mnt/nfs/wmd/research/MemRefine-IAD/README.md
4.0K	/mnt/nfs/wmd/research/MemRefine-IAD/assets
4.0K	/mnt/nfs/wmd/research/MemRefine-IAD/configs
4.0K	/mnt/nfs/wmd/research/MemRefine-IAD/requirements.txt
16K	/mnt/nfs/wmd/research/MemRefine-IAD/release
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
/mnt/nfs/wmd/research/MemRefine-IAD/web/app.py:49:    api_key = st.text_input("API Key", value=os.getenv("VLM_API_KEY", "EMPTY"), type="password")
/mnt/nfs/wmd/research/MemRefine-IAD/web/app.py:86:                    api_key=api_key,
/mnt/nfs/wmd/research/MemRefine-IAD/web/app.py:148:                    api_key=api_key,
/mnt/nfs/wmd/research/MemRefine-IAD/scripts/commercial/MemRefineAgent/run_api_memrefine_zoom_eval_memory.py:245:    api_key: str,
/mnt/nfs/wmd/research/MemRefine-IAD/scripts/commercial/MemRefineAgent/run_api_memrefine_zoom_eval_memory.py:290:    api_key: str,
/mnt/nfs/wmd/research/MemRefine-IAD/scripts/commercial/MemRefineAgent/run_api_memrefine_zoom_eval_memory.py:485:def eval_one_bench(args: argparse.Namespace, api_key: str, bench: str) -> None:
/mnt/nfs/wmd/research/MemRefine-IAD/scripts/commercial/MemRefineAgent/run_api_memrefine_zoom_eval_memory.py:557:                api_key=api_key,
/mnt/nfs/wmd/research/MemRefine-IAD/scripts/commercial/MemRefineAgent/run_api_memrefine_zoom_eval_memory.py:612:    api_key = read_api_key(args.api_key_file)
/mnt/nfs/wmd/research/MemRefine-IAD/scripts/MemRefineIAD/run_qwen_iadr1_zoom_eval.py:221:                "Answer strictly with one token: Yes or No."
/mnt/nfs/wmd/research/MemRefine-IAD/scripts/commercial/raw/run_api_commercial_raw_eval.py:148:    api_key: str,
/mnt/nfs/wmd/research/MemRefine-IAD/scripts/commercial/raw/run_api_commercial_raw_eval.py:200:def eval_one_bench(args: argparse.Namespace, api_key: str, bench: str) -> None:
/mnt/nfs/wmd/research/MemRefine-IAD/scripts/commercial/raw/run_api_commercial_raw_eval.py:259:                api_key=api_key,
/mnt/nfs/wmd/research/MemRefine-IAD/scripts/commercial/raw/run_api_commercial_raw_eval.py:323:    api_key = read_api_key(args.api_key_file)
/mnt/nfs/wmd/research/MemRefine-IAD/web/backend.py:184:    api_key: str = "EMPTY",
/mnt/nfs/wmd/research/MemRefine-IAD/scripts/MemRefineIAD/run_llava_onevision_zoom_eval_memory.py:236:        "Answer strictly with one token: Yes or No."
/mnt/nfs/wmd/research/MemRefine-IAD/scripts/MemRefineIAD/run_qwen_iadr1_zoom_eval_memory.py:245:                "Answer strictly with one token: Yes or No."
/mnt/nfs/wmd/research/MemRefine-IAD/agent/real_build/build_memrefine_sft_via_api.py:149:    api_key: str,
/mnt/nfs/wmd/research/MemRefine-IAD/agent/real_build/build_memrefine_sft_via_api.py:348:                api_key=key,
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
