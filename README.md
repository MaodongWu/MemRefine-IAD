# MemRefine-IAD

A Visual ReAct agent baseline for Industrial Anomaly Detection (IAD), adapted from the GeoVista-style project layout with the core tool pair:

- `image_zoom_in_tool`: crop and zoom local evidence
- `retrieve_kb`: retrieve supporting context from an industrial anomaly knowledge base

## Project Goals

- Keep the **agent loop** simple and auditable.
- Separate **tool implementations** from **agent reasoning**.
- Provide a clean bridge to training stacks (recommended: EasyR1).

## Quick Start

```bash
cd /mnt/nfs/wmd/research/MemRefine-IAD
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python examples/infer_example.py \
  --image /mnt/nfs/wmd/data/MVTec-AD/bottle/test/broken_large/000.png \
  --question "Is there anomaly? localize and explain."
```

If the sample path does not exist yet, use any local image path.

### Run with local vLLM (Qwen2.5-VL)

```bash
export MODEL_PATH=/mnt/nfs/wmd/model/Qwen2.5-VL-7B-Instruct
export VLLM_HOST=localhost
export VLLM_PORT=8000

bash inference/vllm_deploy.sh
```

Then in another shell:

```bash
python examples/infer_example.py \
  --image /mnt/nfs/wmd/data/MVTec-AD/bottle/test/broken_large/000.png \
  --question "Is there anomaly? localize and explain." \
  --backend vllm \
  --host localhost \
  --port 8000 \
  --model-name qwen2.5-vl
```

## Layout

- `agent/`: ReAct loop, prompt templates, tool call parser
- `tools/`: zoom/retrieval tool implementations + unified tool schema (`tools/spec.py`)
- `inference/`: serving and batch inference scripts
- `eval/`: IAD metrics scripts (image-level + pixel-level)
- `scripts/`: data conversion helpers (e.g., SFT jsonl)
- `training/`: EasyR1 integration notes and config stubs
- `configs/`: runtime config yaml files
- `tests/`: smoke tests for tool interface stability

## Recommended Training Strategy

- Keep inference/eval orchestration here.
- Reuse `EasyR1-main` for SFT/RL optimization loops.
- Exchange data through JSONL + unified conversation format.

## Tool Backends

Current tool stack:

- `image_zoom_in_tool`: local crop + save
- `retrieve_kb`: local JSONL industrial knowledge base (fallback to builtin seed entries)

Optional environment variables:

- `IAD_KB_PATH=/abs/path/to/industrial_kb.jsonl`
- `IAD_CROP_DIR=.temp/crops`

## Smoke Test

```bash
cd /mnt/nfs/wmd/research/MemRefine-IAD
python -m unittest tests/test_tools_smoke.py -v
```
