# MemRefine-IAD

MemRefine-IAD is an industrial anomaly detection project with two complementary modes:

1. Known-category inspection: directly use a VLM to judge anomaly and describe location.
2. Unknown-category inspection: use user-provided normal samples for feature-distance anomaly judgment, then call VLM for localization and explanation.

---

## Web App (Real-time Demo)

### Features

- Known category mode:
  - Input: one test image.
  - Output: anomaly decision (`A/B`) + anomaly location + rationale.
- Unknown category mode:
  - Input: one test image + several normal reference images.
  - Output: distance-based anomaly score + threshold-based decision + VLM location/rationale + fused final decision.

### Demo Launch Link

The demo is self-hosted. Users must provide their own VLM service URL/model/API key in the app sidebar.

- Demo entry (after startup): [http://localhost:7862](http://localhost:7862)
- App source: [web/app.py](web/app.py)

Start command:

```bash
cd /mnt/nfs/wmd/research/MemRefine-IAD/web
/mnt/nfs/wmd/conda_envs/vllm/bin/streamlit run app.py --server.port 7862 --server.address 0.0.0.0
```

### Detection Screenshot (Placeholder)

> Replace this placeholder with your final web demo screenshot.

![MemRefine-IAD Web Demo Screenshot Placeholder](assets/web_demo_screenshot_placeholder.png)

---

## Trained Models

The following model entries are prepared. Replace links with your final Hugging Face model pages.

| Model | Description | Download Link |
|---|---|---|
| MemRefine(LLaVA-OneVision-SI-7B) | SFT model for agent-style anomaly reasoning | [HF link (to be updated)](https://huggingface.co/your-org/MemRefine-LLaVA-OneVision-SI-7B) |
| MemRefine(Qwen2.5-VL-Instruct-7B) | MemRefine-compatible Qwen 7B variant | [HF link (to be updated)](https://huggingface.co/your-org/MemRefine-Qwen2.5-VL-7B) |
| MemRefine(Qwen2.5-VL-Instruct-3B) | MemRefine-compatible Qwen 3B variant | [HF link (to be updated)](https://huggingface.co/your-org/MemRefine-Qwen2.5-VL-3B) |

---

## Repository Layout

- `web/`: Streamlit demo app (`web/app.py`, `web/backend.py`)
- `scripts/`: inference, evaluation, and training shell scripts
- `scripts/train/`: training launch scripts you keep for reproducibility
- `data/`: evaluation data/config files used by benchmark runs
- `configs/`: project runtime config files
- `assets/`: figures and static assets for docs/demo
- `release/`: preflight publishing helpers

---

## Quick Start (Web)

Install dependencies:

```bash
cd /mnt/nfs/wmd/research/MemRefine-IAD/web
/mnt/nfs/wmd/conda_envs/vllm/bin/pip install -r requirements.txt
```

Run:

```bash
/mnt/nfs/wmd/conda_envs/vllm/bin/streamlit run /mnt/nfs/wmd/research/MemRefine-IAD/web/app.py --server.port 7862 --server.address 0.0.0.0
```

Then open:

- Local: `http://localhost:7862`
- LAN: `http://<your_server_ip>:7862`

---

## Notes for Public Users

- This repository does not ship hosted VLM endpoints.
- To use the web demo, configure your own OpenAI-compatible VLM service in the sidebar:
  - `Base URL` (e.g., local vLLM OpenAI API endpoint)
  - `Model Name`
  - `API Key`

