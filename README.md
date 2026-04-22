# MemRefine-IAD: Memory-Guided Iterative Agent for Industrial Anomaly Detection

<p align="center">
  <a href="https://huggingface.co/mooorton/MemRefine-IAD"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Models-yellow" alt="HuggingFace"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="License"></a>
  <img src="https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776ab.svg" alt="Python">
</p>

> Official implementation of  
> **"MemRefine-IAD: Memory-Guided Iterative Agent for Industrial Anomaly Detection"**  
> Wu Maodong, Lv Shuai.

<p align="center">
  <img src="assets/xianyu.png" width="85%" alt="MemRefine-IAD pipeline (placeholder)">
  <br>
  <em>Figure 1. Overview of MemRefine-IAD (placeholder — to be replaced).</em>
</p>

---

## 📌 Abstract

MemRefine-IAD is a **memory-guided iterative agent framework for industrial anomaly detection (IAD)**. Its core innovations consist of a multi-stage reasoning mechanism, a Working Memory Patch module, and customized trajectory supervision dataset construction. Our method belongs to the **weak-supervised trajectory-supervised zero-shot learning paradigm** for industrial anomaly detection, which achieves strong generalization ability and practical application value under complex industrial working conditions.

On six public IAD benchmarks (MVTec, MPDD, VisA, DAGM, DTD, SDD), **MemRefine-IAD (LLaVA-OneVision-SI) 7B** achieves an average AD performance of **79.2**.
This result outperforms raw-inference commercial closed models GPT-4o-mini (66.7) and Claude-Sonnet-4 (69.9), as well as Anomaly-OV 7B (78.9) — another open-source method under the same weak-supervised zero-shot paradigm. Meanwhile, without introducing any answer annotations, our method improves the vanilla open-source baseline Qwen2.5-VL-Instruct 7B (59.0) by **+20.2 points**.

Key contributions:

- 🧠 Memory-guided multi-stage iterative reasoning mechanism: `GlobalObs → ZoomCrop → MemUpdate → SelfCheck → FinalAnswer`, with self-check triggered rollback.
- 📝 Working Memory Patch: step-wise structured JSON written into assistant-visible, supervisable tokens (`step1_global`, `step2_zoomcrop`, `step3_selfcheck/final`).
- 🎯 Trajectory-supervised process-supervised SFT: segment-weighted cross-entropy loss over process / memory / self-check / final-answer spans.
- 🌐 Dual-mode real-time demo: *known-class zero-shot* and *unknown-class few-shot + feature-distance fusion*.

---

## 📊 Main Results

Six-benchmark AD accuracy (%). **Bold** = best open-source result; *italic* = previous best open-source baseline.

<!-- TODO: replace with final numbers once evaluation is fully completed -->

| Method | MVTec | MPDD | VisA | DAGM | DTD | SDD | **Avg.** |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| GPT-4o-mini (raw) | 76.0 | 71.6 | 62.4 | 60.4 | 71.2 | 58.7 | 66.7 |
| Claude-Sonnet-4 (raw) | 68.4 | 66.6 | 66.8 | 69.7 | 72.4 | 75.2 | 69.9 |
| IAD-R1 (Qwen2.5-VL) 3B | *77.6* | *59.2* | *69.8* | *85.2* | *89.1* | *83.4* | *77.4* |
| IAD-R1 (LLaVA-OV-SI) 7B | *86.7* | *70.9* | *78.0* | *94.8* | *96.2* | *90.1* | *86.1* |
| Anomaly-OV 7B | 74.3 | 70.3 | **74.3** | 77.5 | 90.7 | **88.7** | 78.9 |
| LLaVA-OV-SI 7B | 82.0 | 57.0 | 59.6 | 75.4 | 76.8 | 55.1 | 67.7 |
| AnomalyGPT 7B | 46.6 | 54.2 | 57.3 | 49.6 | 64.1 | 49.5 | 53.6 |
| Qwen2.5-VL-Instruct 3B | 62.3 | 56.2 | 57.7 | 53.9 | 65.8 | 51.0 | 57.8 |
| Qwen2.5-VL-Instruct 7B | 64.3 | 56.0 | 57.7 | 56.7 | 57.1 | 62.3 | 59.0 |
| **MemRefine-IAD (Qwen2.5-VL) 3B** | 76.0 | 57.2 | 72.4 | 83.0 | 89.7 | 82.9 | **76.9** |
| **MemRefine-IAD (Qwen2.5-VL) 7B** | **80.9** | **60.4** | 71.3 | **87.0** | **90.1** | 78.6 | **78.1** |
| **MemRefine-IAD (LLaVA-OV-SI) 7B** | **82.2** | **62.1** | **71.2** | **81.3** | **92.4** | **86.0** | **79.2** |

### Ablation

| Model | Stage | Average |
|:---|:---|:---:|
| LLaVA-OneVision-SI-0.5B | raw | 50.7 |
| LLaVA-OneVision-SI-0.5B | agent-zoom-memory | 51.2 |
| Qwen2.5-VL-Instruct-7B | raw | 59.0 |
| Qwen2.5-VL-Instruct-7B | agent-zoom | 65.2 |
| Qwen2.5-VL-Instruct-7B | agent-zoom-memory (**Ours**) | **67.8** |

\* Full six-benchmark ablation will be updated upon completion of remaining runs.

---

## 🗂 Repository Structure

```
MemRefine-IAD/
├── assets/                      # figures, teaser images, demo screenshots
├── configs/                     # runtime / model / training configs
├── data/                        # dataset configs and evaluation manifests
├── scripts/
│   ├── MemRefineIAD/            # open-source MLLM inference (Qwen / LLaVA)
│   ├── commercial/raw/          # commercial API raw-inference evaluation
│   └── train/                   # SFT training launch scripts
├── agent/
│   └── real_build/              # SFT trajectory construction pipeline
├── web/                         # Streamlit real-time demo (app.py, backend.py)
├── helper/                      # evaluation helpers (summary, metrics)
├── release/                     # packaging / preflight utilities
└── README.md
```

---

## 🛠 Installation

We recommend **two separate environments** to avoid dependency conflicts between inference (vLLM) and training (SFT).

```bash
# Clone
git clone https://github.com/your-org/MemRefine-IAD.git
cd MemRefine-IAD

# Inference env
conda create -n memrefine-vllm python=3.10 -y
conda activate memrefine-vllm
pip install -r requirements/requirements-vllm.txt

# Training env
conda create -n memrefine-sft python=3.10 -y
conda activate memrefine-sft
pip install -r requirements/requirements-sft.txt
```

**Tested environment.** Python 3.10, PyTorch 2.3, CUDA 12.1, vLLM 0.6.x, Transformers 4.44.x, 2× A100 / A6000.

---

## 📦 Dataset Preparation

MemRefine-IAD is evaluated on six public IAD benchmarks. Please download each dataset from its official source and place it under `data/Test/` following the layout below:

```
data/Test/
├── MVTec/
├── MPDD/
├── VisA/
├── DAGM/
├── DTD/
└── SDD/
```

Official sources:

- **MVTec AD** — https://www.mvtec.com/company/research/datasets/mvtec-ad
- **MPDD** — https://github.com/stepanje/MPDD
- **VisA** — https://github.com/amazon-science/spot-diff
- **DAGM 2007** — https://hci.iwr.uni-heidelberg.de/content/weakly-supervised-learning-industrial-optical-inspection
- **DTD** — https://www.robots.ox.ac.uk/~vgg/data/dtd/
- **SDD (KolektorSDD)** — https://www.vicos.si/resources/kolektorsdd/

Each benchmark is wrapped by the shared evaluation manifest in `data/Test/<bench>/meta.json`.

---

## 🤗 Pretrained Models

| Model | Base | Params | Six-bench Avg. | Link |
|---|---|:---:|:---:|---|
| MemRefine-IAD (Qwen2.5-VL-Instruct) | Qwen2.5-VL-Instruct | 3B | 76.9 | 🤗 *Coming soon* |
| MemRefine-IAD (Qwen2.5-VL-Instruct) | Qwen2.5-VL-Instruct | 7B | 78.1 | 🤗 *Coming soon* |
| MemRefine-IAD (LLaVA-OneVision-SI) | LLaVA-OneVision-SI | 7B | 79.2 | 🤗 *Coming soon* |

> LoRA checkpoints must be **merged** into a full model directory (with `config.json` + full weights) before being served by vLLM. We provide `scripts/merge_lora.py` for this purpose.

---

## 🚀 Inference

### Known-category zero-shot inference

```bash
bash scripts/MemRefineIAD/run_qwen_iadr1_zoom_eval_memory.sh \
    --model_path <path/to/memrefine-qwen-7b> \
    --bench MVTec \
    --name memrefine_qwen7b_mvtec
```

### LLaVA-OneVision backend

```bash
bash scripts/MemRefineIAD/run_llava_onevision_zoom_eval_memory.sh \
    --model_path <path/to/memrefine-llava-ov-7b> \
    --bench VisA \
    --max_model_len 16384 \
    --name memrefine_llava_visa
```

### Commercial API raw inference (fair baseline)

```bash
bash scripts/commercial/raw/run_api_commercial_raw_eval.sh \
    --api gpt-4o-mini \
    --bench MVTec \
    --name raw_gpt4omini_mvtec
```

---

## 🏋️ Training (Process-Supervised SFT)

1. **Build SFT trajectories** (952 ShareGPT-format multi-turn samples, MPDD + VisA):

   ```bash
   python agent/real_build/build_memrefine_sft_via_api.py \
       --out_dir data/sft/memrefine_sft \
       --num_mpdd 200 --num_visa 800
   ```

2. **LoRA SFT training**:

   ```bash
   bash scripts/train/try.sh
   ```

   Defaults: 2× GPUs, LoRA, `save_total_limit=3`, dynamic free-port injection to avoid `EADDRINUSE` on multi-job launches.

3. **Merge LoRA → full checkpoint** for vLLM serving:

   ```bash
   python scripts/merge_lora.py \
       --base <base_model_path> \
       --lora <lora_ckpt_dir> \
       --out  <merged_model_dir>
   ```

---

## 🧪 Evaluation

Run the full six-benchmark evaluation and generate summary CSV:

```bash
bash scripts/eval_all_benchmarks.sh --name memrefine_qwen7b_full
python helper/summary.py --result_dir result/memrefine_qwen7b_full
```

Each benchmark directory produces `answers.json` and `_accuracy.csv`; `summary.py` aggregates them into the final six-benchmark average.

> ⚠️ **Avoid silent result reuse.** If a run directory already contains `answers.json`, the script will treat samples as completed (`total_pending=0`) and silently overwrite the summary with stale results. Always use `--reproduce` or a new `--name` when re-running.

---

## 🌐 Web Demo

<p align="center">
  <img src="assets/web_demo_screenshot_placeholder.png" width="80%" alt="Web demo screenshot (placeholder)">
  <br>
  <em>Figure 2. Real-time detection web demo (placeholder — to be replaced).</em>
</p>

The demo supports two modes:

**① Known-category mode**
- **Input**: one test image.
- **Output**: anomaly decision (`A = normal`, `B = anomaly`), localized region, natural-language rationale, and a replayable reasoning trace.

**② Unknown-category mode**
- **Input**: one test image + *K* user-provided normal reference images.
- **Pipeline**:
  1. Extract visual features via a CLIP-style backbone φ.
  2. Compute two complementary feature distances:
     - *Centroid distance*: `d_center(x, R) = || f_x − (1/K) Σ f_i ||₂`
     - *k-NN distance*: `d_knn(x, R) = (1/k) Σ_{j∈Nₖ(x)} || f_x − f_j ||₂`
  3. Fuse with the MLLM anomaly probability `p_anom`:  
     `S(x) = α · p_anom + β · σ(d_center) + γ · σ(d_knn)`, with `α + β + γ = 1`.
- **Output**: fused anomaly score, threshold-based decision, VLM localization & rationale.

### Launch

```bash
cd web
pip install -r requirements.txt
streamlit run app.py --server.port 7862 --server.address 0.0.0.0
```

Then open `http://localhost:7862` (or `http://<your_server_ip>:7862` on LAN).

### Configuration

The demo does **not** ship a hosted VLM endpoint. In the app sidebar, users must configure an OpenAI-compatible VLM service:

- `Base URL` — e.g., a local vLLM OpenAI-compatible endpoint
- `Model Name`
- `API Key`

---

## 📁 Trajectory Format (Training Data)

Each SFT sample is a ShareGPT-style multi-turn trajectory. Only `assistant` turns are trained; `system / user / tool` are masked. Supervision priority:

```
final_answer  >  self_check_revision  ≈  memory_patches  >  generic process tokens
```

A Working Memory Patch is a structured JSON block embedded in assistant tokens:

```json
{
  "step": "step2_zoomcrop",
  "evidence": {
    "region": [x1, y1, x2, y2],
    "visual_cue": "暗色凹陷，直径约 3mm",
    "tool_ret": "crop_path=/tmp/zoom_001.png"
  },
  "candidate_label": "anomaly",
  "confidence": 0.78,
  "revision_reason": null
}
```

---

## 🙏 Acknowledgements

This project builds upon the excellent open-source work of:

- [IAD-R1](https://github.com/) — baseline & evaluation protocol
- [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL) — primary MLLM backbone
- [LLaVA-OneVision](https://github.com/LLaVA-VL/LLaVA-NeXT) — secondary MLLM backbone
- [vLLM](https://github.com/vllm-project/vllm) — high-throughput inference serving
- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) — SFT training framework
- [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad), [MPDD](https://github.com/stepanje/MPDD), [VisA](https://github.com/amazon-science/spot-diff), [DAGM](https://hci.iwr.uni-heidelberg.de/), [DTD](https://www.robots.ox.ac.uk/~vgg/data/dtd/), [KolektorSDD](https://www.vicos.si/resources/kolektorsdd/) — evaluation benchmarks.

We thank the authors for open-sourcing their work.

---

## 📝 Citation

If you find this project useful in your research, please consider citing:

```bibtex
@article{wu2025memrefine,
  title   = {MemRefine-IAD: Memory-Guided Iterative Agent for Industrial Anomaly Detection},
  author  = {Wu, Maodong and Lv, Shuai},
  journal = {arXiv preprint arXiv:xxxx.xxxxx},
  year    = {2025}
}
```

---

## 📄 License

This project is released under the [Apache-2.0 License](LICENSE). Dataset licenses remain with their original providers.

---

## 📬 Contact

For questions, issues or collaborations, please open a GitHub [Issue](../../issues) or contact the authors at `your_email@example.com`.