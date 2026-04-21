# MemRefine-IAD Web

A Streamlit web app with two modes:

1. **Known category**: direct anomaly + location via VLM call.
2. **Unknown category**: user-provided normal references -> feature-distance anomaly score + VLM location.

## Start

```bash
cd /mnt/nfs/wmd/research/MemRefine-IAD/web

# Option A: install deps into your current env
pip install -r requirements.txt

# Option B: if using existing vllm env
/mnt/nfs/wmd/conda_envs/vllm/bin/pip install -r requirements.txt
```

Run:

```bash
cd /mnt/nfs/wmd/research/MemRefine-IAD/web
streamlit run app.py --server.port 7861 --server.address 0.0.0.0
```

or:

```bash
/mnt/nfs/wmd/conda_envs/vllm/bin/streamlit run /mnt/nfs/wmd/research/MemRefine-IAD/web/app.py --server.port 7861 --server.address 0.0.0.0
```

## VLM Backend

This app calls an OpenAI-compatible VLM endpoint:

- Base URL: `http://127.0.0.1:8000/v1` (default)
- API key: `EMPTY` (default)
- Model name: set in UI sidebar

If you use local vLLM serving, make sure your model server is started first.
