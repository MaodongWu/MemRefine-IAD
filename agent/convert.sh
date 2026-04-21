PYTHONNOUSERSITE=1 /mnt/nfs/wmd/conda_envs/sft/bin/python - << 'PY'
import json, re
from pathlib import Path

root = Path("/mnt/nfs/wmd/research/MemRefine-IAD/agent/real_build")
src = root / "MemRefine_sft.json"
dst = root / "MemRefine_sft_qwen.json"
info_path = root / "dataset_info.json"

data = json.load(src.open("r", encoding="utf-8"))

def clean_user_content(text: str) -> str:
    # 移除显式 <image> 占位，交给 qwen2_vl 模板按 image 字段注入
    lines = text.splitlines()
    lines = [ln for ln in lines if ln.strip() != "<image>"]
    t = "\n".join(lines).strip()
    # 再兜底清一次前缀
    t = re.sub(r'^\s*<image>\s*', '', t).strip()
    return t

for sample in data:
    msgs = sample.get("messages", [])
    for m in msgs:
        if m.get("role") == "user" and isinstance(m.get("content"), str):
            m["content"] = clean_user_content(m["content"])

json.dump(data, dst.open("w", encoding="utf-8"), ensure_ascii=False, indent=2)

if info_path.exists():
    info = json.load(info_path.open("r", encoding="utf-8"))
else:
    info = {}

info["MemRefine_sft_qwen"] = {
    "file_name": "MemRefine_sft_qwen.json",
    "formatting": "sharegpt",
    "columns": {
        "messages": "messages",
        "images": "image"
    },
    "tags": {
        "role_tag": "role",
        "content_tag": "content",
        "user_tag": "user",
        "assistant_tag": "assistant",
        "observation_tag": "tool",
        "function_tag": "function_call",
        "system_tag": "system"
    }
}

json.dump(info, info_path.open("w", encoding="utf-8"), ensure_ascii=False, indent=2)
print("saved:", dst)
print("updated:", info_path)
PY
