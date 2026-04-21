#!/usr/bin/env python3
import argparse
import base64
import csv
import json
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from PIL import Image
from tqdm import tqdm

SYSTEM_PROMPT = (
    "You are MemRefine-IAD inspector.\\n\\n"
    "You can use tool-style calls in text (do not execute real code):\\n"
    "1) image_zoom_in_tool\\n"
    "2) self_check_tool\\n\\n"
    "Tool call format (must be exact):\\n"
    "<tool_call>{\\\"name\\\":\\\"TOOL_NAME\\\",\\\"arguments\\\":{...}}</tool_call>\\n\\n"
    "Workflow:\\n"
    "- First do global inspection and write [MEMORY_PATCH] step1_global.\\n"
    "- If uncertain or suspicious, call image_zoom_in_tool.\\n"
    "- After tool response, write [MEMORY_PATCH] step2_zoomcrop and a candidate answer.\\n"
    "- Then call self_check_tool with stage1 conclusion + zoom evidence + candidate answer/confidence.\\n"
    "- After self-check response, write [MEMORY_PATCH] step3_final and output final answer.\\n\\n"
    "Final answer constraint:\\n"
    "- Output exactly one final tag: <answer>A</answer> or <answer>B</answer>.\\n"
    "- A = No anomaly, B = Anomaly exists.\\n"
)

USER_PROMPT = (
    "<image>\\n"
    "Question: Is there an anomaly in this image?\\n"
    "Options:\\n"
    "A. No anomaly\\n"
    "B. Anomaly exists\\n"
    "Please do global inspection first, then local verification if needed."
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build MemRefine-IAD SFT dataset with external API")
    p.add_argument("--mpdd_tsv", type=str, default="/mnt/nfs/wmd/research/MemRefine-IAD/agent/sft_non_test_MPDD.tsv")
    p.add_argument("--visa_tsv", type=str, default="/mnt/nfs/wmd/research/MemRefine-IAD/agent/sft_non_test_VisA.tsv")
    p.add_argument("--out_dir", type=str, default="/mnt/nfs/wmd/research/MemRefine-IAD/agent/real_build")
    p.add_argument("--api_key_file", type=str, default="/mnt/nfs/wmd/research/MemRefine-IAD/.dashscope_key")
    p.add_argument("--api_url", type=str, default="https://api.vectorengine.ai/v1/chat/completions")
    p.add_argument("--model", type=str, default="gpt-5.4")

    p.add_argument("--n_mpdd", type=int, default=200)
    p.add_argument("--n_visa", type=int, default=800)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--timeout", type=int, default=120)
    p.add_argument("--max_retries", type=int, default=5)
    p.add_argument("--retry_sleep", type=float, default=2.0)

    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--max_tokens", type=int, default=700)
    return p.parse_args()


def read_key(path: str) -> str:
    key = Path(path).read_text(encoding="utf-8").strip()
    if not key:
        raise RuntimeError(f"API key is empty: {path}")
    return key


def load_tsv(path: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            if row.get("abs_path"):
                rows.append(row)
    return rows


def encode_image_data_url(image_path: str, max_side: int = 1024) -> str:
    p = Path(image_path)
    if not p.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")
    with Image.open(str(p)) as im:
        if im.mode != "RGB":
            im = im.convert("RGB")
        w, h = im.size
        scale = min(1.0, float(max_side) / max(w, h))
        if scale < 1.0:
            im = im.resize((int(w * scale), int(h * scale)))
        buf = BytesIO()
        im.save(buf, format="JPEG", quality=92)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def extract_defect_and_product(rel_path: str) -> Tuple[str, str]:
    parts = rel_path.replace("\\", "/").split("/")
    product = parts[1] if len(parts) > 1 else "unknown"
    defect = parts[-2] if len(parts) > 2 else "unknown"
    return product, defect


def expected_label_from_rel(rel_path: str) -> str:
    rp = rel_path.lower()
    return "A" if "/good/" in rp else "B"


def safe_json_extract(text: str) -> Optional[Dict[str, Any]]:
    text = (text or "").strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass

    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def clamp_bbox(v: Any) -> List[float]:
    if isinstance(v, list) and len(v) == 4:
        out: List[float] = []
        for x in v:
            try:
                fx = float(x)
            except Exception:
                fx = 0.5
            out.append(max(0.0, min(1.0, fx)))
        return out
    return [0.45, 0.45, 0.2, 0.2]


def call_api(
    api_url: str,
    api_key: str,
    model: str,
    content: List[Dict[str, Any]],
    temperature: float,
    top_p: float,
    max_tokens: int,
    timeout: int,
    max_retries: int,
    retry_sleep: float,
) -> str:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }
    last: Optional[Exception] = None
    for i in range(max_retries):
        try:
            resp = requests.post(api_url, headers=headers, json=payload, timeout=timeout)
            if resp.status_code >= 400:
                raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:500]}")
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            last = e
            if i < max_retries - 1:
                time.sleep(retry_sleep * (2 ** i))
    raise RuntimeError(f"API failed after retries: {last}")


def build_generation_prompt(rel_path: str, expected_label: str, product: str, defect_type: str) -> str:
    return (
        "You are generating one SFT trajectory for an industrial anomaly agent.\\n"
        "Return ONLY one JSON object with these keys exactly:\\n"
        "step1_global (string), bbox (array of 4 floats in [0,1]), zoom_reason (string),\\n"
        "step2_zoomcrop (string), candidate_answer (\"A\" or \"B\"), candidate_confidence (float 0~1),\\n"
        "self_check_note (string), revised_answer (\"A\" or \"B\" or null), step3_final (string).\\n\\n"
        "Rules:\\n"
        f"- expected_label is {expected_label}. Keep final decision consistent with this label.\\n"
        "- Keep wording concise and technical.\\n"
        "- Do not output markdown, no code block, no extra text.\\n\\n"
        f"Context: rel_path={rel_path}, product={product}, defect_type={defect_type}."
    )


def make_sft_item(idx: int, row: Dict[str, str], model_json: Dict[str, Any]) -> Dict[str, Any]:
    rel_path = row["rel_path"]
    abs_path = row["abs_path"]
    dataset = row["dataset"]
    product, defect_type = extract_defect_and_product(rel_path)
    expected = expected_label_from_rel(rel_path)

    step1 = str(model_json.get("step1_global", "Global observation is inconclusive."))
    bbox = clamp_bbox(model_json.get("bbox"))
    zoom_reason = str(model_json.get("zoom_reason", "verify local evidence"))
    step2 = str(model_json.get("step2_zoomcrop", "Local inspection refines the initial judgment."))

    cand = str(model_json.get("candidate_answer", expected)).strip().upper()
    if cand not in ("A", "B"):
        cand = expected
    try:
        cconf = float(model_json.get("candidate_confidence", 0.8))
    except Exception:
        cconf = 0.8
    cconf = max(0.0, min(1.0, cconf))

    note = str(model_json.get("self_check_note", "Cross-check confirms the decision path."))
    revised = model_json.get("revised_answer", None)
    revised = str(revised).upper() if revised is not None else None
    if revised not in ("A", "B"):
        revised = None

    final_label = revised if revised in ("A", "B") else expected
    if final_label != expected:
        final_label = expected

    step3 = str(model_json.get("step3_final", "Finalize answer after consistency check."))

    item_id = f"memrefine_sft_{dataset.lower()}_{idx:06d}_{product}_{defect_type}".replace(" ", "_")

    msg_assistant_1 = (
        "[MEMORY_PATCH]\\n"
        f"step1_global: {step1}\\n"
        "[/MEMORY_PATCH]\\n"
        f"<tool_call>{{\"name\":\"image_zoom_in_tool\",\"arguments\":{{\"bbox\":{json.dumps(bbox)},\"reason\":{json.dumps(zoom_reason)}}}}}</tool_call>"
    )

    tool_1 = {
        "ok": True,
        "crop_path": f".temp/crops/{item_id}_crop1.png",
        "bbox": bbox,
        "note": "placeholder crop response for SFT",
    }

    msg_assistant_2 = (
        "[MEMORY_PATCH]\\n"
        f"step2_zoomcrop: {step2}\\n"
        f"candidate_answer: {cand}\\n"
        f"candidate_confidence: {cconf:.2f}\\n"
        "[/MEMORY_PATCH]\\n"
        "<tool_call>{\"name\":\"self_check_tool\",\"arguments\":{"
        f"\"stage1_conclusion\":{json.dumps(step1)},"
        f"\"zoom_evidence\":{json.dumps(step2)},"
        f"\"candidate_answer\":{json.dumps(cand)},"
        f"\"candidate_confidence\":{cconf:.2f}"
        "}}</tool_call>"
    )

    tool_2 = {
        "consistency": True,
        "revised_answer": revised,
        "note": note,
    }

    msg_assistant_3 = (
        "[MEMORY_PATCH]\\n"
        f"step3_final: {step3}\\n"
        "[/MEMORY_PATCH]\\n"
        f"Final decision: {'no anomaly' if final_label == 'A' else 'anomaly exists'}.\\n"
        f"<answer>{final_label}</answer>"
    )

    return {
        "id": item_id,
        "source_dataset": dataset,
        "image": abs_path,
        "task_type": "anomaly_detection_mcq",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT},
            {"role": "assistant", "content": msg_assistant_1},
            {"role": "tool", "name": "image_zoom_in_tool", "content": json.dumps(tool_1, ensure_ascii=False)},
            {"role": "assistant", "content": msg_assistant_2},
            {"role": "tool", "name": "self_check_tool", "content": json.dumps(tool_2, ensure_ascii=False)},
            {"role": "assistant", "content": msg_assistant_3},
        ],
        "label": final_label,
        "metadata": {
            "product": product,
            "defect_type": defect_type,
            "normal_flag": "good",
            "sft_weight": 1.0,
        },
        "train_targets": {
            "learn_roles": ["assistant"],
            "mask_roles": ["system", "user", "tool"],
            "priority": ["final_answer", "self_check_revision", "memory_patches"],
        },
    }


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    key = read_key(args.api_key_file)

    mpdd_rows = load_tsv(args.mpdd_tsv)
    visa_rows = load_tsv(args.visa_tsv)
    rnd = random.Random(args.seed)

    if len(mpdd_rows) < args.n_mpdd:
        raise RuntimeError(f"MPDD rows not enough: need {args.n_mpdd}, got {len(mpdd_rows)}")
    if len(visa_rows) < args.n_visa:
        raise RuntimeError(f"VisA rows not enough: need {args.n_visa}, got {len(visa_rows)}")

    sel_mpdd = rnd.sample(mpdd_rows, args.n_mpdd)
    sel_visa = rnd.sample(visa_rows, args.n_visa)
    selected = sel_mpdd + sel_visa
    rnd.shuffle(selected)

    selected_tsv = out_dir / "selected_200_800.tsv"
    with selected_tsv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "id", "rel_path", "abs_path"], delimiter="\t")
        w.writeheader()
        w.writerows(selected)

    def one(i: int, row: Dict[str, str]) -> Tuple[int, Optional[Dict[str, Any]], Optional[str]]:
        rel_path = row["rel_path"]
        abs_path = row["abs_path"]
        product, defect_type = extract_defect_and_product(rel_path)
        expected = expected_label_from_rel(rel_path)

        try:
            image_data = encode_image_data_url(abs_path)
            prompt = build_generation_prompt(rel_path, expected, product, defect_type)
            content = [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_data}},
            ]
            raw = call_api(
                api_url=args.api_url,
                api_key=key,
                model=args.model,
                content=content,
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens,
                timeout=args.timeout,
                max_retries=args.max_retries,
                retry_sleep=args.retry_sleep,
            )
            obj = safe_json_extract(raw)
            if obj is None:
                return i, None, f"invalid_json_response: {raw[:200]}"
            item = make_sft_item(i, row, obj)
            return i, item, None
        except Exception as e:
            return i, None, str(e)

    results: List[Optional[Dict[str, Any]]] = [None] * len(selected)
    errors: List[Dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=max(1, int(args.concurrency))) as ex:
        futures = [ex.submit(one, i, row) for i, row in enumerate(selected)]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Building SFT"):
            i, item, err = fut.result()
            if item is not None:
                results[i] = item
            else:
                errors.append({"index": i, "row": selected[i], "error": err})

    data = [x for x in results if x is not None]

    out_json = out_dir / "MemRefine_sft.json"
    out_jsonl = out_dir / "MemRefine_sft.jsonl"
    err_json = out_dir / "build_errors.json"
    info_json = out_dir / "dataset_info.json"

    out_json.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    with out_jsonl.open("w", encoding="utf-8") as f:
        for x in data:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")

    err_json.write_text(json.dumps(errors, ensure_ascii=False, indent=2), encoding="utf-8")

    dataset_info = {
        "MemRefine_sft": {
            "file_name": "MemRefine_sft.json",
            "formatting": "sharegpt",
            "columns": {
                "messages": "messages",
                "images": "image",
            },
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant",
                "observation_tag": "tool",
                "function_tag": "function_call",
                "system_tag": "system",
            },
        }
    }
    info_json.write_text(json.dumps(dataset_info, ensure_ascii=False, indent=2), encoding="utf-8")

    stats = {
        "total_selected": len(selected),
        "success": len(data),
        "failed": len(errors),
        "output_json": str(out_json),
        "output_jsonl": str(out_jsonl),
        "dataset_info": str(info_json),
        "error_file": str(err_json),
        "selected_tsv": str(selected_tsv),
    }
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
