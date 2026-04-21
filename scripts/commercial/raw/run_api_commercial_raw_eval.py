import argparse
import base64
import json
import os
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from PIL import Image
from tqdm import tqdm

BENCHES = ["test_DAGM", "test_MVTec", "test_DTD", "test_MPDD", "test_SDD", "test_VisA"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MemRefine-IAD commercial raw inference (no agent logic).")
    p.add_argument("--api_url", type=str, default="https://api.vectorengine.ai/v1/chat/completions")
    p.add_argument("--api_key_file", type=str, default="/mnt/nfs/wmd/research/MemRefine-IAD/.dashscope_key")
    p.add_argument("--model", type=str, required=True)

    p.add_argument("--name", type=str, default="memrefine_raw_api")
    p.add_argument("--test_dataset", type=str, default="test_MVTec")
    p.add_argument("--run_all_benches", action="store_true")
    p.add_argument("--few_shot_model", type=int, default=0)
    p.add_argument("--similar_template", action="store_true")
    p.add_argument("--reproduce", action="store_true")

    p.add_argument("--data_path", type=str, default="/mnt/nfs/wmd/data/Industrial_test_real")
    p.add_argument("--test_json_root", type=str, default="/mnt/nfs/wmd/research/IAD-R1-main/data/Test")
    p.add_argument("--result_root", type=str, default="/mnt/nfs/wmd/research/IAD-R1-main/result")
    p.add_argument("--summary_py", type=str, default="/mnt/nfs/wmd/research/IAD-R1-main/helper/summary.py")
    p.add_argument("--normal_flag", type=str, default="good")

    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--max_tokens", type=int, default=64)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--timeout", type=int, default=120)
    p.add_argument("--max_retries", type=int, default=5)
    p.add_argument("--retry_sleep", type=float, default=2.0)
    return p.parse_args()


def read_api_key(path: str) -> str:
    key = Path(path).read_text(encoding="utf-8").strip()
    if not key:
        raise RuntimeError(f"API key empty: {path}")
    return key


def encode_image_data_url(image_path: str) -> str:
    with Image.open(image_path) as im:
        if im.mode != "RGB":
            im = im.convert("RGB")
        max_side = 1024
        w, h = im.size
        scale = min(1.0, float(max_side) / max(w, h))
        if scale < 1.0:
            im = im.resize((int(w * scale), int(h * scale)))
        buf = BytesIO()
        im.save(buf, format="JPEG", quality=95)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def parse_conversation(conv: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[str], List[str]]:
    questions: List[Dict[str, Any]] = []
    answers: List[str] = []
    qtypes: List[str] = []
    for t in conv:
        if not isinstance(t, dict):
            continue
        q = str(t.get("Question", "")).strip()
        a = str(t.get("Answer", "")).strip()
        opts = t.get("Options", {}) or {}
        qt = str(t.get("type", "")).strip()
        if q:
            questions.append({"text": q, "options": opts})
            answers.append(a)
            qtypes.append(qt)
    return questions, answers, qtypes


def build_prompt(question_text: str, options: Dict[str, str], few_shot_num: int) -> str:
    option_lines = "\n".join([f"{k}: {v}" for k, v in options.items()])
    fs_hint = ""
    if few_shot_num > 0:
        fs_hint = f"You are also given {few_shot_num} normal reference image(s) before the test image. Use them as comparison templates.\n"
    return (
        f"{fs_hint}"
        "You are an industrial anomaly inspector.\n"
        f"Question: {question_text}\n"
        "Options:\n"
        f"{option_lines}\n"
        "Return ONLY the option key (one of A/B/C/D/E if provided). No explanation."
    )


def map_answer_to_option(response_text: str, options: Dict[str, str]) -> str:
    if not response_text:
        return "E"

    text = str(response_text).strip()
    upper = text.upper()

    # 1) exact single-letter key
    if len(upper) == 1 and upper in options:
        return upper

    # 2) anywhere contains a standalone key letter
    for k in options.keys():
        if re.search(rf"\b{re.escape(k.upper())}\b", upper):
            return k.upper()

    low = text.lower().strip().strip(".").strip("!")

    # 3) exact match option text
    for k, v in options.items():
        if low == str(v).lower().strip().strip(".").strip("!"):
            return k.upper()

    # 4) yes/no semantic mapping for binary options
    if re.search(r"\byes\b", low):
        for k, v in options.items():
            if str(v).lower().strip().strip(".") == "yes":
                return k.upper()
    if re.search(r"\bno\b", low):
        for k, v in options.items():
            if str(v).lower().strip().strip(".") == "no":
                return k.upper()

    # 5) fuzzy contains
    for k, v in options.items():
        ov = str(v).lower().strip().strip(".")
        if ov and (ov in low or low in ov):
            return k.upper()

    return "E"


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

    last_err: Optional[Exception] = None
    for i in range(max_retries):
        try:
            resp = requests.post(api_url, headers=headers, json=payload, timeout=timeout)
            if resp.status_code >= 400:
                raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:600]}")
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            last_err = e
            if i < max_retries - 1:
                time.sleep(retry_sleep * (2 ** i))
    raise RuntimeError(f"API failed after retries: {last_err}")


def iter_samples(chat_ad: Any) -> List[Tuple[str, Dict[str, Any]]]:
    items: List[Tuple[str, Dict[str, Any]]] = []
    if isinstance(chat_ad, dict):
        for k, v in chat_ad.items():
            if isinstance(v, dict):
                items.append((str(k).strip(), v))
    elif isinstance(chat_ad, list):
        for obj in chat_ad:
            if isinstance(obj, dict):
                rel = str(obj.get("image", "")).strip()
                if rel:
                    items.append((rel, obj))
    return items


def eval_one_bench(args: argparse.Namespace, api_key: str, bench: str) -> None:
    model_name = args.model.replace("/", "_")
    result_dir = Path(args.result_root) / args.name / bench
    result_dir.mkdir(parents=True, exist_ok=True)

    answers_json_path = result_dir / f"answers_{args.few_shot_model}_shot_{model_name}.json"
    trace_jsonl_path = result_dir / f"trace_{args.few_shot_model}_shot_{model_name}.jsonl"

    if answers_json_path.exists() and not args.reproduce:
        all_answers = json.load(answers_json_path.open("r", encoding="utf-8"))
    else:
        all_answers = []

    existing_images = set(a.get("image") for a in all_answers)

    test_json = Path(args.test_json_root) / f"{bench}_format.json"
    chat_ad = json.load(test_json.open("r", encoding="utf-8"))

    items = iter_samples(chat_ad)
    pending = [(rel_path, obj) for rel_path, obj in items if rel_path and rel_path not in existing_images]

    print(
        f"[INFO] bench={bench} total_pending={len(pending)} model={args.model} "
        f"concurrency={max(1, int(args.concurrency))}"
    )

    def process_one(rel_path: str, text_gt: Dict[str, Any]) -> Optional[Tuple[Dict[str, Any], Dict[str, Any]]]:
        conv = text_gt.get("conversation", []) or []
        questions, answers, qtypes = parse_conversation(conv)
        if not questions or not answers:
            return None

        q = questions[0]
        a = answers[0]
        qt = qtypes[0] if qtypes else "Anomaly Detection"
        options = q.get("options", {}) or {}

        few = text_gt.get("similar_templates" if args.similar_template else "random_templates", []) or []
        few = few[: max(0, args.few_shot_model)]

        image_abs = str(Path(args.data_path) / rel_path)
        if not Path(image_abs).is_file():
            return None

        few_abs = [str(Path(args.data_path) / p) for p in few]
        content: List[Dict[str, Any]] = []

        for f in few_abs:
            if Path(f).is_file():
                content.append({"type": "image_url", "image_url": {"url": encode_image_data_url(f)}})

        content.append({"type": "image_url", "image_url": {"url": encode_image_data_url(image_abs)}})
        content.append({"type": "text", "text": build_prompt(q.get("text", ""), options, len(few_abs))})

        raw_answer = ""
        err_msg = None
        try:
            raw_answer = call_api(
                api_url=args.api_url,
                api_key=api_key,
                model=args.model,
                content=content,
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens,
                timeout=args.timeout,
                max_retries=args.max_retries,
                retry_sleep=args.retry_sleep,
            )
        except Exception as e:
            err_msg = str(e)

        pred = map_answer_to_option(raw_answer, options) if raw_answer else "E"

        answer_entry = {
            "image": rel_path,
            "question": q,
            "question_type": qt,
            "correct_answer": a,
            "gpt_answer": pred,
        }
        trace = {
            "image": rel_path,
            "raw_answer": raw_answer,
            "mapped_answer": pred,
            "options": options,
        }
        if err_msg is not None:
            trace["error"] = err_msg
        return answer_entry, trace

    with trace_jsonl_path.open("w", encoding="utf-8") as tf:
        with ThreadPoolExecutor(max_workers=max(1, int(args.concurrency))) as ex:
            fut_map = {ex.submit(process_one, rel_path, text_gt): rel_path for rel_path, text_gt in pending}
            for fut in tqdm(as_completed(fut_map), total=len(fut_map), desc=f"Evaluating {bench}"):
                ret = fut.result()
                if ret is None:
                    continue
                answer_entry, trace = ret
                all_answers.append(answer_entry)
                tf.write(json.dumps(trace, ensure_ascii=False) + "\n")
                answers_json_path.write_text(json.dumps(all_answers, ensure_ascii=False, indent=2), encoding="utf-8")

    # summary
    summary_cmd = [
        os.environ.get("PYTHON_BIN", os.sys.executable),
        args.summary_py,
        "--answers_json_path",
        str(answers_json_path),
        "--normal_flag",
        args.normal_flag,
    ]
    proc = subprocess.run(summary_cmd, capture_output=True, text=True)
    print(f"[INFO] bench={bench} summary_returncode={proc.returncode}")
    if proc.stdout.strip():
        print("\n".join(proc.stdout.strip().splitlines()[-6:]))
    if proc.stderr.strip():
        print("[WARN] summary stderr tail:")
        print("\n".join(proc.stderr.strip().splitlines()[-6:]))


def main() -> None:
    args = parse_args()
    api_key = read_api_key(args.api_key_file)

    benches = BENCHES if args.run_all_benches else [args.test_dataset]
    for bench in benches:
        eval_one_bench(args, api_key, bench)


if __name__ == "__main__":
    main()
