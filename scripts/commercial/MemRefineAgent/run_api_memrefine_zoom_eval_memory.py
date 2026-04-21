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
    p = argparse.ArgumentParser(description="IAD-R1 eval via OpenAI-compatible API with Zoom + Working Memory.")
    p.add_argument("--api_url", type=str, default="https://api.vectorengine.ai/v1/chat/completions")
    p.add_argument("--api_key_file", type=str, default="/mnt/nfs/wmd/research/MemRefine-IAD/.dashscope_key")
    p.add_argument("--model", type=str, required=True)

    p.add_argument("--name", type=str, default="agent_zoom_mem_api")
    p.add_argument("--test_dataset", type=str, default="test_MVTec")
    p.add_argument("--run_all_benches", action="store_true")
    p.add_argument("--few_shot_model", type=int, default=0)
    p.add_argument("--similar_template", action="store_true")
    p.add_argument("--reproduce", action="store_true")

    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=0.8)
    p.add_argument("--max_tokens", type=int, default=512)
    p.add_argument("--timeout", type=int, default=120)
    p.add_argument("--max_retries", type=int, default=5)
    p.add_argument("--retry_sleep", type=float, default=2.0)

    p.add_argument("--data_path", type=str, default="/mnt/nfs/wmd/data/Industrial_test_real")
    p.add_argument("--test_json_root", type=str, default="/mnt/nfs/wmd/research/IAD-R1-main/data/Test")
    p.add_argument("--result_root", type=str, default="/mnt/nfs/wmd/research/IAD-R1-main/result")
    p.add_argument("--summary_py", type=str, default="/mnt/nfs/wmd/research/IAD-R1-main/helper/summary.py")
    p.add_argument("--normal_flag", type=str, default="good")

    p.add_argument("--zoom_conf_low", type=float, default=0.35)
    p.add_argument("--zoom_conf_high", type=float, default=0.75)
    p.add_argument("--enable_working_memory", action="store_true")
    p.add_argument("--memory_max_chars", type=int, default=240)
    p.add_argument("--enable_self_check", action="store_true")
    p.add_argument("--self_check_scope", type=str, default="zoom_only", choices=["zoom_only", "all"])
    return p.parse_args()


def read_api_key(path: str) -> str:
    key = Path(path).read_text(encoding="utf-8").strip()
    if not key:
        raise RuntimeError(f"API key is empty: {path}")
    return key


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


def _normalize_bbox(x1: int, y1: int, x2: int, y2: int, w: int, h: int) -> Tuple[int, int, int, int]:
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w - 1))
    y2 = max(0, min(y2, h - 1))
    if x2 <= x1:
        x2 = min(w - 1, x1 + max(1, w // 8))
    if y2 <= y1:
        y2 = min(h - 1, y1 + max(1, h // 8))
    return x1, y1, x2, y2


def center_bbox(w: int, h: int, ratio: float = 0.5) -> Tuple[int, int, int, int]:
    cw, ch = int(w * ratio), int(h * ratio)
    x1 = (w - cw) // 2
    y1 = (h - ch) // 2
    x2 = x1 + cw
    y2 = y1 + ch
    return _normalize_bbox(x1, y1, x2, y2, w, h)


def parse_zoom_decision(text: str, w: int, h: int) -> Tuple[Optional[str], Optional[float], bool, Tuple[int, int, int, int]]:
    low = (text or "").lower()
    answer = None
    if re.search(r"\byes\b", low):
        answer = "A"
    elif re.search(r"\bno\b", low):
        answer = "B"

    conf = None
    m_conf = re.search(r"confidence\s*[:=]?\s*([0-9]+(?:\.[0-9]+)?)", low)
    if m_conf:
        try:
            conf = float(m_conf.group(1))
            if conf > 1:
                conf = conf / 100.0
            if conf < 0 or conf > 1:
                conf = None
        except Exception:
            conf = None

    need_zoom = bool(re.search(r"need_zoom\s*[:=]?\s*(true|yes|1)", low))
    bbox = center_bbox(w, h, ratio=0.5)
    m_bbox = re.search(r"bbox\s*[:=]?\s*\[\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*\]", low)
    if m_bbox:
        x1, y1, x2, y2 = [int(m_bbox.group(i)) for i in range(1, 5)]
        bbox = _normalize_bbox(x1, y1, x2, y2, w, h)
    return answer, conf, need_zoom, bbox


def _map_semantic_yes_no_to_option(options: Dict[str, str], semantic: str) -> Optional[str]:
    target = semantic.strip().lower()
    for k, v in (options or {}).items():
        txt = str(v).strip().lower().strip(".")
        if txt == target:
            return str(k).upper()
    return None


def extract_answer_letter(response_text: str, options: Dict[str, str]) -> str:
    low = (response_text or "").lower()
    m_ans_tag = re.search(r"<answer>(.*?)</answer>", response_text or "", re.IGNORECASE | re.DOTALL)
    if m_ans_tag:
        low = m_ans_tag.group(1).strip().lower()

    if re.search(r"\byes\b", low):
        mapped = _map_semantic_yes_no_to_option(options, "yes")
        if mapped:
            return mapped
    if re.search(r"\bno\b", low):
        mapped = _map_semantic_yes_no_to_option(options, "no")
        if mapped:
            return mapped

    if re.search(r"\b(a)\b", low):
        return "A"
    if re.search(r"\b(b)\b", low):
        return "B"

    for k, v in (options or {}).items():
        if str(v).lower().strip(".") in low:
            return str(k).upper()
    return "E"


def extract_semantic_yes_no(response_text: str) -> Optional[str]:
    low = (response_text or "").lower()
    if re.search(r"\byes\b", low):
        return "Yes"
    if re.search(r"\bno\b", low):
        return "No"
    return None


def _clip_text(text: str, max_chars: int) -> str:
    t = (text or "").strip().replace("\n", " ")
    return t if len(t) <= max_chars else (t[: max_chars - 3] + "...")


def build_working_memory_summary(memory: Dict[str, str]) -> str:
    if not memory:
        return "empty"
    keys = ["step1_global", "step2_zoomcrop", "step3_selfcheck"]
    lines: List[str] = []
    for k in keys:
        if memory.get(k):
            lines.append(f"{k}: {memory[k]}")
    return "\n".join(lines) if lines else "empty"


def encode_image_data_url(img: Image.Image, fmt: str = "JPEG") -> str:
    buf = BytesIO()
    if img.mode != "RGB":
        img = img.convert("RGB")
    img.save(buf, format=fmt, quality=95)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def build_stage1_text(has_fewshot: bool, few_num: int, question_text: str) -> str:
    prefix = ""
    if has_fewshot:
        prefix = (
            f"You will receive {few_num} normal reference image(s) first, then 1 test image. "
            "Compare carefully.\n"
        )
    return (
        prefix
        + f"Question: {question_text}\n"
        + "Return plain JSON with keys: answer(Yes/No), confidence(0~1), need_zoom(true/false), "
        + "bbox([x1,y1,x2,y2] in image pixels), reason."
    )


def build_stage2_text(has_fewshot: bool, few_num: int, question_text: str, memory_summary: str = "empty") -> str:
    prefix = ""
    if has_fewshot:
        prefix = f"You will receive {few_num} normal references, then full test image, then zoom-crop image.\n"
    return (
        prefix
        + "Use full image + zoom crop to refine your decision.\n"
        + f"Working memory:\n{memory_summary}\n"
        + f"Question: {question_text}\n"
        + "Return plain JSON with keys: answer(Yes/No), confidence(0~1), need_zoom(true/false), "
        + "bbox([x1,y1,x2,y2]), reason."
    )


def build_self_check_text(question_text: str, stage1_text: str, stage2_text: str, used_zoom: bool, memory_summary: str = "empty") -> str:
    return (
        "You are a strict industrial inspection self-checker.\n"
        f"Question: {question_text}\n"
        f"Used zoom: {used_zoom}\n"
        f"Stage1 output: {stage1_text}\n"
        f"Current output: {stage2_text}\n"
        f"Working memory:\n{memory_summary}\n"
        "Check consistency of defect existence and evidence. "
        "Return concise final answer with explicit Yes/No."
    )


def call_chat_api(
    api_url: str,
    api_key: str,
    model: str,
    text: str,
    images: List[Image.Image],
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

    content: List[Dict[str, Any]] = [{"type": "text", "text": text}]
    for im in images:
        content.append({"type": "image_url", "image_url": {"url": encode_image_data_url(im)}})

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
                raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:800]}")
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            last_err = e
            if i < max_retries - 1:
                time.sleep(retry_sleep * (2 ** i))
    raise RuntimeError(f"API call failed after retries: {last_err}")


def run_batch_samples(
    api_url: str,
    api_key: str,
    model: str,
    batch_items: List[Dict[str, Any]],
    concurrency: int,
    temperature: float,
    top_p: float,
    max_tokens: int,
    timeout: int,
    max_retries: int,
    retry_sleep: float,
    zoom_conf_low: float,
    zoom_conf_high: float,
    enable_working_memory: bool,
    memory_max_chars: int,
    enable_self_check: bool,
    self_check_scope: str,
) -> List[Tuple[str, Dict[str, Any]]]:
    results: List[Tuple[str, Dict[str, Any]]] = [("E", {"error": "init"}) for _ in batch_items]

    ctx: List[Dict[str, Any]] = []
    for i, item in enumerate(batch_items):
        q_text = str(item["question_obj"].get("text", "")).strip()
        opts = item["question_obj"].get("options", {}) or {}

        try:
            with Image.open(item["image_path"]) as im:
                im = im.convert("RGB")
                w, h = im.size
                image_main = im.copy()
        except Exception as e:
            results[i] = ("E", {"error": f"open_image_failed: {e}"})
            continue

        few_imgs: List[Image.Image] = []
        for p in item.get("few_shot_paths", []):
            try:
                few_imgs.append(Image.open(p).convert("RGB"))
            except Exception:
                continue

        ctx.append(
            {
                "idx": i,
                "opts": opts,
                "w": w,
                "h": h,
                "image_main": image_main,
                "few_imgs": few_imgs,
                "q_text": q_text,
                "stage1_response": "",
                "final_response": "",
                "bbox": center_bbox(w, h),
                "conf1": None,
                "need_zoom1": False,
                "used_zoom": False,
                "working_memory": {},
            }
        )

    # Stage 1 in parallel
    def _stage1_one(c: Dict[str, Any]) -> Tuple[int, str, Optional[str]]:
        text = build_stage1_text(bool(c["few_imgs"]), len(c["few_imgs"]), c["q_text"])
        imgs = c["few_imgs"] + [c["image_main"]]
        try:
            out = call_chat_api(api_url, api_key, model, text, imgs, temperature, top_p, max_tokens, timeout, max_retries, retry_sleep)
            return c["idx"], out, None
        except Exception as e:
            return c["idx"], "", str(e)

    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as ex:
        futs = [ex.submit(_stage1_one, c) for c in ctx]
        for f in as_completed(futs):
            idx, out_text, err = f.result()
            c = next(x for x in ctx if x["idx"] == idx)
            if err:
                results[idx] = ("E", {"error": f"stage1_api_failed: {err}"})
                continue
            c["stage1_response"] = out_text
            if enable_working_memory:
                c["working_memory"]["step1_global"] = _clip_text(out_text, memory_max_chars)

            _, conf1, need_zoom1, bbox = parse_zoom_decision(out_text, c["w"], c["h"])
            c["conf1"] = conf1
            c["need_zoom1"] = need_zoom1
            c["bbox"] = bbox
            should_zoom = bool(need_zoom1 or (conf1 is not None and zoom_conf_low <= conf1 <= zoom_conf_high))
            if not should_zoom:
                c["final_response"] = out_text
            else:
                c["used_zoom"] = True

    # Stage 2 for zoom-needed samples
    zoom_ctx = [c for c in ctx if c.get("used_zoom")]

    def _stage2_one(c: Dict[str, Any]) -> Tuple[int, str, Optional[str]]:
        x1, y1, x2, y2 = c["bbox"]
        crop = c["image_main"].crop((x1, y1, x2, y2))
        text = build_stage2_text(
            bool(c["few_imgs"]),
            len(c["few_imgs"]),
            c["q_text"],
            memory_summary=build_working_memory_summary(c["working_memory"]) if enable_working_memory else "empty",
        )
        imgs = c["few_imgs"] + [c["image_main"], crop]
        try:
            out = call_chat_api(api_url, api_key, model, text, imgs, temperature, top_p, max_tokens, timeout, max_retries, retry_sleep)
            return c["idx"], out, None
        except Exception as e:
            return c["idx"], "", str(e)

    if zoom_ctx:
        with ThreadPoolExecutor(max_workers=max(1, concurrency)) as ex:
            futs = [ex.submit(_stage2_one, c) for c in zoom_ctx]
            for f in as_completed(futs):
                idx, out_text, err = f.result()
                c = next(x for x in ctx if x["idx"] == idx)
                if err:
                    c["final_response"] = c["stage1_response"]
                    c["stage2_error"] = err
                else:
                    c["final_response"] = out_text
                    if enable_working_memory:
                        c["working_memory"]["step2_zoomcrop"] = _clip_text(out_text, memory_max_chars)

    # Optional self-check
    if enable_self_check:
        sc_ctx = [c for c in ctx if (self_check_scope == "all" or c.get("used_zoom"))]

        def _selfcheck_one(c: Dict[str, Any]) -> Tuple[int, str, Optional[str]]:
            stage2_text = c["final_response"] or c["stage1_response"]
            text = build_self_check_text(
                c["q_text"],
                c["stage1_response"],
                stage2_text,
                bool(c["used_zoom"]),
                memory_summary=build_working_memory_summary(c["working_memory"]) if enable_working_memory else "empty",
            )
            imgs = [c["image_main"]]
            if c["used_zoom"]:
                x1, y1, x2, y2 = c["bbox"]
                imgs.append(c["image_main"].crop((x1, y1, x2, y2)))
            try:
                out = call_chat_api(api_url, api_key, model, text, imgs, temperature, top_p, max_tokens, timeout, max_retries, retry_sleep)
                return c["idx"], out, None
            except Exception as e:
                return c["idx"], "", str(e)

        with ThreadPoolExecutor(max_workers=max(1, concurrency)) as ex:
            futs = [ex.submit(_selfcheck_one, c) for c in sc_ctx]
            for f in as_completed(futs):
                idx, out_text, err = f.result()
                c = next(x for x in ctx if x["idx"] == idx)
                if err:
                    c["self_check_error"] = err
                    continue
                c["self_check_response"] = out_text
                if enable_working_memory:
                    c["working_memory"]["step3_selfcheck"] = _clip_text(out_text, memory_max_chars)
                semantic = extract_semantic_yes_no(out_text)
                if semantic is not None:
                    c["final_response"] = semantic
                    c["self_check_applied"] = True
                else:
                    c["self_check_applied"] = False

    for c in ctx:
        if results[c["idx"]][0] == "E" and "stage1_api_failed" in results[c["idx"]][1].get("error", ""):
            continue
        final_response = c["final_response"] or c["stage1_response"]
        pred = extract_answer_letter(final_response, c["opts"])
        trace = {
            "stage1_response": c["stage1_response"],
            "final_response": final_response,
            "used_zoom": c["used_zoom"],
            "bbox": list(c["bbox"]),
            "conf1": c["conf1"],
            "need_zoom1": c["need_zoom1"],
            "working_memory_enabled": bool(enable_working_memory),
            "self_check_enabled": bool(enable_self_check),
            "self_check_scope": self_check_scope if enable_self_check else "disabled",
            "self_check_applied": bool(c.get("self_check_applied", False)),
        }
        if enable_working_memory:
            trace["working_memory"] = c.get("working_memory", {})
        if "self_check_response" in c:
            trace["self_check_response"] = c["self_check_response"]
        if "stage2_error" in c:
            trace["stage2_error"] = c["stage2_error"]
        if "self_check_error" in c:
            trace["self_check_error"] = c["self_check_error"]
        results[c["idx"]] = (pred, trace)

    return results


def eval_one_bench(args: argparse.Namespace, api_key: str, bench: str) -> None:
    model_name = args.model.replace("/", "_")
    result_dir = Path(args.result_root) / args.name / bench
    result_dir.mkdir(parents=True, exist_ok=True)
    answers_json_path = result_dir / f"answers_{args.few_shot_model}_shot_{model_name}_agentzoom_api.json"
    trace_jsonl_path = result_dir / f"trace_{args.few_shot_model}_shot_{model_name}_agentzoom_api.jsonl"

    if answers_json_path.exists() and not args.reproduce:
        all_answers = json.load(answers_json_path.open("r", encoding="utf-8"))
    else:
        all_answers = []

    done_images = set(a.get("image") for a in all_answers)

    test_json = Path(args.test_json_root) / f"{bench}_format.json"
    chat_ad = json.load(test_json.open("r", encoding="utf-8"))

    # Support both formats:
    # 1) list[dict] where each item contains "image"
    # 2) dict[str, dict] where key is rel image path
    sample_items: List[Tuple[str, Dict[str, Any]]] = []
    if isinstance(chat_ad, dict):
        for k, v in chat_ad.items():
            if isinstance(v, dict):
                sample_items.append((str(k).strip(), v))
    elif isinstance(chat_ad, list):
        for obj in chat_ad:
            if isinstance(obj, dict):
                rel = str(obj.get("image", "")).strip()
                if rel:
                    sample_items.append((rel, obj))

    pending: List[Dict[str, Any]] = []
    for rel_path, obj in sample_items:
        if not rel_path or rel_path in done_images:
            continue

        conv = obj.get("conversation", []) or []
        questions, answers, qtypes = parse_conversation(conv)
        if not questions or not answers:
            continue

        q = questions[0]
        a = answers[0]
        qt = qtypes[0] if qtypes else "Anomaly Detection"

        few = obj.get("similar_templates" if args.similar_template else "random_templates", []) or []
        few = few[: max(0, args.few_shot_model)]
        few_abs = [str(Path(args.data_path) / p) for p in few]

        image_abs = str(Path(args.data_path) / rel_path)
        if not Path(image_abs).is_file():
            continue

        pending.append(
            {
                "rel_path": rel_path,
                "image_path": image_abs,
                "few_shot_paths": few_abs,
                "question_obj": q,
                "correct_answer": a,
                "question_type": qt,
            }
        )

    print(f"[INFO] bench={bench} total_pending={len(pending)} concurrency={args.concurrency} self_check={args.enable_self_check}")

    with trace_jsonl_path.open("w", encoding="utf-8") as tf:
        for i in tqdm(range(0, len(pending), max(1, args.concurrency)), desc=f"Evaluating {bench}"):
            batch_items = pending[i : i + max(1, args.concurrency)]
            batch_results = run_batch_samples(
                api_url=args.api_url,
                api_key=api_key,
                model=args.model,
                batch_items=batch_items,
                concurrency=args.concurrency,
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens,
                timeout=args.timeout,
                max_retries=args.max_retries,
                retry_sleep=args.retry_sleep,
                zoom_conf_low=args.zoom_conf_low,
                zoom_conf_high=args.zoom_conf_high,
                enable_working_memory=args.enable_working_memory,
                memory_max_chars=args.memory_max_chars,
                enable_self_check=args.enable_self_check,
                self_check_scope=args.self_check_scope,
            )

            for item, (pred, trace) in zip(batch_items, batch_results):
                all_answers.append(
                    {
                        "image": item["rel_path"],
                        "question": item["question_obj"],
                        "question_type": item["question_type"],
                        "correct_answer": item["correct_answer"],
                        "gpt_answer": pred,
                    }
                )
                tf.write(json.dumps({"image": item["rel_path"], "trace": trace}, ensure_ascii=False) + "\n")

            with answers_json_path.open("w", encoding="utf-8") as wf:
                json.dump(all_answers, wf, ensure_ascii=False, indent=2)

    # summary
    summary_cmd = [
        os.environ.get("PYTHON_BIN", os.sys.executable),
        args.summary_py,
        "--answers_json_path",
        str(answers_json_path),
        "--normal_flag",
        args.normal_flag,
    ]
    try:
        proc = subprocess.run(summary_cmd, capture_output=True, text=True)
        print(f"[INFO] bench={bench} summary_returncode={proc.returncode}")
        if proc.stdout.strip():
            print(proc.stdout.strip().splitlines()[-5:])
        if proc.stderr.strip():
            print("[WARN] summary_stderr_tail:", "\\n".join(proc.stderr.strip().splitlines()[-5:]))
    except Exception as e:
        print(f"[WARN] bench={bench} summary failed: {e}")


def main() -> None:
    args = parse_args()
    api_key = read_api_key(args.api_key_file)

    benches = BENCHES if args.run_all_benches else [args.test_dataset]
    for b in benches:
        eval_one_bench(args, api_key, b)


if __name__ == "__main__":
    main()
