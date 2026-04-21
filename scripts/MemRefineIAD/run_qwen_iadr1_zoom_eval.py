import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="IAD-R1-style eval with built-in Zoom tool (no RAG).")
    p.add_argument("--model-path", type=str, required=True)
    p.add_argument("--name", type=str, default="agent_zoom_qwen7b")
    p.add_argument("--test_dataset", type=str, default="test_MVTec")
    p.add_argument("--few_shot_model", type=int, default=0)
    p.add_argument("--similar_template", action="store_true")
    p.add_argument("--reproduce", action="store_true")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--concurrency", type=int, default=0, help="Effective inference batch size. If >0, overrides --batch_size.")
    p.add_argument("--tensor_parallel_size", type=int, default=1)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=0.8)
    p.add_argument("--max_tokens", type=int, default=512)
    p.add_argument("--data_path", type=str, default="/mnt/nfs/wmd/data/Industrial_test_real")
    p.add_argument("--test_json_root", type=str, default="/mnt/nfs/wmd/research/IAD-R1-main/data/Test")
    p.add_argument("--result_root", type=str, default="/mnt/nfs/wmd/research/IAD-R1-main/result")
    p.add_argument("--summary_py", type=str, default="/mnt/nfs/wmd/research/IAD-R1-main/helper/summary.py")
    p.add_argument("--normal_flag", type=str, default="good")
    p.add_argument("--zoom_conf_low", type=float, default=0.35)
    p.add_argument("--zoom_conf_high", type=float, default=0.75)
    p.add_argument("--enable_self_check", action="store_true", help="Enable Self-Check tool stage.")
    p.add_argument(
        "--self_check_scope",
        type=str,
        default="zoom_only",
        choices=["zoom_only", "all"],
        help="Apply Self-Check on zoomed samples only or all samples.",
    )
    return p.parse_args()


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

    need_zoom = False
    if re.search(r"need_zoom\s*[:=]?\s*(true|yes|1)", low):
        need_zoom = True

    bbox = center_bbox(w, h, ratio=0.5)
    m_bbox = re.search(
        r"bbox\s*[:=]?\s*\[\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*\]",
        low,
    )
    if m_bbox:
        x1, y1, x2, y2 = [int(m_bbox.group(i)) for i in range(1, 5)]
        bbox = _normalize_bbox(x1, y1, x2, y2, w, h)

    return answer, conf, need_zoom, bbox


def _map_semantic_yes_no_to_option(options: Dict[str, str], semantic: str) -> Optional[str]:
    """
    semantic: 'yes' or 'no'
    Map semantic answer to current sample's A/B option text.
    """
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

    # Prefer semantic yes/no and map via per-sample options (A/B may be flipped).
    if re.search(r"\byes\b", low):
        mapped = _map_semantic_yes_no_to_option(options, "yes")
        if mapped:
            return mapped
    if re.search(r"\bno\b", low):
        mapped = _map_semantic_yes_no_to_option(options, "no")
        if mapped:
            return mapped

    # Fallback: model explicitly outputs option letter.
    if re.search(r"\b(a)\b", low):
        return "A"
    if re.search(r"\b(b)\b", low):
        return "B"

    # Fallback: fuzzy against option text
    for k, v in (options or {}).items():
        if str(v).lower().strip(".") in low:
            return str(k).upper()
    return "E"


def build_stage1_prompt(has_fewshot: bool, few_num: int, question_text: str) -> List[Dict[str, Any]]:
    content: List[Dict[str, Any]] = []
    if has_fewshot:
        content.append(
            {
                "type": "text",
                "text": (
                    f"Following are {few_num} normal reference image(s) for comparison. "
                    "Then a test image is provided."
                ),
            }
        )
        for _ in range(few_num):
            content.append({"type": "image"})
        content.append({"type": "text", "text": "Now the test image:"})

    content.append({"type": "image"})
    content.append(
        {
            "type": "text",
            "text": (
                f"{question_text}\n"
                "Return plain JSON with keys: answer(Yes/No), confidence(0~1), "
                "need_zoom(true/false), bbox([x1,y1,x2,y2] in image pixels), reason."
            ),
        }
    )
    return [{"role": "user", "content": content}]


def build_stage2_prompt(has_fewshot: bool, few_num: int, question_text: str) -> List[Dict[str, Any]]:
    content: List[Dict[str, Any]] = []
    if has_fewshot:
        content.append(
            {
                "type": "text",
                "text": (
                    f"Following are {few_num} normal reference image(s) for comparison. "
                    "Then the test full image and zoomed crop are provided."
                ),
            }
        )
        for _ in range(few_num):
            content.append({"type": "image"})

    content.append({"type": "text", "text": "Test full image:"})
    content.append({"type": "image"})
    content.append({"type": "text", "text": "Zoomed crop of suspicious region:"})
    content.append({"type": "image"})
    content.append(
        {
            "type": "text",
            "text": (
                f"{question_text}\n"
                "Answer strictly with one token: Yes or No."
            ),
        }
    )
    return [{"role": "user", "content": content}]


def build_self_check_prompt(question_text: str, stage1_text: str, stage2_text: str, used_zoom: bool) -> List[Dict[str, Any]]:
    context_text = (
        "You are a strict industrial anomaly self-check tool.\n"
        "Given image evidence and prior model reasoning, decide final answer.\n"
        "Rules:\n"
        "1) Final answer must be Yes or No only.\n"
        "2) If prior reasoning and image evidence conflict, prefer image evidence.\n"
        "3) Keep consistency between rationale and final answer.\n\n"
        f"Question: {question_text}\n"
        f"Used zoom: {used_zoom}\n"
        f"Stage1 output: {stage1_text}\n"
        f"Stage2 output: {stage2_text}\n"
    )
    content: List[Dict[str, Any]] = [
        {"type": "text", "text": "Test full image:"},
        {"type": "image"},
    ]
    if used_zoom:
        content.append({"type": "text", "text": "Zoomed suspicious crop:"})
        content.append({"type": "image"})
    content.append({"type": "text", "text": context_text + "\nReturn only: Yes or No."})
    return [{"role": "user", "content": content}]


def extract_semantic_yes_no(response_text: str) -> Optional[str]:
    low = (response_text or "").lower()
    m_ans_tag = re.search(r"<answer>(.*?)</answer>", response_text or "", re.IGNORECASE | re.DOTALL)
    if m_ans_tag:
        low = m_ans_tag.group(1).strip().lower()
    if re.search(r"\byes\b", low):
        return "yes"
    if re.search(r"\bno\b", low):
        return "no"
    return None


def run_batch_samples(
    llm: LLM,
    tokenizer: Any,
    batch_items: List[Dict[str, Any]],
    sampling_params: SamplingParams,
    zoom_conf_low: float,
    zoom_conf_high: float,
    enable_self_check: bool = False,
    self_check_scope: str = "zoom_only",
) -> List[Tuple[str, Dict[str, Any]]]:
    """
    True batched two-stage inference:
    - Stage1 batch: global decision + whether to zoom
    - Stage2 batch: only for samples that need zoom
    """
    results: List[Tuple[str, Dict[str, Any]]] = [("E", {"error": "uninitialized"}) for _ in batch_items]
    ctx: List[Dict[str, Any]] = []
    stage1_inputs: List[Dict[str, Any]] = []

    for i, item in enumerate(batch_items):
        q = item["question_obj"]
        q_text = str(q.get("text", "Is there any defect in the object?"))
        opts = q.get("options", {"A": "Yes.", "B": "No."})

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

        messages1 = build_stage1_prompt(has_fewshot=len(few_imgs) > 0, few_num=len(few_imgs), question_text=q_text)
        prompt1 = tokenizer.apply_chat_template(messages1, tokenize=False, add_generation_prompt=True)
        stage1_inputs.append({"prompt": prompt1, "multi_modal_data": {"image": few_imgs + [image_main]}})
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
            }
        )

    if not stage1_inputs:
        return results

    try:
        outputs1 = llm.generate(stage1_inputs, sampling_params=sampling_params)
    except Exception as e:
        for c in ctx:
            results[c["idx"]] = ("E", {"error": f"stage1_generate_failed: {e}"})
        return results

    stage2_inputs: List[Dict[str, Any]] = []
    stage2_ctx_ref: List[int] = []

    for c, out in zip(ctx, outputs1):
        text1 = out.outputs[0].text if out.outputs else ""
        c["stage1_response"] = text1
        _, conf1, need_zoom1, bbox = parse_zoom_decision(text1, c["w"], c["h"])
        c["conf1"] = conf1
        c["need_zoom1"] = need_zoom1
        c["bbox"] = bbox
        should_zoom = bool(need_zoom1 or (conf1 is not None and zoom_conf_low <= conf1 <= zoom_conf_high))

        if not should_zoom:
            c["final_response"] = text1
            continue

        c["used_zoom"] = True
        x1, y1, x2, y2 = bbox
        crop = c["image_main"].crop((x1, y1, x2, y2))
        messages2 = build_stage2_prompt(
            has_fewshot=len(c["few_imgs"]) > 0,
            few_num=len(c["few_imgs"]),
            question_text=c["q_text"],
        )
        prompt2 = tokenizer.apply_chat_template(messages2, tokenize=False, add_generation_prompt=True)
        stage2_inputs.append({"prompt": prompt2, "multi_modal_data": {"image": c["few_imgs"] + [c["image_main"], crop]}})
        stage2_ctx_ref.append(c["idx"])

    if stage2_inputs:
        try:
            outputs2 = llm.generate(stage2_inputs, sampling_params=sampling_params)
            out2_map: Dict[int, str] = {}
            for idx, out in zip(stage2_ctx_ref, outputs2):
                out2_map[idx] = out.outputs[0].text if out.outputs else ""
            for c in ctx:
                if c["idx"] in out2_map:
                    c["final_response"] = out2_map[c["idx"]]
        except Exception as e:
            for c in ctx:
                if c["used_zoom"]:
                    c["final_response"] = c["stage1_response"]
                    c["stage2_error"] = str(e)

    # Optional Self-Check stage
    if enable_self_check:
        sc_inputs: List[Dict[str, Any]] = []
        sc_ctx_ref: List[int] = []
        for c in ctx:
            if self_check_scope == "zoom_only" and not c["used_zoom"]:
                continue
            stage2_text = c["final_response"] or c["stage1_response"]
            messages_sc = build_self_check_prompt(
                question_text=c["q_text"],
                stage1_text=c["stage1_response"],
                stage2_text=stage2_text,
                used_zoom=bool(c["used_zoom"]),
            )
            prompt_sc = tokenizer.apply_chat_template(messages_sc, tokenize=False, add_generation_prompt=True)
            images_sc = [c["image_main"]]
            if c["used_zoom"]:
                x1, y1, x2, y2 = c["bbox"]
                images_sc.append(c["image_main"].crop((x1, y1, x2, y2)))
            sc_inputs.append({"prompt": prompt_sc, "multi_modal_data": {"image": images_sc}})
            sc_ctx_ref.append(c["idx"])

        if sc_inputs:
            try:
                sc_outputs = llm.generate(sc_inputs, sampling_params=sampling_params)
                sc_map: Dict[int, str] = {}
                for idx, out in zip(sc_ctx_ref, sc_outputs):
                    sc_map[idx] = out.outputs[0].text if out.outputs else ""
                for c in ctx:
                    if c["idx"] in sc_map:
                        c["self_check_response"] = sc_map[c["idx"]]
                        semantic = extract_semantic_yes_no(sc_map[c["idx"]])
                        if semantic is not None:
                            # Keep downstream option mapping dynamic via extract_answer_letter.
                            c["final_response"] = semantic
                            c["self_check_applied"] = True
                        else:
                            c["self_check_applied"] = False
            except Exception as e:
                for c in ctx:
                    if c["idx"] in sc_ctx_ref:
                        c["self_check_error"] = str(e)

    for c in ctx:
        final_response = c["final_response"] or c["stage1_response"]
        pred = extract_answer_letter(final_response, c["opts"])
        trace = {
            "stage1_response": c["stage1_response"],
            "final_response": final_response,
            "used_zoom": c["used_zoom"],
            "bbox": list(c["bbox"]),
            "conf1": c["conf1"],
            "need_zoom1": c["need_zoom1"],
            "self_check_enabled": bool(enable_self_check),
            "self_check_scope": self_check_scope if enable_self_check else "disabled",
            "self_check_applied": bool(c.get("self_check_applied", False)),
        }
        if "self_check_response" in c:
            trace["self_check_response"] = c["self_check_response"]
        if "stage2_error" in c:
            trace["stage2_error"] = c["stage2_error"]
        if "self_check_error" in c:
            trace["self_check_error"] = c["self_check_error"]
        results[c["idx"]] = (pred, trace)

    return results


def main() -> None:
    args = parse_args()
    torch.manual_seed(1234)
    effective_batch = int(args.concurrency) if int(args.concurrency or 0) > 0 else int(args.batch_size)
    effective_batch = max(1, effective_batch)

    model_name = os.path.split(args.model_path.rstrip("/"))[-1]
    if args.similar_template:
        model_name = model_name + "_Similar_template"

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    llm = LLM(
        model=args.model_path,
        trust_remote_code=True,
        dtype="bfloat16",
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=4096,
        enforce_eager=True,
        limit_mm_per_prompt={"image": 12},
    )

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        stop_token_ids=[tokenizer.eos_token_id],
    )

    result_dir = Path(args.result_root) / args.name / args.test_dataset
    result_dir.mkdir(parents=True, exist_ok=True)
    answers_json_path = result_dir / f"answers_{args.few_shot_model}_shot_{model_name}_agentzoom_vllm.json"
    trace_jsonl_path = result_dir / f"trace_{args.few_shot_model}_shot_{model_name}_agentzoom_vllm.jsonl"

    if answers_json_path.exists() and not args.reproduce:
        all_answers = json.load(answers_json_path.open("r", encoding="utf-8"))
    else:
        all_answers = []
    done_images = set(a.get("image") for a in all_answers if isinstance(a, dict))

    test_json = Path(args.test_json_root) / f"{args.test_dataset}_format.json"
    chat_ad = json.load(test_json.open("r", encoding="utf-8"))

    pending: List[Dict[str, Any]] = []
    for rel_path in chat_ad.keys():
        if rel_path in done_images and not args.reproduce:
            continue

        obj = chat_ad[rel_path]
        conv = obj.get("conversation", [])
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

    print(
        f"[INFO] total_pending={len(pending)} effective_batch={effective_batch} "
        f"enable_self_check={args.enable_self_check} self_check_scope={args.self_check_scope}"
    )

    with trace_jsonl_path.open("w", encoding="utf-8") as tf:
        for i in tqdm(range(0, len(pending), effective_batch), desc=f"Evaluating {args.test_dataset}"):
            batch_items = pending[i : i + effective_batch]
            batch_results = run_batch_samples(
                llm=llm,
                tokenizer=tokenizer,
                batch_items=batch_items,
                sampling_params=sampling_params,
                zoom_conf_low=args.zoom_conf_low,
                zoom_conf_high=args.zoom_conf_high,
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

    # keep same summary entry as IAD-R1
    iad_r1_root = Path("/mnt/nfs/wmd/research/IAD-R1-main")
    if str(iad_r1_root) not in sys.path:
        sys.path.append(str(iad_r1_root))

    try:
        from helper.summary import caculate_accuracy_mmad

        caculate_accuracy_mmad(str(answers_json_path), normal_flag=args.normal_flag)
    except Exception as e:
        print(f"[WARN] summary failed: {e}")
        print(f"[INFO] You can run summary manually on: {answers_json_path}")


if __name__ == "__main__":
    main()
