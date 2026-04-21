import base64
import io
import json
import math
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

try:
    import requests
except Exception:  # pragma: no cover
    requests = None


# Optional deps for better feature extraction and localization.
try:
    import torch
    import torchvision.transforms as T
    import torchvision.models as tv_models
except Exception:  # pragma: no cover
    torch = None
    T = None
    tv_models = None

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None


@dataclass
class DistanceResult:
    score: float
    threshold: float
    predicted_label: str
    normal_scores: List[float]


@dataclass
class VLMResult:
    ok: bool
    anomaly_label: Optional[str]
    location: str
    rationale: str
    raw_text: str
    error: str = ""


class FeatureExtractor:
    """Feature extractor with graceful fallback.

    Priority:
    1) ResNet18 penultimate features (if torch/torchvision available)
    2) Color-statistics + gradient histogram fallback (pure numpy/PIL)
    """

    def __init__(self, device: str = "cpu") -> None:
        self.device = device
        self.use_torch = bool(torch is not None and tv_models is not None and T is not None)

        if self.use_torch:
            weights = tv_models.ResNet18_Weights.IMAGENET1K_V1
            model = tv_models.resnet18(weights=weights)
            self.model = torch.nn.Sequential(*list(model.children())[:-1]).eval().to(self.device)
            self.tf = T.Compose(
                [
                    T.Resize(256),
                    T.CenterCrop(224),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
        else:
            self.model = None
            self.tf = None

    def extract(self, img: Image.Image) -> np.ndarray:
        img = img.convert("RGB")

        if self.use_torch:
            with torch.no_grad():
                x = self.tf(img).unsqueeze(0).to(self.device)
                feat = self.model(x).flatten().detach().cpu().numpy().astype(np.float32)
            norm = np.linalg.norm(feat) + 1e-12
            return feat / norm

        # Fallback descriptor: color moments + coarse texture histogram.
        arr = np.asarray(img.resize((224, 224)), dtype=np.float32) / 255.0
        mean = arr.mean(axis=(0, 1))
        std = arr.std(axis=(0, 1))

        gray = (0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]).astype(np.float32)
        gx = np.abs(np.diff(gray, axis=1, prepend=gray[:, :1]))
        gy = np.abs(np.diff(gray, axis=0, prepend=gray[:1, :]))
        mag = np.sqrt(gx * gx + gy * gy)
        hist, _ = np.histogram(mag, bins=32, range=(0.0, float(mag.max() + 1e-8)))
        hist = hist.astype(np.float32)
        hist = hist / (hist.sum() + 1e-12)

        feat = np.concatenate([mean, std, hist], axis=0).astype(np.float32)
        norm = np.linalg.norm(feat) + 1e-12
        return feat / norm


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(1.0 - np.dot(a, b) / ((np.linalg.norm(a) + 1e-12) * (np.linalg.norm(b) + 1e-12)))


def _l2_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def anomaly_distance_judgement(
    query_img: Image.Image,
    normal_imgs: List[Image.Image],
    extractor: FeatureExtractor,
    metric: str = "cosine",
    threshold_k: float = 3.0,
) -> DistanceResult:
    if not normal_imgs:
        raise ValueError("normal_imgs is empty.")

    dist_fn = _cosine_distance if metric == "cosine" else _l2_distance

    normal_feats = [extractor.extract(im) for im in normal_imgs]
    query_feat = extractor.extract(query_img)

    center = np.mean(np.stack(normal_feats, axis=0), axis=0)

    normal_scores = [dist_fn(f, center) for f in normal_feats]
    q_score = dist_fn(query_feat, center)

    mu = float(np.mean(normal_scores))
    sigma = float(np.std(normal_scores))
    threshold = mu + threshold_k * (sigma + 1e-8)

    label = "B" if q_score > threshold else "A"
    return DistanceResult(
        score=q_score,
        threshold=threshold,
        predicted_label=label,
        normal_scores=normal_scores,
    )


def _pil_to_data_url(img: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    mime = "image/png" if fmt.upper() == "PNG" else "image/jpeg"
    return f"data:{mime};base64,{b64}"


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    # Try direct JSON first.
    text = text.strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # Fallback: first {...} block.
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return None
    candidate = match.group(0)
    try:
        obj = json.loads(candidate)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def call_vlm_for_anomaly_and_location(
    image: Image.Image,
    model_name: str,
    base_url: str = "http://127.0.0.1:8000/v1",
    api_key: str = "EMPTY",
    known_category: str = "",
    timeout_s: int = 120,
) -> VLMResult:
    if requests is None:
        return VLMResult(
            ok=False,
            anomaly_label=None,
            location="",
            rationale="",
            raw_text="",
            error="requests is not installed.",
        )

    prompt = (
        "You are an industrial anomaly inspector. "
        "Given one image, decide anomaly and location. "
        "Return strict JSON with keys: "
        "anomaly_label (A or B), location, rationale. "
        "A means normal; B means anomaly. "
        "location should be concise, e.g., 'top-left screw edge'."
    )
    if known_category.strip():
        prompt += f" Known category: {known_category.strip()}."

    data_url = _pil_to_data_url(image)
    payload = {
        "model": model_name,
        "temperature": 0.0,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ],
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    try:
        resp = requests.post(
            f"{base_url.rstrip('/')}/chat/completions",
            headers=headers,
            data=json.dumps(payload),
            timeout=timeout_s,
        )
        resp.raise_for_status()
        body = resp.json()
        text = body["choices"][0]["message"]["content"]

        parsed = _extract_json_object(text)
        if parsed is None:
            return VLMResult(
                ok=False,
                anomaly_label=None,
                location="",
                rationale="",
                raw_text=text,
                error="Model returned non-JSON output.",
            )

        label = str(parsed.get("anomaly_label", "")).strip().upper()
        if label not in {"A", "B"}:
            label = None

        return VLMResult(
            ok=True,
            anomaly_label=label,
            location=str(parsed.get("location", "")).strip(),
            rationale=str(parsed.get("rationale", "")).strip(),
            raw_text=text,
        )

    except Exception as e:
        return VLMResult(
            ok=False,
            anomaly_label=None,
            location="",
            rationale="",
            raw_text="",
            error=str(e),
        )


def localize_by_normal_difference(
    query_img: Image.Image,
    normal_imgs: List[Image.Image],
) -> Tuple[Optional[Image.Image], str]:
    """Generate an interpretable diff heatmap and coarse bbox from normal references."""
    if not normal_imgs:
        return None, ""

    q = query_img.convert("RGB")
    w, h = q.size
    q_arr = np.asarray(q, dtype=np.float32)

    normals = []
    for im in normal_imgs:
        normals.append(np.asarray(im.convert("RGB").resize((w, h)), dtype=np.float32))

    mean_normal = np.mean(np.stack(normals, axis=0), axis=0)
    diff = np.mean(np.abs(q_arr - mean_normal), axis=2)

    # Robust threshold by percentile.
    thr = float(np.percentile(diff, 95))
    mask = (diff >= thr).astype(np.uint8)

    if cv2 is not None:
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if n_labels <= 1:
            return Image.fromarray((diff / (diff.max() + 1e-8) * 255).astype(np.uint8)), "No salient anomalous region found"

        # Largest non-background component.
        areas = stats[1:, cv2.CC_STAT_AREA]
        idx = int(np.argmax(areas)) + 1
        x, y, bw, bh, _ = stats[idx]
        bbox_text = f"x={x}, y={y}, w={bw}, h={bh}"

        heat = (diff / (diff.max() + 1e-8) * 255).astype(np.uint8)
        heat_rgb = np.stack([heat, np.zeros_like(heat), 255 - heat], axis=2)
        overlay = (0.55 * q_arr + 0.45 * heat_rgb).astype(np.uint8)
        cv2.rectangle(overlay, (x, y), (x + bw, y + bh), (255, 0, 0), 2)
        return Image.fromarray(overlay), bbox_text

    # Fallback bbox via nonzero min-max.
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return Image.fromarray((diff / (diff.max() + 1e-8) * 255).astype(np.uint8)), "No salient anomalous region found"

    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    bbox_text = f"x={x1}, y={y1}, w={x2-x1+1}, h={y2-y1+1}"

    heat = (diff / (diff.max() + 1e-8) * 255).astype(np.uint8)
    heat_rgb = np.stack([heat, np.zeros_like(heat), 255 - heat], axis=2)
    overlay = (0.55 * q_arr + 0.45 * heat_rgb).astype(np.uint8)
    return Image.fromarray(overlay), bbox_text


def aggregate_unknown_mode_decision(distance_label: str, vlm_label: Optional[str]) -> str:
    """Conservative fusion for unknown-class mode.

    Priority rule:
    - If either says anomaly (B), output B.
    - Else output A.
    """
    if distance_label == "B" or vlm_label == "B":
        return "B"
    return "A"
