import base64
import io
import json
import os
import re
import time
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


# Common OpenAI-compatible defaults.
DEFAULT_OPENAI_ENDPOINT = "https://api.openai.com/v1/chat/completions"
DEFAULT_MODEL_NAME = "gpt-4o-mini"


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
    bbox: Optional[List[float]]
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
            # Keep web startup fast/stable: by default do not trigger online weight download.
            # Set MEMREFINE_WEB_USE_IMAGENET=1 if you explicitly want pretrained ResNet features.
            use_imagenet = os.getenv("MEMREFINE_WEB_USE_IMAGENET", "0") == "1"
            try:
                weights = tv_models.ResNet18_Weights.IMAGENET1K_V1 if use_imagenet else None
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
            except Exception:
                # Fallback to non-torch handcrafted descriptor below.
                self.use_torch = False
                self.model = None
                self.tf = None
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


def _prepare_request_image(img: Image.Image, max_side: int = 1280) -> Image.Image:
    """Resize image before API upload to reduce payload and latency."""
    out = img.convert("RGB")
    w, h = out.size
    long_side = max(w, h)
    if long_side <= max_side:
        return out

    scale = float(max_side) / float(long_side)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    return out.resize((nw, nh), Image.Resampling.LANCZOS)


def _pil_to_data_url(img: Image.Image, fmt: str = "JPEG", jpeg_quality: int = 85) -> str:
    buf = io.BytesIO()
    if fmt.upper() == "JPEG":
        img.save(buf, format="JPEG", quality=int(jpeg_quality), optimize=True)
    else:
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


def _parse_bbox(raw_bbox: Any) -> Optional[List[float]]:
    """Parse bbox as normalized [x1,y1,x2,y2] in [0,1]."""
    if not isinstance(raw_bbox, (list, tuple)) or len(raw_bbox) != 4:
        return None
    try:
        vals = [float(v) for v in raw_bbox]
    except Exception:
        return None
    x1, y1, x2, y2 = vals
    if x2 <= x1 or y2 <= y1:
        return None
    # If model outputs absolute-ish values, clamp anyway.
    vals = [max(0.0, min(1.0, v)) for v in vals]
    x1, y1, x2, y2 = vals
    if x2 <= x1 or y2 <= y1:
        return None
    return vals


def draw_bbox_overlay(image: Image.Image, bbox: Optional[List[float]], label: str = "") -> Image.Image:
    """Draw normalized bbox on image and return overlay image."""
    arr = np.asarray(image.convert("RGB")).copy()
    h, w = arr.shape[:2]
    if bbox is None:
        return Image.fromarray(arr)
    x1, y1, x2, y2 = bbox
    p1 = (int(x1 * w), int(y1 * h))
    p2 = (int(x2 * w), int(y2 * h))

    if cv2 is not None:
        cv2.rectangle(arr, p1, p2, (255, 64, 64), 3)
        if label:
            cv2.putText(arr, label, (p1[0], max(20, p1[1] - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 64, 64), 2)
    else:
        # Fallback: rough rectangle by numpy slicing.
        x1i, y1i = p1
        x2i, y2i = p2
        x1i, x2i = max(0, min(w - 1, x1i)), max(0, min(w - 1, x2i))
        y1i, y2i = max(0, min(h - 1, y1i)), max(0, min(h - 1, y2i))
        arr[y1i:y1i + 3, x1i:x2i] = [255, 64, 64]
        arr[y2i - 3:y2i, x1i:x2i] = [255, 64, 64]
        arr[y1i:y2i, x1i:x1i + 3] = [255, 64, 64]
        arr[y1i:y2i, x2i - 3:x2i] = [255, 64, 64]
    return Image.fromarray(arr)


def call_vlm_for_anomaly_and_location(
    image: Image.Image,
    model_name: Optional[str] = None,
    endpoint_url: Optional[str] = None,
    api_key: Optional[str] = None,
    known_category: str = "",
    user_description: str = "",
    timeout_s: int = 120,
    max_image_side: int = 1280,
    jpeg_quality: int = 85,
    retries: int = 2,
) -> VLMResult:
    if requests is None:
        return VLMResult(
            ok=False,
            anomaly_label=None,
            location="",
            rationale="",
            bbox=None,
            raw_text="",
            error="requests is not installed.",
        )

    model_name = (model_name or DEFAULT_MODEL_NAME).strip()
    endpoint_url = (endpoint_url or DEFAULT_OPENAI_ENDPOINT).strip()
    api_key = (api_key or "").strip()

    if not api_key:
        return VLMResult(
            ok=False,
            anomaly_label=None,
            location="",
            rationale="",
            bbox=None,
            raw_text="",
            error="Missing API key. Please provide it in the web sidebar.",
        )

    prompt = (
        "You are an industrial anomaly inspector. "
        "Given one image, decide anomaly and location with a bounding box. "
        "Return strict JSON with keys: "
        "anomaly_label (A or B), bbox, location, rationale. "
        "A means normal; B means anomaly. "
        "bbox must be either null or [x1,y1,x2,y2] normalized to [0,1]. "
        "location should be concise, e.g., 'top-left screw edge'."
    )
    if known_category.strip():
        prompt += f" Known category: {known_category.strip()}."
    if user_description.strip():
        prompt += f" Extra context from user: {user_description.strip()}."

    req_img = _prepare_request_image(image, max_side=max_image_side)
    data_url = _pil_to_data_url(req_img, fmt="JPEG", jpeg_quality=jpeg_quality)
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
        # Accept either full chat/completions endpoint or base URL.
        if endpoint_url.rstrip("/").endswith("/chat/completions"):
            post_url = endpoint_url.rstrip("/")
        else:
            post_url = f"{endpoint_url.rstrip('/')}/chat/completions"

        # Split connect/read timeout to fail fast on bad handshakes but keep generation window.
        connect_timeout = min(20, max(5, int(timeout_s // 6)))
        request_timeout = (connect_timeout, int(timeout_s))
        last_error = ""
        attempts = max(1, int(retries) + 1)

        for attempt in range(attempts):
            try:
                resp = requests.post(
                    post_url,
                    headers=headers,
                    data=json.dumps(payload),
                    timeout=request_timeout,
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
                        bbox=None,
                        raw_text=text,
                        error="Model returned non-JSON output.",
                    )

                label = str(parsed.get("anomaly_label", "")).strip().upper()
                if label not in {"A", "B"}:
                    label = None

                bbox = _parse_bbox(parsed.get("bbox"))
                return VLMResult(
                    ok=True,
                    anomaly_label=label,
                    location=str(parsed.get("location", "")).strip(),
                    rationale=str(parsed.get("rationale", "")).strip(),
                    bbox=bbox,
                    raw_text=text,
                )
            except Exception as inner:
                last_error = str(inner)
                # Retry transient handshake/transport failures with short backoff.
                if attempt < attempts - 1:
                    time.sleep(1.5 * (2 ** attempt))
                    continue
                break

        raise RuntimeError(last_error or "Unknown request error.")

    except Exception as e:
        return VLMResult(
            ok=False,
            anomaly_label=None,
            location="",
            rationale="",
            bbox=None,
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
