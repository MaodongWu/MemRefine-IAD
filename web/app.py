import os
from typing import List

import streamlit as st
from PIL import Image

from backend import (
    DEFAULT_MODEL_NAME,
    DEFAULT_OPENAI_ENDPOINT,
    FeatureExtractor,
    aggregate_unknown_mode_decision,
    anomaly_distance_judgement,
    call_vlm_for_anomaly_and_location,
    draw_bbox_overlay,
    localize_by_normal_difference,
)


st.set_page_config(page_title="MemRefine-IAD Web", page_icon="IAD", layout="wide")
st.markdown(
    """
    <style>
    .main > div { padding-top: 1.1rem; }
    .block-card {
      border: 1px solid rgba(120,140,170,0.35);
      background: linear-gradient(180deg, rgba(248,251,255,0.92), rgba(244,248,252,0.78));
      border-radius: 14px;
      padding: 14px 16px;
      margin-bottom: 12px;
    }
    .result-badge {
      display: inline-block;
      border-radius: 999px;
      padding: 6px 12px;
      font-weight: 700;
      margin-right: 8px;
      border: 1px solid rgba(0,0,0,0.08);
    }
    .ok { background: #e8f8ed; color: #1e7a39; }
    .ng { background: #fdeceb; color: #b42318; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_extractor(device: str) -> FeatureExtractor:
    return FeatureExtractor(device=device)


def _read_upload_as_pil(uploaded_file) -> Image.Image:
    return Image.open(uploaded_file).convert("RGB")


def _label_to_text(label: str) -> str:
    if label == "A":
        return "正常"
    if label == "B":
        return "异常"
    return "未知"


def _load_known_categories() -> List[str]:
    visa_root = "/mnt/nfs/wmd/data/VisA"
    if not os.path.isdir(visa_root):
        return ["pcb1", "pcb2", "pcb3", "pcb4"]
    cats = []
    for x in sorted(os.listdir(visa_root)):
        p = os.path.join(visa_root, x)
        if os.path.isdir(p) and x != "split_csv":
            cats.append(x)
    return cats if cats else ["pcb1", "pcb2", "pcb3", "pcb4"]


st.title("MemRefine-IAD Real-time Detection Demo")
st.caption(
    "Mode-1: known category direct detection. "
    "Mode-2: unknown category with few normal references + distance fusion."
)

with st.sidebar:
    st.header("Runtime Config")
    st.caption("Fill your own endpoint/model/key. Supports local vLLM and external API.")
    endpoint_url = st.text_input(
        "Endpoint URL",
        value=os.getenv("VLM_ENDPOINT_URL", DEFAULT_OPENAI_ENDPOINT),
        help="Can be full endpoint (.../chat/completions) or base URL (.../v1).",
    )
    model_name = st.text_input("Model Name", value=os.getenv("VLM_MODEL_NAME", DEFAULT_MODEL_NAME))
    api_key = st.text_input("API Key", value=os.getenv("VLM_API_KEY", ""), type="password")
    timeout_s = st.slider("VLM timeout (seconds)", 30, 300, 120, 10)
    st.markdown("**External API speed tuning**")
    max_image_side = st.slider("Max image side for API upload", 640, 2048, 1280, 64)
    jpeg_quality = st.slider("JPEG quality for API upload", 55, 95, 85, 5)
    retries = st.slider("Retry count (on transient network errors)", 0, 4, 2, 1)

    device = st.selectbox("Feature device", ["cpu", "cuda"], index=0)
    metric = st.selectbox("Distance metric", ["cosine", "l2"], index=0)
    threshold_k = st.slider("Threshold k (mu + k*sigma)", 1.0, 6.0, 3.0, 0.1)

known_categories = _load_known_categories()

known_tab, unknown_tab = st.tabs([
    "Known Category: Direct Detection",
    "Unknown Category: Few-shot Distance + VLM",
])

with known_tab:
    st.subheader("Known Category Inference")
    col1, col2 = st.columns([1, 1])

    with col1:
        default_idx = known_categories.index("pcb1") if "pcb1" in known_categories else 0
        known_category = st.selectbox("Known category", known_categories, index=default_idx)
        known_query = st.file_uploader("Upload query image", type=["jpg", "jpeg", "png"], key="known_query")
        run_known = st.button("Run known-category detection", type="primary")

    with col2:
        if known_query is not None:
            st.image(_read_upload_as_pil(known_query), caption="Query Image", width="stretch")

    if run_known:
        if known_query is None:
            st.error("Please upload a query image first.")
        else:
            img = _read_upload_as_pil(known_query)
            with st.spinner("Calling VLM for anomaly + bbox localization..."):
                vr = call_vlm_for_anomaly_and_location(
                    image=img,
                    model_name=model_name,
                    endpoint_url=endpoint_url,
                    api_key=api_key,
                    known_category=known_category,
                    timeout_s=timeout_s,
                    max_image_side=max_image_side,
                    jpeg_quality=jpeg_quality,
                    retries=retries,
                )

            if not vr.ok:
                st.error(f"VLM call failed: {vr.error}")
                if vr.raw_text:
                    st.code(vr.raw_text)
            else:
                decision_text = _label_to_text(vr.anomaly_label or "Unknown")
                badge_cls = "ok" if vr.anomaly_label == "A" else "ng"
                st.markdown(
                    f"""
                    <div class="block-card">
                      <span class="result-badge {badge_cls}">检测结果：{decision_text}</span>
                      <span>类别：{known_category}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                boxed = draw_bbox_overlay(img, vr.bbox, label="anomaly" if vr.anomaly_label == "B" else "normal")
                st.image(boxed, caption="Detection overlay", width="stretch")
                st.write(f"**定位描述**: {vr.location or '未给出'}")
                st.write(f"**解释**: {vr.rationale or '未给出'}")
                with st.expander("Raw model output"):
                    st.code(vr.raw_text)

with unknown_tab:
    st.subheader("Unknown Category Inference")
    st.caption("Upload normal references, one query image, and optional text description.")

    col1, col2 = st.columns([1, 1])

    with col1:
        unknown_query = st.file_uploader("Upload query image", type=["jpg", "jpeg", "png"], key="unknown_query")
        normal_refs = st.file_uploader(
            "Upload normal reference images (multiple)",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            key="normal_refs",
        )
        unknown_desc = st.text_area(
            "Optional text description for this unknown product",
            placeholder="e.g., Metallic cylinder with smooth reflective surface; anomalies may appear as scratches or dents.",
            height=90,
        )
        run_unknown = st.button("Run unknown-category detection", type="primary")

    with col2:
        if unknown_query is not None:
            st.image(_read_upload_as_pil(unknown_query), caption="Query Image", width="stretch")
        if normal_refs:
            st.write(f"Normal references: {len(normal_refs)}")

    if run_unknown:
        if unknown_query is None:
            st.error("Please upload a query image.")
        elif not normal_refs:
            st.error("Please upload at least one normal reference image.")
        else:
            q_img = _read_upload_as_pil(unknown_query)
            n_imgs: List[Image.Image] = [_read_upload_as_pil(f) for f in normal_refs]
            extractor = get_extractor(device=device)

            with st.spinner("Computing feature-distance anomaly score..."):
                dr = anomaly_distance_judgement(
                    query_img=q_img,
                    normal_imgs=n_imgs,
                    extractor=extractor,
                    metric=metric,
                    threshold_k=threshold_k,
                )

            with st.spinner("Calling VLM for semantic detection + bbox..."):
                vr = call_vlm_for_anomaly_and_location(
                    image=q_img,
                    model_name=model_name,
                    endpoint_url=endpoint_url,
                    api_key=api_key,
                    known_category="unknown",
                    user_description=unknown_desc,
                    timeout_s=timeout_s,
                    max_image_side=max_image_side,
                    jpeg_quality=jpeg_quality,
                    retries=retries,
                )

            with st.spinner("Generating difference-based localization map..."):
                overlay, diff_bbox = localize_by_normal_difference(q_img, n_imgs)

            fused = aggregate_unknown_mode_decision(dr.predicted_label, vr.anomaly_label)

            c1, c2, c3 = st.columns(3)
            c1.metric("Distance score", f"{dr.score:.6f}")
            c2.metric("Threshold", f"{dr.threshold:.6f}")
            c3.metric("Distance decision", _label_to_text(dr.predicted_label))

            badge_cls = "ok" if fused == "A" else "ng"
            st.markdown(
                f"""
                <div class="block-card">
                  <span class="result-badge {badge_cls}">融合结果：{_label_to_text(fused)}</span>
                  <span>VLM判定：{_label_to_text(vr.anomaly_label or 'Unknown')} | 距离判定：{_label_to_text(dr.predicted_label)}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if vr.bbox is not None:
                boxed = draw_bbox_overlay(q_img, vr.bbox, label="anomaly" if fused == "B" else "normal")
                st.image(boxed, caption="VLM bbox overlay", width="stretch")
            elif overlay is not None:
                st.image(overlay, caption="Difference localization overlay", width="stretch")

            st.write(f"**定位描述**: {vr.location or '未给出'}")
            st.write(f"**Diff-map bbox**: {diff_bbox or '未给出'}")
            st.write(f"**解释**: {vr.rationale or '未给出'}")
            with st.expander("Raw details"):
                st.write(
                    {
                        "normal_scores": dr.normal_scores,
                        "vlm_ok": vr.ok,
                        "vlm_error": vr.error,
                        "vlm_bbox": vr.bbox,
                        "vlm_raw": vr.raw_text,
                    }
                )
