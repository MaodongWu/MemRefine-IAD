import os
from typing import List

import streamlit as st
from PIL import Image

from backend import (
    FeatureExtractor,
    aggregate_unknown_mode_decision,
    anomaly_distance_judgement,
    call_vlm_for_anomaly_and_location,
    localize_by_normal_difference,
)


st.set_page_config(page_title="MemRefine-IAD Web", page_icon="🔎", layout="wide")


@st.cache_resource
def get_extractor(device: str) -> FeatureExtractor:
    return FeatureExtractor(device=device)


def _read_upload_as_pil(uploaded_file) -> Image.Image:
    return Image.open(uploaded_file).convert("RGB")


def _label_to_text(label: str) -> str:
    if label == "A":
        return "A (Normal / No anomaly)"
    if label == "B":
        return "B (Anomaly exists)"
    return "Unknown"


st.title("MemRefine-IAD Real-time Detection Demo")
st.caption(
    "Mode-1: Known category, direct anomaly + location by VLM. "
    "Mode-2: Unknown category, few normal references + feature distance + VLM location."
)

with st.sidebar:
    st.header("Runtime Config")
    model_name = st.text_input(
        "VLM Model Name",
        value=os.getenv("VLM_MODEL_NAME", "/mnt/nfs/wmd/model/MemRefine/MemRefine_model/MemRefine(LLaVA-OneVision-SI-7B)"),
    )
    base_url = st.text_input("OpenAI-Compatible Base URL", value=os.getenv("VLM_BASE_URL", "http://127.0.0.1:8000/v1"))
    api_key = st.text_input("API Key", value=os.getenv("VLM_API_KEY", "EMPTY"), type="password")
    timeout_s = st.slider("VLM timeout (seconds)", 30, 300, 120, 10)

    device = st.selectbox("Feature device", ["cpu", "cuda"], index=0)
    metric = st.selectbox("Distance metric", ["cosine", "l2"], index=0)
    threshold_k = st.slider("Threshold k (mu + k*sigma)", 1.0, 6.0, 3.0, 0.1)

extractor = get_extractor(device=device)

known_tab, unknown_tab = st.tabs([
    "Known Category: Direct Model Detection",
    "Unknown Category: Few-shot Distance + Model",
])

with known_tab:
    st.subheader("Known Category Inference")
    col1, col2 = st.columns([1, 1])

    with col1:
        known_category = st.text_input("Known category (optional)", value="pcb1")
        known_query = st.file_uploader("Upload query image", type=["jpg", "jpeg", "png"], key="known_query")
        run_known = st.button("Run known-category detection", type="primary")

    with col2:
        if known_query is not None:
            st.image(_read_upload_as_pil(known_query), caption="Query Image", use_container_width=True)

    if run_known:
        if known_query is None:
            st.error("Please upload a query image first.")
        else:
            img = _read_upload_as_pil(known_query)
            with st.spinner("Calling VLM for anomaly + location..."):
                vr = call_vlm_for_anomaly_and_location(
                    image=img,
                    model_name=model_name,
                    base_url=base_url,
                    api_key=api_key,
                    known_category=known_category,
                    timeout_s=timeout_s,
                )

            if not vr.ok:
                st.error(f"VLM call failed: {vr.error}")
                if vr.raw_text:
                    st.code(vr.raw_text)
            else:
                st.success("Done.")
                st.write(f"**Final anomaly decision:** {_label_to_text(vr.anomaly_label or 'Unknown')}")
                st.write(f"**Location:** {vr.location or 'N/A'}")
                st.write(f"**Rationale:** {vr.rationale or 'N/A'}")
                with st.expander("Raw model output"):
                    st.code(vr.raw_text)

with unknown_tab:
    st.subheader("Unknown Category Inference")
    st.caption("Upload several normal images as reference, then upload one query image.")

    col1, col2 = st.columns([1, 1])

    with col1:
        unknown_query = st.file_uploader("Upload query image", type=["jpg", "jpeg", "png"], key="unknown_query")
        normal_refs = st.file_uploader(
            "Upload normal reference images (multiple)",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            key="normal_refs",
        )
        run_unknown = st.button("Run unknown-category detection", type="primary")

    with col2:
        if unknown_query is not None:
            st.image(_read_upload_as_pil(unknown_query), caption="Query Image", use_container_width=True)
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

            with st.spinner("Computing feature-distance anomaly score..."):
                dr = anomaly_distance_judgement(
                    query_img=q_img,
                    normal_imgs=n_imgs,
                    extractor=extractor,
                    metric=metric,
                    threshold_k=threshold_k,
                )

            with st.spinner("Calling VLM for semantic location..."):
                vr = call_vlm_for_anomaly_and_location(
                    image=q_img,
                    model_name=model_name,
                    base_url=base_url,
                    api_key=api_key,
                    known_category="unknown",
                    timeout_s=timeout_s,
                )

            with st.spinner("Generating difference-based localization map..."):
                overlay, diff_bbox = localize_by_normal_difference(q_img, n_imgs)

            fused = aggregate_unknown_mode_decision(dr.predicted_label, vr.anomaly_label)

            st.success("Done.")
            c1, c2, c3 = st.columns(3)
            c1.metric("Distance score", f"{dr.score:.6f}")
            c2.metric("Threshold", f"{dr.threshold:.6f}")
            c3.metric("Distance decision", _label_to_text(dr.predicted_label))

            st.write(f"**VLM decision:** {_label_to_text(vr.anomaly_label or 'Unknown')}")
            st.write(f"**Fused final decision:** {_label_to_text(fused)}")
            st.write(f"**VLM location:** {vr.location or 'N/A'}")
            st.write(f"**Diff-map bbox:** {diff_bbox or 'N/A'}")
            st.write(f"**VLM rationale:** {vr.rationale or 'N/A'}")

            if overlay is not None:
                st.image(overlay, caption="Difference localization overlay", use_container_width=True)

            with st.expander("Raw details"):
                st.write({
                    "normal_scores": dr.normal_scores,
                    "vlm_ok": vr.ok,
                    "vlm_error": vr.error,
                    "vlm_raw": vr.raw_text,
                })
