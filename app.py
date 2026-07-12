import streamlit as st
import numpy as np
import time
from PIL import Image

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Diabetic Retinopathy Detection",
    page_icon="👁️",
    layout="wide",
)

from utils.preprocessing import preprocess_image
from utils.inference import load_model, predict
from utils.gradcam import generate_gradcam_overlay
from utils.report import get_clinical_report

MODEL_PATH = "models/dr_model.keras"

GRADE_INFO = {
    0: ("No DR",            "🟢", "Low",          "12 months"),
    1: ("Mild NPDR",        "🟡", "Low-Moderate", "6–12 months"),
    2: ("Moderate NPDR",    "🟠", "Moderate",     "3–6 months"),
    3: ("Severe NPDR",      "🔴", "High",         "1 month"),
    4: ("Proliferative DR", "🆘", "Very High",    "Immediate"),
}

@st.cache_resource(show_spinner="Loading model…")
def get_model():
    return load_model(MODEL_PATH)

st.title("👁️ Diabetic Retinopathy Detection System")
st.caption("EfficientNet-B3 + CBAM · Trained on APTOS 2019· Grades 0–4")
st.divider()

col_left, col_right = st.columns([1, 1.6], gap="large")

with col_left:
    st.subheader("Upload Fundus Image")
    uploaded = st.file_uploader("PNG / JPG / JPEG", type=["png", "jpg", "jpeg"])

    if uploaded:
        raw_pil = Image.open(uploaded).convert("RGB")
        st.image(raw_pil, caption="Uploaded fundus image", width="stretch")

        st.markdown("**Image info**")
        st.table({
            "Field": ["Filename", "Dimensions", "Preprocessing", "Model input"],
            "Value": [uploaded.name, f"{raw_pil.width}×{raw_pil.height}px",
                      "Green-channel CLAHE", "300×300×3"],
        })
        st.markdown("**Model info**")
        st.table({
            "Field": ["Architecture", "Attention", "Trained on", "Classes"],
            "Value": ["EfficientNet-B3", "CBAM", "APTOS 2019 Blindness Detection", "5 (Grade 0–4)"],
        })

        if st.button("🔍 Analyze image", use_container_width=True, type="primary"):
            with st.spinner("CLAHE → Inference → Grad-CAM…"):
                t0 = time.time()
                model = get_model()
                processed   = preprocess_image(np.array(raw_pil))
                probs, grade = predict(model, processed)
                heatmap_pil  = generate_gradcam_overlay(model, processed, raw_pil)
                elapsed = time.time() - t0

            st.session_state["results"] = dict(
                probs=probs, grade=grade, heatmap=heatmap_pil,
                elapsed=elapsed, raw_pil=raw_pil
            )
            st.rerun()

with col_right:
    if "results" not in st.session_state:
        st.info("Upload a fundus image and click **Analyze image** to see results here.")
    else:
        r = st.session_state["results"]
        grade = r["grade"]; probs = r["probs"]
        label, icon, risk, followup = GRADE_INFO[grade]

        st.subheader(f"{icon} Grade {grade} — {label} detected")
        st.caption(f"Inference completed in {r['elapsed']:.2f}s")

        m1, m2, m3 = st.columns(3)
        m1.metric("Confidence", f"{probs[grade]*100:.1f}%")
        m2.metric("Risk level",  risk)
        m3.metric("Follow-up",   followup)
        st.divider()

        st.markdown("**Grade probabilities**")
        for i, (lbl, ico, _, _) in GRADE_INFO.items():
            pct = float(probs[i]) * 100
            st.progress(min(int(pct), 100), text=f"Grade {i} — {lbl}:  **{pct:.1f}%**")
        st.divider()

        st.markdown("**Grad-CAM explainability**")
        gc1, gc2 = st.columns(2)
        gc1.image(r["raw_pil"],  caption="Original",         width="stretch")
        gc2.image(r["heatmap"],  caption="Grad-CAM overlay",  width="stretch")
        st.divider()

        report = get_clinical_report(grade, probs)
        if grade >= 3:
            st.error(f"**{report['headline']}**")
        elif grade == 2:
            st.warning(f"**{report['headline']}**")
        else:
            st.success(f"**{report['headline']}**")
        st.markdown(report["body"])
        st.divider()
        st.caption(
            "⚠️ This tool is for research/screening support only. "
            "Not a substitute for clinical diagnosis. Always consult a qualified ophthalmologist."
        )