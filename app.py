
"""
Streamlit dashboard for the Lung Nodule Screening demo project.
 
Run with:  streamlit run app.py
"""
import sys
import os
import json
import io
import numpy as np
import torch
import streamlit as st
import matplotlib.pyplot as plt
from fpdf import FPDF
 
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from preprocessing import normalize_for_model, normalize_to_uint8
from synthetic_data import generate_dataset
from model import build_model
from gradcam import GradCAM
 
st.set_page_config(page_title="Lung Nodule Screening (Demo)", layout="wide",
                    page_icon="🫁")
 
MODEL_PATH = "models/densenet121_demo.pth"
METRICS_PATH = "models/training_results.json"
 
# ---------- custom styling ----------
CUSTOM_CSS = """
<style>
:root {
    --primary: #0E7C7B;
    --primary-dark: #0A5E5D;
    --accent: #E8F4F3;
    --ink: #1A2E35;
    --muted: #5B7480;
}
 
/* Overall font + background tone */
html, body, [class*="css"] {
    font-family: "Inter", "Segoe UI", sans-serif;
}
.main .block-container {
    padding-top: 1.5rem;
    max-width: 1200px;
}
 
/* ---------- animation keyframes ---------- */
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes fadeIn {
    from { opacity: 0; }
    to   { opacity: 1; }
}
@keyframes gentlePulse {
    0%, 100% { box-shadow: 0 0 0 0 rgba(14, 124, 123, 0.25); }
    50%      { box-shadow: 0 0 0 8px rgba(14, 124, 123, 0); }
}
@keyframes shimmerGradient {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
 
/* Every page fades/slides in on load -- subtle, not distracting */
.main .block-container > div {
    animation: fadeInUp 0.45s ease-out;
}
 
/* Hero header bar -- slow animated gradient sheen */
.app-header {
    background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 50%, var(--primary) 100%);
    background-size: 200% 200%;
    animation: shimmerGradient 8s ease infinite, fadeIn 0.5s ease-out;
    padding: 1.6rem 2rem;
    border-radius: 14px;
    color: white;
    margin-bottom: 1.4rem;
    box-shadow: 0 4px 14px rgba(14, 124, 123, 0.25);
}
.app-header h1 {
    color: white !important;
    font-size: 1.7rem;
    margin: 0 0 0.3rem 0;
}
.app-header p {
    color: #E8F4F3;
    margin: 0;
    font-size: 0.95rem;
}
 
/* Section cards -- fade in, lift slightly on hover */
.info-card {
    background: var(--accent);
    border-left: 4px solid var(--primary);
    padding: 1rem 1.2rem;
    border-radius: 8px;
    margin-bottom: 1rem;
    animation: fadeInUp 0.5s ease-out;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.info-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(14, 124, 123, 0.12);
}
 
/* Sidebar polish */
section[data-testid="stSidebar"] {
    background-color: #F4F8F9;
    border-right: 1px solid #E0E8EA;
}
section[data-testid="stSidebar"] h1 {
    font-size: 1.25rem;
    color: var(--primary-dark);
}
/* Sidebar nav options get a smooth hover slide */
section[data-testid="stSidebar"] label {
    transition: transform 0.15s ease, color 0.15s ease;
}
section[data-testid="stSidebar"] label:hover {
    transform: translateX(3px);
    color: var(--primary-dark);
}
 
/* Metric cards -- fade in staggered, lift on hover */
div[data-testid="stMetric"] {
    background: white;
    border: 1px solid #E0E8EA;
    border-radius: 10px;
    padding: 0.8rem 1rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    animation: fadeInUp 0.5s ease-out;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
div[data-testid="stMetric"]:hover {
    transform: translateY(-3px);
    box-shadow: 0 6px 14px rgba(0,0,0,0.08);
}
 
/* Buttons -- smooth hover, gentle press feedback */
.stButton > button {
    border-radius: 8px;
    border: 1px solid var(--primary);
    font-weight: 500;
    transition: transform 0.12s ease, box-shadow 0.2s ease, background-color 0.2s ease;
}
.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 10px rgba(14, 124, 123, 0.18);
}
.stButton > button:active {
    transform: translateY(0px) scale(0.98);
}
.stButton > button[kind="primary"] {
    background-color: var(--primary);
    border-color: var(--primary);
    animation: gentlePulse 2.5s ease-in-out infinite;
}
.stButton > button[kind="primary"]:hover {
    background-color: var(--primary-dark);
    border-color: var(--primary-dark);
    animation: none;
}
 
/* Images -- soft fade-in as they render (e.g. CT slices, Grad-CAM) */
[data-testid="stImage"] img {
    animation: fadeIn 0.5s ease-out;
    transition: transform 0.2s ease;
    border-radius: 6px;
}
[data-testid="stImage"] img:hover {
    transform: scale(1.02);
}
 
/* Alerts (info/warning/error boxes) fade+slide in */
div[data-testid="stAlert"] {
    animation: fadeInUp 0.4s ease-out;
}
 
/* Reduce excess vertical whitespace between elements */
div[data-testid="stVerticalBlock"] > div {
    gap: 0.5rem;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
 
 
def render_header(title: str, subtitle: str = ""):
    """Consistent branded header used at the top of every page."""
    st.markdown(
        f"""<div class="app-header">
              <h1>🫁 {title}</h1>
              {f'<p>{subtitle}</p>' if subtitle else ''}
            </div>""",
        unsafe_allow_html=True,
    )
 
 
# ---------- caching ----------
@st.cache_resource
def load_model():
    model = build_model(pretrained=False)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model
 
 
@st.cache_data
def load_metrics():
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH) as f:
            return json.load(f)
    return None
 
 
@st.cache_data
def load_sample_scans():
    images, labels, _ = generate_dataset(n_per_class=6, size=64, seed=7)
    return images, labels
 
 
CLASS_NAMES = {0: "No nodule detected", 1: "Nodule-like region detected"}
 
# ---------- sidebar ----------
st.sidebar.markdown(
    """<div style="text-align:center; padding: 0.5rem 0 1rem 0;">
         <span style="font-size:2rem;">🫁</span><br>
         <span style="font-weight:600; font-size:1.1rem; color:#0A5E5D;">
           Lung Nodule Screening
         </span>
       </div>""",
    unsafe_allow_html=True,
)
page = st.sidebar.radio("Navigate", ["Home", "Upload / Select Scan", "Analysis & Results",
                                      "Explainability", "Model Performance", "Report"],
                         label_visibility="collapsed")
 
st.sidebar.markdown("---")
st.sidebar.warning(
    "⚠️ **Research/educational demo only.** Not a medical device. "
    "Not for clinical use or diagnosis."
)
 
if "selected_img" not in st.session_state:
    st.session_state.selected_img = None
    st.session_state.selected_label = None
    st.session_state.result = None
 
# ============================================================
if page == "Home":
    render_header("AI-Based Pulmonary Nodule Detection & Risk Screening",
                  "DenseNet121 + Grad-CAM screening pipeline")
 
    st.markdown(
        """<div class="info-card">
        <b>Pipeline:</b> CT slice → HU windowing/normalization → DenseNet121
        (transfer learning) classifier → confidence score → Grad-CAM heatmap → PDF report.
        </div>""",
        unsafe_allow_html=True,
    )
 
    c1, c2, c3 = st.columns(3)
    c1.metric("Model", "DenseNet121")
    c2.metric("Explainability", "Grad-CAM")
    c3.metric("Task", "Nodule Screening")

 
# ============================================================
elif page == "Upload / Select Scan":
    render_header("Upload or Select a CT Patch",
                  "Real DICOM/.mhd upload hooks into the same preprocessing pipeline used here")
 
    images, labels = load_sample_scans()
    n_cols = 6
    for row_start in range(0, len(images), n_cols):
        cols = st.columns(n_cols)
        for j, i in enumerate(range(row_start, min(row_start + n_cols, len(images)))):
            with cols[j]:
                disp = normalize_to_uint8(images[i])
                st.image(disp, caption=f"Sample {i+1}", use_container_width=True)
                if st.button("Select", key=f"sel_{i}"):
                    st.session_state.selected_img = images[i]
                    st.session_state.selected_label = labels[i]
                    st.session_state.result = None
 
    st.markdown("---")
    uploaded = st.file_uploader("Or upload your own grayscale image (PNG/JPG, treated as a raw HU-like array for demo purposes)",
                                 type=["png", "jpg", "jpeg"])
    if uploaded is not None:
        from PIL import Image
        img = np.array(Image.open(uploaded).convert("L")).astype(np.float32)
        img = (img / 255.0) * 1400 - 1000  # map back into a pseudo-HU range for demo
        st.session_state.selected_img = img
        st.session_state.selected_label = None
        st.session_state.result = None
        st.success("Uploaded image loaded.")
 
    if st.session_state.selected_img is not None:
        st.markdown("### Selected scan")
        st.image(normalize_to_uint8(st.session_state.selected_img), width=200)
    else:
        st.warning("Select a sample patch above (or upload an image) to continue to Analysis.")
 
# ============================================================
elif page == "Analysis & Results":
    render_header("Run AI Analysis", "Preprocess, classify, and review confidence for the selected scan")
    if st.session_state.selected_img is None:
        st.warning("Go to **Upload / Select Scan** first.")
    else:
        img = st.session_state.selected_img
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Original")
            st.image(normalize_to_uint8(img), use_container_width=True)
        with c2:
            st.subheader("Preprocessed (lung-windowed, normalized)")
            processed = normalize_for_model(img)
            st.image((processed * 255).astype(np.uint8), use_container_width=True)
 
        if st.button("▶ Run Model", type="primary"):
            model = load_model()
            tensor = torch.from_numpy(processed).unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                out = model(tensor)
                probs = torch.softmax(out, dim=1)[0].numpy()
                pred_class = int(np.argmax(probs))
            st.session_state.result = {
                "pred_class": pred_class,
                "probs": probs.tolist(),
                "tensor": tensor,
            }
 
        if st.session_state.result:
            r = st.session_state.result
            st.markdown("---")
            m1, m2, m3 = st.columns(3)
            m1.metric("Prediction", CLASS_NAMES[r["pred_class"]])
            m2.metric("Confidence", f"{max(r['probs'])*100:.1f}%")
            risk = "Higher" if r["pred_class"] == 1 and max(r["probs"]) > 0.7 else \
                   ("Indeterminate" if r["pred_class"] == 1 else "Low")
            m3.metric("Risk category (demo heuristic)", risk)
 
            fig, ax = plt.subplots(figsize=(4, 3))
            ax.bar(["No nodule", "Nodule-like"], r["probs"], color=["#4CAF50", "#E53935"])
            ax.set_ylabel("Probability")
            ax.set_ylim(0, 1)
            st.pyplot(fig)
 
            st.caption(
                "Risk category here is a simple confidence-based heuristic for "
                "this demo, not a clinically validated malignancy-risk score. "
                "A real deployment on LIDC-IDRI would use the radiologist "
                "malignancy-likelihood (1-5) annotations for this instead."
            )
 
# ============================================================
elif page == "Explainability":
    render_header("Grad-CAM Explainability", "Visualizing which regions influenced the model's prediction")
    if st.session_state.result is None:
        st.warning("Run the analysis first on the **Analysis & Results** page.")
    else:
        model = load_model()
        tensor = st.session_state.result["tensor"]
        target_layer = model.features.norm5  # last conv feature block of DenseNet121
        cam_tool = GradCAM(model, target_layer)
        cam, class_idx, probs = cam_tool.generate(tensor)
 
        orig = tensor[0, 0].numpy()
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(orig, cmap="gray")
        axes[0].set_title("Preprocessed CT")
        axes[0].axis("off")
 
        axes[1].imshow(cam, cmap="jet")
        axes[1].set_title("Grad-CAM heatmap")
        axes[1].axis("off")
 
        axes[2].imshow(orig, cmap="gray")
        axes[2].imshow(cam, cmap="jet", alpha=0.45)
        axes[2].set_title("Overlay")
        axes[2].axis("off")
 
        st.pyplot(fig)
        st.info(
            "🔍 Highlighted regions show where the model focused when making "
            "its prediction. **This is model attention, not proof of disease** "
            "— it does not confirm the presence of any abnormality."
        )
        st.session_state.gradcam_fig = fig
 
# ============================================================
elif page == "Model Performance":
    render_header("Model Performance", "Evaluated on held-out synthetic-demo test patients (patient-level split)")
    metrics = load_metrics()
    if metrics is None:
        st.warning("No training results found. Run `python src/train.py` first.")
    else:
        tm = metrics["test_metrics"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", f"{tm['accuracy']*100:.1f}%")
        c2.metric("Sensitivity (Recall)", f"{tm['sensitivity_recall']*100:.1f}%")
        c3.metric("Specificity", f"{tm['specificity']*100:.1f}%")
        c4.metric("ROC-AUC", f"{tm['roc_auc']:.3f}" if tm['roc_auc'] else "N/A")
 
        c5, c6 = st.columns(2)
        c5.metric("Precision", f"{tm['precision']*100:.1f}%")
        c6.metric("F1 Score", f"{tm['f1_score']*100:.1f}%")
 
        st.markdown("#### Confusion Matrix")
        cm = np.array(tm["confusion_matrix"])
        fig, ax = plt.subplots(figsize=(3, 3))
        ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=14)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["No nodule", "Nodule"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["No nodule", "Nodule"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        st.pyplot(fig)
 
        st.markdown("#### ROC Curve")
        fig2, ax2 = plt.subplots(figsize=(4, 4))
        ax2.plot(metrics["roc_fpr"], metrics["roc_tpr"], label="Model")
        ax2.plot([0, 1], [0, 1], "--", color="gray", label="Random")
        ax2.set_xlabel("False Positive Rate"); ax2.set_ylabel("True Positive Rate")
        ax2.legend()
        st.pyplot(fig2)
 
        st.markdown("#### Training Curves")
        fig3, ax3 = plt.subplots(figsize=(5, 3))
        ax3.plot(metrics["history"]["train_loss"], label="Train loss")
        ax3.plot(metrics["history"]["val_accuracy"], label="Val accuracy")
        ax3.legend()
        st.pyplot(fig3)
 
        st.caption(f"Note from training run: {metrics['note']}")
 
        with st.expander("Why sensitivity & specificity matter more than accuracy here"):
            st.write("""
In a screening context, a **false negative** (missing a real nodule) can
delay diagnosis and treatment — so **sensitivity (recall)** must be kept
high. A **false positive** causes unnecessary anxiety/follow-up scans, so
**specificity** also matters, but a screening tool is usually tuned to
favor sensitivity even at some specificity cost. Accuracy alone can hide
poor performance on the minority/positive class, especially with imbalanced
data — which is exactly the situation in real nodule datasets (most CT
regions do not contain nodules).
""")
 
# ============================================================
elif page == "Report":
    render_header("Downloadable PDF Report", "Export the current prediction, visuals, and disclaimer as a PDF")
    if st.session_state.result is None:
        st.warning("Run an analysis first on **Analysis & Results**.")
    else:
        r = st.session_state.result
        if st.button("Generate PDF Report"):
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Helvetica", "B", 16)
            pdf.cell(0, 10, "Lung Nodule Screening - Analysis Report", ln=True)
            pdf.set_font("Helvetica", "", 11)
            pdf.ln(4)
            pdf.multi_cell(0, 7,
                "Scan type: Demo CT patch (synthetic)\n"
                f"Predicted class: {CLASS_NAMES[r['pred_class']]}\n"
                f"Confidence: {max(r['probs'])*100:.1f}%\n"
                f"Model: DenseNet121 (transfer learning), trained on synthetic demo data\n"
            )
            pdf.ln(4)
            pdf.set_font("Helvetica", "B", 12)
            pdf.cell(0, 8, "Medical Disclaimer", ln=True)
            pdf.set_font("Helvetica", "", 10)
            pdf.multi_cell(0, 6,
                "This report is generated by a research/educational demo system. "
                "It does NOT constitute a medical diagnosis. Grad-CAM highlighted "
                "regions represent model attention only, not confirmed pathology. "
                "Consult a qualified radiologist/physician for any real clinical "
                "concern."
            )
 
            # save preprocessed + gradcam images if available
            tmp_dir = "reports/_tmp"
            os.makedirs(tmp_dir, exist_ok=True)
            orig_path = f"{tmp_dir}/orig.png"
            plt.imsave(orig_path, r["tensor"][0, 0].numpy(), cmap="gray")
            pdf.ln(4)
            pdf.set_font("Helvetica", "B", 12)
            pdf.cell(0, 8, "Scan Image", ln=True)
            pdf.image(orig_path, w=80)
 
            out_path = "reports/lung_screening_report.pdf"
            pdf.output(out_path)
            with open(out_path, "rb") as f:
                st.download_button("⬇ Download PDF Report", f, file_name="lung_screening_report.pdf",
                                    mime="application/pdf")
            st.success("Report generated.")