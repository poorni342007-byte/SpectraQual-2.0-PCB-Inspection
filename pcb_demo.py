
# ============================================================
# SpectraQual 2.0 — PCB Defect Detection Demo
# Run with: streamlit run pcb_demo.py
# Requirements: pip install streamlit torch torchvision pillow opencv-python-headless numpy
# ============================================================

import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image, ImageFilter
import numpy as np
import cv2
import io
import random

# ── Page config ─────────────────────────────────────────────
st.set_page_config(
    page_title="SpectraQual 2.0 PCB Demo",
    page_icon="🔬",
    layout="centered",
)

# ── Custom CSS — industrial‑green‑on‑dark aesthetic ─────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Exo+2:wght@300;600;800&display=swap');

:root {
    --green:  #00ff9d;
    --red:    #ff3c5a;
    --amber:  #ffb800;
    --bg:     #0b0f0e;
    --panel:  #111918;
    --border: #1f2e2a;
}

html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg) !important;
    color: #d4f0e8;
    font-family: 'Exo 2', sans-serif;
}

/* Hide default Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }

/* Hero title */
.hero {
    text-align: center;
    padding: 2.2rem 0 1rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 1.8rem;
}
.hero h1 {
    font-family: 'Exo 2', sans-serif;
    font-weight: 800;
    font-size: 2.4rem;
    letter-spacing: -0.5px;
    margin: 0;
    background: linear-gradient(120deg, var(--green) 0%, #00d4ff 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero p {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.78rem;
    color: #557a6a;
    margin: 0.3rem 0 0;
    letter-spacing: 2px;
    text-transform: uppercase;
}

/* Result boxes */
.result-good {
    background: #031a0f;
    border: 2px solid var(--green);
    border-radius: 10px;
    padding: 1.2rem 1.6rem;
    text-align: center;
    box-shadow: 0 0 24px #00ff9d33;
}
.result-bad {
    background: #1a030a;
    border: 2px solid var(--red);
    border-radius: 10px;
    padding: 1.2rem 1.6rem;
    text-align: center;
    box-shadow: 0 0 24px #ff3c5a33;
}
.result-label {
    font-family: 'Share Tech Mono', monospace;
    font-size: 1.55rem;
    font-weight: 700;
    letter-spacing: 1px;
}
.result-conf {
    font-size: 0.82rem;
    opacity: 0.7;
    margin-top: 0.25rem;
    font-family: 'Share Tech Mono', monospace;
    letter-spacing: 1px;
}
.defect-tag {
    display: inline-block;
    background: #2a0a10;
    border: 1px solid var(--red);
    border-radius: 4px;
    color: var(--red);
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.72rem;
    padding: 2px 8px;
    margin: 3px 2px;
    letter-spacing: 1px;
}
.good-tag {
    display: inline-block;
    background: #021a0e;
    border: 1px solid var(--green);
    border-radius: 4px;
    color: var(--green);
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.72rem;
    padding: 2px 8px;
    margin: 3px 2px;
    letter-spacing: 1px;
}

/* Stacked info row */
.info-row {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.76rem;
    color: #557a6a;
    display: flex;
    justify-content: space-between;
    border-top: 1px solid var(--border);
    padding-top: 0.6rem;
    margin-top: 1rem;
}

/* Streamlit widget overrides */
[data-testid="stFileUploader"],
[data-testid="stCameraInput"],
div[data-baseweb="select"] {
    background: var(--panel) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
}
button[kind="primary"], .stButton > button {
    background: var(--green) !important;
    color: #000 !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-weight: 700 !important;
    letter-spacing: 1px !important;
    border: none !important;
    border-radius: 6px !important;
    padding: 0.5rem 2rem !important;
}
button[kind="primary"]:hover, .stButton > button:hover {
    background: #00d48a !important;
}
[data-testid="stCheckbox"] label {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.82rem;
    color: #7ab8a0;
}
.stSlider label, .stCheckbox label {
    color: #7ab8a0 !important;
}
.section-label {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.72rem;
    color: #557a6a;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
}
</style>
""", unsafe_allow_html=True)


# ── Hero ─────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>SpectraQual 2.0</h1>
  <p>PCB Visual Inspection Demo · MobileNetV2 · Transfer Learning</p>
</div>
""", unsafe_allow_html=True)


# ── Model (cached so it loads only once) ─────────────────────
@st.cache_resource
def load_model():
    """
    Load MobileNetV2 with ImageNet weights, freeze the backbone,
    and attach a small binary classification head (Good / Defective).
    In a real project you would fine-tune this on actual PCB images.
    """
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)

    # Freeze every parameter in the backbone
    for param in model.parameters():
        param.requires_grad = False

    # Replace the final classifier with a 2-class head
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(in_features, 2),
    )

    model.eval()
    return model

try:
    model = load_model()
    st.sidebar.success("✅ Model loaded (MobileNetV2)")
except Exception as e:
    st.error(f"Model failed to load: {e}")
    st.stop()


# ── ImageNet normalisation transform ────────────────────────
imagenet_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# ── Optional OpenCV pre-processing ───────────────────────────
def preprocess_pcb(pil_img: Image.Image, edge_enhance: bool, grayscale: bool) -> Image.Image:
    """
    Apply optional PCB-friendly filters:
      - Grayscale → helps cut board-shine colour casts
      - Edge enhance → sharpens solder-joint boundaries
    Returns a PIL Image always in RGB mode.
    """
    img = np.array(pil_img.convert("RGB"))

    if grayscale:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    if edge_enhance:
        # Laplacian edge layer blended back for sharpness
        gray_for_edge = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        edges = cv2.Laplacian(gray_for_edge, cv2.CV_64F)
        edges = np.uint8(np.clip(np.abs(edges), 0, 255))
        edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        img = cv2.addWeighted(img, 0.85, edges_rgb, 0.15, 0)

    return Image.fromarray(img)


# ── Common PCB defect labels by confidence bucket ────────────
DEFECT_LABELS = {
    "high":   ["Short Circuit", "Missing Component"],
    "medium": ["Solder Bridge", "Cold Joint", "Lifted Pad"],
    "low":    ["Possible Contamination", "Marginal Solder"],
}

def pick_defect_labels(conf: float) -> list[str]:
    """Return plausible defect names based on confidence level."""
    if conf >= 0.80:
        return DEFECT_LABELS["high"]
    elif conf >= 0.65:
        return DEFECT_LABELS["medium"][:2]
    else:
        return [random.choice(DEFECT_LABELS["low"])]


# ── Inference helper ──────────────────────────────────────────
def run_inference(pil_img: Image.Image) -> tuple[str, float, torch.Tensor]:
    """
    Run a forward pass and return (label, confidence, raw_logits).
    Because the head is NOT fine-tuned, we inject a small random
    perturbation to simulate plausible demo outputs; in production
    you would remove this and use real fine-tuned weights.
    """
    tensor = imagenet_transform(pil_img).unsqueeze(0)  # [1, 3, 224, 224]

    with torch.no_grad():
        logits = model(tensor)                          # [1, 2]

    # ── Demo bias ──────────────────────────────────────────
    # Without real PCB training data the head is random.
    # We nudge the output so the demo feels realistic.
    # Remove / replace with your fine-tuned weights in production.
    noise = torch.randn_like(logits) * 0.6
    logits = logits + noise
    # ───────────────────────────────────────────────────────

    probs = torch.softmax(logits, dim=1)[0]             # [2]
    pred_idx = int(torch.argmax(probs).item())
    confidence = float(probs[pred_idx].item())
    label = "Good" if pred_idx == 0 else "Defective"
    return label, confidence, probs


# ── Sidebar: preprocessing options ───────────────────────────
st.sidebar.markdown("### ⚙️ Pre-processing")
use_grayscale    = st.sidebar.checkbox("Grayscale (reduce shine)", value=False)
use_edge_enhance = st.sidebar.checkbox("Edge enhance (sharpness)", value=True)
st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ About")
st.sidebar.info(
    "Backbone: MobileNetV2 (ImageNet)\n"
    "Head: 2-class linear (untrained demo)\n"
    "Threshold: 0.50 · Classes: Good / Defective"
)


# ── Input section ─────────────────────────────────────────────
st.markdown('<p class="section-label">01 · Input Source</p>', unsafe_allow_html=True)
use_webcam = st.checkbox("📷  Use webcam instead of file upload", value=False)

uploaded_image = None

if use_webcam:
    cam_photo = st.camera_input("Point camera at your PCB printout")
    if cam_photo:
        uploaded_image = Image.open(cam_photo)
else:
    file = st.file_uploader(
        "Upload a PCB photo (JPG / PNG)",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )
    if file:
        uploaded_image = Image.open(file)


# ── Predict ───────────────────────────────────────────────────
if uploaded_image is not None:
    st.markdown("---")
    st.markdown('<p class="section-label">02 · Preview</p>', unsafe_allow_html=True)

    try:
        # Apply optional CV filtering
        processed_img = preprocess_pcb(uploaded_image, use_edge_enhance, use_grayscale)
        st.image(processed_img, use_container_width=True, caption="Processed input (224 px preview)")
    except Exception as e:
        st.error(f"Image processing error: {e}")
        st.stop()

    st.markdown("---")
    st.markdown('<p class="section-label">03 · Run Inspection</p>', unsafe_allow_html=True)

    if st.button("🔬  Analyse PCB"):
        with st.spinner("Running inference …"):
            try:
                label, confidence, probs = run_inference(processed_img)
            except Exception as e:
                st.error(f"Inference error: {e}")
                st.stop()

        # ── Result card ───────────────────────────────────────
        pct = confidence * 100
        is_defective = label == "Defective"
        css_class = "result-bad" if is_defective else "result-good"
        icon = "❌" if is_defective else "✅"
        color_hex = "#ff3c5a" if is_defective else "#00ff9d"

        # Build tag HTML
        if is_defective:
            defect_names = pick_defect_labels(confidence)
            tags_html = " ".join(
                f'<span class="defect-tag">{d}</span>' for d in defect_names
            )
        else:
            tags_html = '<span class="good-tag">PASS · No defects detected</span>'

        good_pct = float(probs[0].item()) * 100
        bad_pct  = float(probs[1].item()) * 100

        st.markdown(f"""
        <div class="{css_class}">
            <div class="result-label" style="color:{color_hex}">
                {icon} {label} &nbsp;·&nbsp; {pct:.1f}% confidence
            </div>
            <div class="result-conf">
                {'⚠️ Defects flagged:' if is_defective else ''}
                {tags_html}
            </div>
            <div class="info-row">
                <span>GOOD: {good_pct:.1f}%</span>
                <span>DEFECTIVE: {bad_pct:.1f}%</span>
                <span>MODEL: MobileNetV2</span>
                <span>IMG: 224×224</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Confidence bar ────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        st.progress(confidence, text=f"Confidence: {pct:.1f}%")

        # ── Raw probability expander ──────────────────────────
        with st.expander("🔢 Raw class probabilities"):
            col1, col2 = st.columns(2)
            col1.metric("Good",      f"{good_pct:.2f}%")
            col2.metric("Defective", f"{bad_pct:.2f}%")
            st.caption(
                "Note: The binary head is un-fine-tuned in this demo. "
                "Replace the model weights with your trained checkpoint for production accuracy."
            )
else:
    # Placeholder when no image yet
    st.info("⬆️  Upload a photo or enable the webcam to begin inspection.")

# ── Footer ────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; margin-top:3rem; font-family:'Share Tech Mono',monospace;
            font-size:0.68rem; color:#2a4a3e; letter-spacing:1px;">
  SPECTRAQUAL 2.0 · DEMO MODE · NO REAL PCB TRAINING DATA · FOR PROTOTYPING ONLY
</div>
""", unsafe_allow_html=True)

