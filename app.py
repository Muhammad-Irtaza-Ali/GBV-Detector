import json
from pathlib import Path

import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parent
#MODEL_DIR = ROOT / "GBV_BERT"
MODEL_NAME = "muhammadirtazaali/gbv-bert-urdu"
DATASET_PATH = ROOT / "GBV_detect_Urdu.json"

st.set_page_config(
    page_title="GBV Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.session_state.setdefault("draft_text", "")


@st.cache_resource(show_spinner="Loading model...")
def load_model_components():
    # tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    # model = AutoModelForSequenceClassification.from_pretrained(
    #     MODEL_DIR,
    #     local_files_only=True,
    # )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    with DATASET_PATH.open("r", encoding="utf-8") as handle:
        rows = json.load(handle)

    if hasattr(model.config, "id2label") and model.config.id2label:
        labels = [
            model.config.id2label.get(i, f"class_{i}")
            for i in range(model.config.num_labels)
        ]
    else:
        labels = sorted({row.get("label") for row in rows if row.get("label")})

    return tokenizer, model, labels


@st.cache_data(show_spinner=False)
def get_examples():
    with DATASET_PATH.open("r", encoding="utf-8") as handle:
        rows = json.load(handle)

    examples = []
    for row in rows[:6]:
        text = (row.get("text") or "").strip()
        if text:
            examples.append((row.get("label", "unknown"), text))
    return examples


def get_readable_label(label_idx, label_name):
    if label_idx == 0:
        return "Safe Content"
    return "GBV-Related Risk"


def predict_label(text, tokenizer, model, labels):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=256,
        padding=True,
    )

    with torch.inference_mode():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=-1)[0].cpu().tolist()
    results = []
    for index, prob in enumerate(probs):
        readable_label = get_readable_label(
            index, labels[index] if index < len(labels) else f"class_{index}"
        )
        results.append(
            {
                "label": readable_label,
                "raw_label": labels[index] if index < len(labels) else f"class_{index}",
                "label_idx": index,
                "confidence": round(float(prob) * 100, 1),
            }
        )
    results.sort(key=lambda item: item["confidence"], reverse=True)
    return results


# ── Professional, compact CSS ──────────────────────────────────────────────
custom_css = """
<style>
    /* Reduce default Streamlit padding */
    .block-container {
        padding-top: 1.2rem !important;
        padding-bottom: 1.5rem !important;
        max-width: 1100px;
    }

    /* Hero – tighter */
    .hero {
        background: linear-gradient(135deg, #1e3a8a 0%, #1e40af 50%, #0f766e 100%);
        border-radius: 14px;
        padding: 1.4rem 1.8rem;
        color: white;
        margin-bottom: 1.4rem;
        box-shadow: 0 4px 18px rgba(30, 58, 138, 0.25);
    }
    .hero h1 {
        font-size: 1.65rem;
        font-weight: 700;
        margin: 0 0 0.35rem 0;
        letter-spacing: -0.02em;
    }
    .hero p {
        font-size: 0.92rem;
        opacity: 0.92;
        margin: 0;
        line-height: 1.45;
    }

    /* Label key – compact pills */
    .label-key {
        display: flex;
        gap: 0.75rem;
        margin-top: 0.9rem;
        flex-wrap: wrap;
    }
    .label-pill {
        background: rgba(255,255,255,0.15);
        border: 1px solid rgba(255,255,255,0.28);
        border-radius: 8px;
        padding: 0.35rem 0.75rem;
        font-size: 0.82rem;
        font-weight: 500;
    }

    /* Cards */
    .card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.1rem 1.25rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        margin-bottom: 1rem;
    }
    .card h4 {
        font-size: 0.95rem;
        font-weight: 600;
        color: #1e293b;
        margin: 0 0 0.75rem 0;
    }

    /* Result cards – compact */
    .result-gbv {
        background: #fef2f2;
        border-left: 4px solid #dc2626;
        border-radius: 10px;
        padding: 1rem 1.15rem;
        margin: 0.9rem 0;
    }
    .result-safe {
        background: #f0fdf4;
        border-left: 4px solid #16a34a;
        border-radius: 10px;
        padding: 1rem 1.15rem;
        margin: 0.9rem 0;
    }
    .result-title {
        font-size: 1.15rem;
        font-weight: 700;
        margin: 0 0 0.25rem 0;
        color: #0f172a;
    }
    .result-sub {
        font-size: 0.88rem;
        color: #475569;
        margin: 0 0 0.5rem 0;
    }
    .conf-badge {
        display: inline-block;
        font-size: 0.8rem;
        font-weight: 600;
        padding: 0.25rem 0.7rem;
        border-radius: 999px;
    }
    .conf-gbv { background: #fee2e2; color: #991b1b; }
    .conf-safe { background: #dcfce7; color: #166534; }

    /* Progress / score rows */
    .score-row {
        margin-bottom: 0.55rem;
    }

    /* Text area */
    .stTextArea textarea {
        border-radius: 10px !important;
        border: 1px solid #cbd5e1 !important;
        font-size: 0.95rem !important;
        min-height: 140px !important;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        padding: 0.45rem 1rem !important;
        height: auto !important;
    }

    /* Example buttons – smaller */
    div[data-testid="stVerticalBlock"] .stButton > button {
        font-size: 0.82rem !important;
        padding: 0.35rem 0.7rem !important;
    }

    /* Captions under examples */
    .stCaption {
        font-size: 0.78rem !important;
        color: #64748b !important;
        margin-top: -0.35rem !important;
        margin-bottom: 0.6rem !important;
        line-height: 1.35 !important;
    }

    /* About section */
    .about-text {
        font-size: 0.85rem;
        color: #475569;
        line-height: 1.55;
    }

    /* Hide Streamlit branding for cleaner look */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
"""

st.markdown(custom_css, unsafe_allow_html=True)

# ── Hero ───────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="hero">
        <h1>🛡️ GBV Risk Detection</h1>
        <p>On-device AI classifier for detecting Gender-Based Violence content in Urdu text.</p>
        <div class="label-key">
            <div class="label-pill">Label 0 → Safe Content</div>
            <div class="label-pill">Label 1 → GBV-Related Risk</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Main layout ────────────────────────────────────────────────────────────
col1, col2 = st.columns([1.7, 1], gap="medium")

with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    text_input = st.text_area(
        "Urdu text to analyze",
        height=160,
        value=st.session_state.get("draft_text", ""),
        placeholder="Paste or type Urdu text here…",
        label_visibility="collapsed",
    )
    st.session_state["draft_text"] = text_input
    st.markdown("</div>", unsafe_allow_html=True)

    analyze = st.button("Analyze Text", use_container_width=True, type="primary")

    if analyze:
        if text_input.strip():
            tokenizer, model, labels = load_model_components()
            with st.spinner("Analyzing…"):
                results = predict_label(text_input, tokenizer, model, labels)

            top = results[0]
            st.success("Analysis complete", icon="✅")

            if top["label_idx"] == 1:
                st.markdown(
                    f"""
                    <div class="result-gbv">
                        <div class="result-title">⚠️ {top['label']}</div>
                        <p class="result-sub">This content appears to be <strong>GBV-related</strong>.</p>
                        <span class="conf-badge conf-gbv">{top['confidence']}% confidence</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div class="result-safe">
                        <div class="result-title">✅ {top['label']}</div>
                        <p class="result-sub">This content appears to be <strong>safe</strong>.</p>
                        <span class="conf-badge conf-safe">{top['confidence']}% confidence</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("<h4>Category scores</h4>", unsafe_allow_html=True)
            for item in results:
                st.progress(
                    item["confidence"] / 100.0,
                    text=f"{item['label']}  ·  {item['confidence']}%",
                )
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.warning("Please enter some text before analyzing.")

with col2:
    # Quick examples
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("<h4>Quick examples</h4>", unsafe_allow_html=True)
    examples = get_examples()
    for idx, (label, example) in enumerate(examples):
        short_label = (label[:18] + "…") if len(label) > 18 else label
        if st.button(short_label, key=f"ex-{idx}", use_container_width=True):
            st.session_state["draft_text"] = example
            st.rerun()
        st.caption(example[:90] + ("…" if len(example) > 90 else ""))
    st.markdown("</div>", unsafe_allow_html=True)

    # About
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("<h4>About</h4>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="about-text">
            <strong>Model</strong> · XLM-RoBERTa fine-tuned on Urdu GBV data<br>
            <strong>Type</strong> · Binary (GBV vs Non-GBV)<br>
            <strong>Inference</strong> · Fully local · no data leaves the device<br>
            <strong>Privacy</strong> · On-device only
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)