import streamlit as st
import numpy as np
import json
import pickle
import random
from PIL import Image
import os

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🌄 Scenic Vibe Detector",
    page_icon="🌄",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .vibe-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        border-radius: 16px;
        padding: 24px;
        margin: 16px 0;
        border: 1px solid #e94560;
        text-align: center;
    }
    .vibe-title {
        font-size: 2rem;
        font-weight: 800;
        color: #e94560;
        text-transform: uppercase;
        letter-spacing: 4px;
        margin-bottom: 8px;
    }
    .vibe-description {
        font-size: 1rem;
        color: #a8b2d8;
        font-style: italic;
        margin-bottom: 16px;
    }
    .quote-box {
        background: rgba(233, 69, 96, 0.1);
        border-left: 4px solid #e94560;
        padding: 16px 20px;
        border-radius: 0 8px 8px 0;
        font-size: 1.1rem;
        color: #ccd6f6;
        font-style: italic;
    }
    .confidence-label {
        color: #64ffda;
        font-weight: 600;
        font-size: 0.9rem;
    }
    .stProgress > div > div { background-color: #e94560; }
    h1 { color: #ccd6f6 !important; }
    .subtitle { color: #8892b0; text-align: center; margin-top: -10px; }
</style>
""", unsafe_allow_html=True)

# ─── Load Assets ────────────────────────────────────────────────────────────
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

@st.cache_resource
def load_models():
    """Load TensorFlow models and metadata."""
    import tensorflow as tf

    # Load classifier
    classifier_path = os.path.join(MODEL_DIR, "vibe_classifier_best.h5")
    classifier = tf.keras.models.load_model(classifier_path)

    # Load quote generator
    quote_gen_path = os.path.join(MODEL_DIR, "quote_generator_best.h5")
    quote_gen = tf.keras.models.load_model(quote_gen_path)

    # Load tokenizer
    with open(os.path.join(MODEL_DIR, "tokenizer.pkl"), "rb") as f:
        tokenizer = pickle.load(f)

    # Load metadata
    with open(os.path.join(MODEL_DIR, "label_map.json")) as f:
        label_map = json.load(f)

    with open(os.path.join(MODEL_DIR, "vibe_meta.json")) as f:
        vibe_meta = json.load(f)

    return classifier, quote_gen, tokenizer, label_map, vibe_meta


@st.cache_data
def load_meta_only():
    """Load only metadata (no TF) for fast startup preview."""
    with open(os.path.join(MODEL_DIR, "label_map.json")) as f:
        label_map = json.load(f)
    with open(os.path.join(MODEL_DIR, "vibe_meta.json")) as f:
        vibe_meta = json.load(f)
    return label_map, vibe_meta


# ─── Inference Helpers ───────────────────────────────────────────────────────
IMG_SIZE = 128

def preprocess_image(pil_img: Image.Image) -> np.ndarray:
    img = pil_img.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, 0)


def generate_quote_nn(quote_gen, tokenizer, vibe_label: str, vibe_meta: dict, max_seq_len: int = 12) -> str:
    """Greedy-decode a quote from the quote generator model."""
    import tensorflow as tf

    vibe_labels = vibe_meta["vibe_labels"]
    vibe_idx = vibe_labels.index(vibe_label)
    vibe_one_hot = np.zeros((1, len(vibe_labels)), dtype=np.float32)
    vibe_one_hot[0, vibe_idx] = 1.0

    word_index = tokenizer.word_index
    index_word = {v: k for k, v in word_index.items()}

    sos_token = word_index.get("<sos>", 1)
    eos_token = word_index.get("<eos>", 2)

    generated = [sos_token]
    for _ in range(max_seq_len):
        seq = tf.keras.preprocessing.sequence.pad_sequences(
            [generated], maxlen=max_seq_len, padding="pre"
        )
        preds = quote_gen.predict([vibe_one_hot, seq], verbose=0)
        next_token = int(np.argmax(preds[0, -1]))
        if next_token == eos_token or next_token == 0:
            break
        generated.append(next_token)

    words = [index_word.get(t, "") for t in generated[1:]]
    quote = " ".join(w for w in words if w).capitalize()
    return quote if quote else random.choice(vibe_meta["quote_corpus"][vibe_label])


def fallback_quote(vibe_label: str, vibe_meta: dict) -> str:
    return random.choice(vibe_meta["quote_corpus"][vibe_label])


# ─── Vibe Emoji Map ──────────────────────────────────────────────────────────
VIBE_EMOJI = {
    "urban_grit":      "🏙️",
    "mystic_green":    "🌿",
    "ethereal_frost":  "❄️",
    "epic_solitude":   "⛰️",
    "serene_horizon":  "🌊",
    "wanderlust":      "🗺️",
    "golden_hour":     "🌅",
    "midnight_dream":  "🌙",
}

# ─── UI ──────────────────────────────────────────────────────────────────────
st.markdown("# 🌄 Scenic Vibe Detector")
st.markdown('<p class="subtitle">Upload a landscape photo — discover its vibe & receive a custom quote</p>', unsafe_allow_html=True)
st.markdown("---")

uploaded = st.file_uploader(
    "Drop your scenic image here",
    type=["jpg", "jpeg", "png", "webp"],
    help="Best results with landscapes: mountains, forests, cities, oceans, streets",
)

if uploaded:
    pil_img = Image.open(uploaded)
    st.image(pil_img, caption="Your uploaded scene", use_container_width=True)

    with st.spinner("✨ Detecting vibe..."):
        try:
            classifier, quote_gen, tokenizer, label_map, vibe_meta = load_models()
            models_loaded = True
        except Exception as e:
            st.warning(f"⚠️ Could not load TF models ({e}). Using fallback quote mode.")
            models_loaded = False
            _, vibe_meta = load_meta_only()
            label_map = vibe_meta  # has vibe_labels

        if models_loaded:
            arr = preprocess_image(pil_img)
            probs = classifier.predict(arr, verbose=0)[0]
            idx_to_vibe = label_map["idx_to_vibe"]
            top_idx = int(np.argmax(probs))
            vibe_label = idx_to_vibe[str(top_idx)]
            confidence = float(probs[top_idx])

            # Try NN quote, fallback to corpus
            try:
                quote = generate_quote_nn(quote_gen, tokenizer, vibe_label, vibe_meta)
            except Exception:
                quote = fallback_quote(vibe_label, vibe_meta)
        else:
            # Purely random fallback when models aren't available
            vibe_label = random.choice(vibe_meta["vibe_labels"])
            confidence = None
            quote = fallback_quote(vibe_label, vibe_meta)

    # ── Result Card ────────────────────────────────────────────────────────
    emoji = VIBE_EMOJI.get(vibe_label, "🎨")
    description = vibe_meta["vibe_descriptions"].get(vibe_label, "")
    display_name = vibe_label.replace("_", " ").title()

    st.markdown(f"""
    <div class="vibe-card">
        <div class="vibe-title">{emoji} {display_name}</div>
        <div class="vibe-description">{description}</div>
        <div class="quote-box">"{quote}"</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Confidence bars ────────────────────────────────────────────────────
    if models_loaded:
        st.markdown("#### 📊 Vibe Confidence Scores")
        sorted_vibes = sorted(
            [(idx_to_vibe[str(i)], float(probs[i])) for i in range(len(probs))],
            key=lambda x: x[1], reverse=True
        )
        for v, p in sorted_vibes:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f'<span class="confidence-label">{VIBE_EMOJI.get(v,"🎨")} {v.replace("_"," ").title()}</span>', unsafe_allow_html=True)
                st.progress(p)
            with col2:
                st.markdown(f"**{p*100:.1f}%**")

    # ── Re-roll quote ──────────────────────────────────────────────────────
    st.markdown("---")
    if st.button("🎲 Get a different quote"):
        new_quote = fallback_quote(vibe_label, vibe_meta)
        st.markdown(f'<div class="quote-box">"{new_quote}"</div>', unsafe_allow_html=True)

else:
    st.markdown("""
    ### How it works
    1. **Upload** any scenic landscape photo
    2. **Model 1** (MobileNetV2) classifies the scene into one of 8 vibes
    3. **Model 2** (sequence model) generates a unique quote for that vibe
    4. Enjoy your personalised vibe card ✨

    **The 8 Vibes:** 🏙️ Urban Grit · 🌿 Mystic Green · ❄️ Ethereal Frost · ⛰️ Epic Solitude · 🌊 Serene Horizon · 🗺️ Wanderlust · 🌅 Golden Hour · 🌙 Midnight Dream
    """)
