import streamlit as st
import numpy as np
import json
import pickle
import random
from PIL import Image, UnidentifiedImageError
import os

# ─── Page Config ─────────────────────────────────────────
st.set_page_config(
    page_title="🌄 Scenic Vibe Detector",
    page_icon="🌄",
    layout="centered"
)

# ─── Constants ───────────────────────────────────────────
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
IMG_SIZE = 128

VIBE_EMOJI = {
    "urban_grit": "🏙️",
    "mystic_green": "🌿",
    "ethereal_frost": "❄️",
    "epic_solitude": "⛰️",
    "serene_horizon": "🌊",
    "wanderlust": "🗺️",
    "golden_hour": "🌅",
    "midnight_dream": "🌙",
}

# ─── Load Metadata Only (always safe) ─────────────────────
@st.cache_data
def load_meta():
    try:
        with open(os.path.join(MODEL_DIR, "vibe_meta.json")) as f:
            vibe_meta = json.load(f)
        return vibe_meta
    except Exception as e:
        st.error(f"Metadata load failed: {e}")
        return None

# ─── Load Models (optional) ──────────────────────────────
@st.cache_resource
def load_models():
    try:
        import tensorflow as tf

        classifier = tf.keras.models.load_model(
            os.path.join(MODEL_DIR, "vibe_classifier_best.h5")
        )
        quote_gen = tf.keras.models.load_model(
            os.path.join(MODEL_DIR, "quote_generator_best.h5")
        )

        with open(os.path.join(MODEL_DIR, "tokenizer.pkl"), "rb") as f:
            tokenizer = pickle.load(f)

        with open(os.path.join(MODEL_DIR, "label_map.json")) as f:
            label_map = json.load(f)

        return classifier, quote_gen, tokenizer, label_map

    except Exception as e:
        st.warning(f"⚠️ Models not loaded: {e}")
        return None

# ─── Image Preprocessing ─────────────────────────────────
def preprocess_image(img):
    try:
        img = img.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        arr = np.array(img, dtype=np.float32) / 255.0
        return np.expand_dims(arr, 0)
    except Exception:
        return None

# ─── Quote Generator (safe fallback) ─────────────────────
def fallback_quote(vibe, vibe_meta):
    try:
        return random.choice(vibe_meta["quote_corpus"][vibe])
    except Exception:
        return "A moment captured beyond words."

# ─── UI ─────────────────────────────────────────────────
st.title("🌄 Scenic Vibe Detector")
st.write("Upload a scenic image and get its vibe + quote ✨")

uploaded = st.file_uploader(
    "Upload image",
    type=["jpg", "jpeg", "png", "webp"]
)

vibe_meta = load_meta()
models = load_models()

if uploaded and vibe_meta:

    # ─── SAFE IMAGE LOAD ────────────────────────────────
    try:
        uploaded.seek(0)
        pil_img = Image.open(uploaded)
        pil_img.load()
        pil_img = pil_img.convert("RGB")

        st.image(pil_img, caption="Uploaded Image")

    except UnidentifiedImageError:
        st.error("❌ Invalid image file.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Image error: {e}")
        st.stop()

    # ─── INFERENCE ──────────────────────────────────────
    with st.spinner("Analyzing..."):

        if models:
            try:
                classifier, quote_gen, tokenizer, label_map = models

                arr = preprocess_image(pil_img)
                if arr is None:
                    raise ValueError("Image preprocessing failed")

                probs = classifier.predict(arr, verbose=0)[0]

                idx_to_vibe = label_map["idx_to_vibe"]
                top_idx = int(np.argmax(probs))

                vibe = idx_to_vibe[str(top_idx)]
                confidence = float(probs[top_idx])

            except Exception as e:
                st.warning(f"Inference failed: {e}")
                vibe = random.choice(vibe_meta["vibe_labels"])
                confidence = None
        else:
            vibe = random.choice(vibe_meta["vibe_labels"])
            confidence = None

        quote = fallback_quote(vibe, vibe_meta)

    # ─── DISPLAY RESULT ────────────────────────────────
    emoji = VIBE_EMOJI.get(vibe, "🎨")
    name = vibe.replace("_", " ").title()

    st.markdown(f"## {emoji} {name}")
    st.markdown(f"*{vibe_meta['vibe_descriptions'].get(vibe, '')}*")
    st.markdown(f"> {quote}")

    if confidence is not None:
        st.progress(confidence)
        st.write(f"Confidence: {confidence*100:.2f}%")

    # ─── REROLL ───────────────────────────────────────
    if st.button("🎲 New Quote"):
        st.markdown(f"> {fallback_quote(vibe, vibe_meta)}")

else:
    st.info("Upload an image to begin.")