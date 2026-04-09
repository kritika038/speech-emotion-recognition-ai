import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import tempfile

from src.predict import predict_emotion

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(page_title="Speech Emotion AI", layout="centered")

# ==============================
# HEADER
# ==============================
st.title("🎙️ Speech Emotion Recognition AI")
st.markdown("Deep Learning powered emotion detection from audio")

st.divider()

# ==============================
# MODEL SELECT
# ==============================
st.subheader("🧠 Model")
st.info("Using Production Model (Random Forest + Feature Engineering)")

# ==============================
# FILE UPLOAD
# ==============================
st.subheader("📂 Upload WAV file")
uploaded_file = st.file_uploader("Upload Audio", type=["wav"])

if uploaded_file is not None:

    # Save temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        temp_path = tmp.name

    # Audio player
    st.audio(uploaded_file, format="audio/wav")

    st.write("🔍 Processing audio...")

    result = predict_emotion(temp_path)

    if isinstance(result, str):
        st.error(result)

    else:
        emotion, confidence, probs, top_preds = result

        # ==============================
        # OUTPUT
        # ==============================
        st.success(f"🎯 Predicted Emotion: **{emotion}**")

        # Confidence level
        if confidence > 0.7:
            st.success(f"🔥 High Confidence: {confidence}")
        elif confidence > 0.5:
            st.warning(f"⚡ Medium Confidence: {confidence}")
        else:
            st.error(f"⚠️ Low Confidence: {confidence}")

        # ==============================
        # TOP PREDICTIONS
        # ==============================
        st.subheader("🔝 Top Predictions")
        for label, prob in top_preds:
            st.write(f"{label}: {round(prob, 2)}")

        # ==============================
        # DISTRIBUTION GRAPH
        # ==============================
        st.subheader("📊 Prediction Distribution")

        labels = list(range(len(probs)))

        fig, ax = plt.subplots()
        ax.bar(range(len(probs)), probs)
        ax.set_xticks(range(len(probs)))
        ax.set_xticklabels(["angry", "happy", "neutral", "sad"])
        ax.set_ylabel("Probability")

        st.pyplot(fig)

        # ==============================
        # INTERPRETATION
        # ==============================
        st.subheader("🧠 Interpretation")

        if confidence > 0.7:
            st.write("Model is highly confident about this emotion.")
        elif confidence > 0.5:
            st.write("Moderate confidence. Could vary slightly.")
        else:
            st.write("Low confidence. Try clearer or longer audio.")

# ==============================
# FOOTER
# ==============================
st.divider()

st.markdown("### 🧠 Model Comparison")
st.markdown("""
- Random Forest → Production optimized  
- CNN → Used during experimentation  
""")