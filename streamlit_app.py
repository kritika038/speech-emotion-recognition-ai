import streamlit as st
import tempfile
import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

from src.predict import predict_emotion

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(
    page_title="Speech Emotion AI",
    page_icon="🎙️",
    layout="centered"
)

# ==============================
# CUSTOM CSS (UI UPGRADE 🔥)
# ==============================
st.markdown("""
<style>
.big-title {
    font-size:40px !important;
    font-weight:700;
}
.subtitle {
    font-size:18px;
    color:gray;
}
.box {
    padding:15px;
    border-radius:10px;
    background-color:#1e293b;
}
</style>
""", unsafe_allow_html=True)

# ==============================
# HEADER
# ==============================
st.markdown('<p class="big-title">🎙️ Speech Emotion Recognition AI</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Deep Learning powered emotion detection from audio</p>', unsafe_allow_html=True)

st.markdown("---")

# ==============================
# MODEL SELECTOR 🔥
# ==============================
model_choice = st.selectbox(
    "🧠 Select Model",
    ["CNN (Deep Learning)", "Random Forest (Baseline)"]
)

# ==============================
# FILE UPLOAD
# ==============================
uploaded_file = st.file_uploader("📂 Upload WAV file", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(uploaded_file.read())
            temp_path = tmp.name

        with st.spinner("🔍 Processing audio..."):

            if model_choice == "CNN (Deep Learning)":
                result = predict_emotion(temp_path)
            else:
                from src.predict_ml import predict_emotion_ml
                result = predict_emotion_ml(temp_path)

        # ==============================
        # RESULT DISPLAY
        # ==============================
        if isinstance(result, tuple):
            emotion, confidence, probabilities, top_preds = result

            st.success(f"🎯 Predicted Emotion: {emotion}")
            st.info(f"📊 Confidence: {confidence:.2f}")

            # Top predictions
            st.subheader("🔝 Top Predictions")
            for label, prob in top_preds:
                st.write(f"{label}: {prob:.2f}")

            # ==============================
            # PROBABILITY GRAPH
            # ==============================
            st.subheader("📊 Prediction Distribution")

            labels = ["happy", "sad", "angry", "neutral"]

            fig, ax = plt.subplots()
            ax.bar(labels, probabilities)
            ax.set_ylim(0, 1)

            st.pyplot(fig)

            # ==============================
            # AUDIO WAVEFORM
            # ==============================
            st.subheader("🎧 Audio Waveform")

            audio, sr = librosa.load(temp_path)

            fig2, ax2 = plt.subplots()
            librosa.display.waveshow(audio, sr=sr, ax=ax2)

            st.pyplot(fig2)

        else:
            st.error(result)

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")

    finally:
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.remove(temp_path)

# ==============================
# FOOTER INFO
# ==============================
st.markdown("---")

st.subheader("🧠 Model Comparison")
st.write("""
- Random Forest → Baseline (~74% accuracy)  
- CNN → Improved performance + generalization  
""")

st.subheader("⚙️ System Info")
st.write("""
- Input: Audio (.wav)  
- Feature: Mel Spectrogram  
- Model: CNN + ML  
- Output: Emotion + Confidence + Distribution  
""")

st.caption("Built with ❤️ using Deep Learning + Streamlit")