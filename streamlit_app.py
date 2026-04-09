import streamlit as st
import time
import tempfile

from src.predict import predict_emotion

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(
    page_title="Speech Emotion AI",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ==============================
# CUSTOM CSS
# ==============================
st.markdown("""
<style>
.big-title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
}
.sub-text {
    text-align: center;
    color: #9aa0a6;
    margin-bottom: 20px;
}
.result-box {
    background-color: #1f4f3a;
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    font-size: 20px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# ==============================
# HEADER
# ==============================
st.markdown('<div class="big-title">🎙️ Speech Emotion Recognition</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-text">Upload audio and detect human emotions using AI</div>', unsafe_allow_html=True)

st.divider()

st.markdown("🧠 **Model:** Random Forest (MFCC + Chroma + Mel Features)")

# ==============================
# FILE UPLOAD
# ==============================
uploaded_file = st.file_uploader("📂 Upload WAV file", type=["wav"])

# ==============================
# EMOJI + INFO MAP
# ==============================
emoji_map = {
    "happy": "😊",
    "sad": "😢",
    "angry": "😠",
    "neutral": "😐"
}

emotion_info = {
    "happy": "Positive emotion detected 😊",
    "sad": "Low mood detected 😢",
    "angry": "High intensity emotion detected 😠",
    "neutral": "Stable emotional state 😐"
}

# ==============================
# PREDICTION
# ==============================
if uploaded_file:
    st.audio(uploaded_file, format="audio/wav")

    if st.button("🚀 Predict Emotion"):
        try:
            start = time.time()

            # Save temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
                temp_file.write(uploaded_file.getvalue())
                temp_path = temp_file.name

            with st.spinner("🔍 Analyzing audio..."):
                result = predict_emotion(temp_path)

            end = time.time()
            latency = round(end - start, 2)

            if "emotion" in result:
                emotion = result["emotion"]
                confidence = result["confidence"]

                emoji = emoji_map.get(emotion.lower(), "🎭")

                # RESULT BOX
                st.markdown(
                    f"""
                    <div class="result-box">
                        {emoji} Predicted Emotion: {emotion.capitalize()} ({confidence}%)
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # CONFIDENCE
                st.markdown(f"**Confidence Score:** {confidence}%")
                st.progress(int(confidence))

                # INSIGHT
                st.info(emotion_info.get(emotion.lower(), "Emotion detected"))

                # LATENCY
                st.info(f"⏱️ Processing Time: {latency} sec")

            else:
                st.error("Prediction failed")

        except Exception as e:
            st.error(f"⚠️ Error: {e}")

# ==============================
# FOOTER
# ==============================
st.markdown("---")
st.caption("💡 Tip: Try different tones or speakers for varied results")
st.caption("Built with ❤️ using Machine Learning and Streamlit")