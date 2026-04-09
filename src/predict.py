import os
import numpy as np
import librosa
from tensorflow.keras.models import load_model

# ==============================
# CONFIG
# ==============================
MODEL_PATH = "models/cnn_model.keras"
IMG_SIZE = 128
EMOTIONS = ["happy", "sad", "angry", "neutral"]

# ==============================
# LOAD MODEL
# ==============================
if not os.path.exists(MODEL_PATH):
    raise Exception("❌ CNN model not found. Train using train_cnn.py")

model = load_model(MODEL_PATH)


# ==============================
# FEATURE EXTRACTION
# ==============================
def extract_spectrogram(file_path):
    audio, sr = librosa.load(file_path, duration=3, offset=0.5)

    spec = librosa.feature.melspectrogram(y=audio, sr=sr)
    spec = librosa.power_to_db(spec)

    spec = np.resize(spec, (IMG_SIZE, IMG_SIZE))

    # Normalize safely
    if np.max(spec) != 0:
        spec = spec / np.max(spec)

    # Add channel + batch dimension
    spec = spec[..., np.newaxis]
    spec = np.expand_dims(spec, axis=0)

    return spec


# ==============================
# PREDICTION
# ==============================
def predict_emotion(file_path):
    try:
        spec = extract_spectrogram(file_path)

        prediction = model.predict(spec)
        predicted_class = np.argmax(prediction)

        confidence = float(np.max(prediction))
        confidence = round(confidence, 2)

        # Top 2 predictions
        top_indices = np.argsort(prediction[0])[-2:][::-1]
        top_predictions = [
            (EMOTIONS[i], float(prediction[0][i]))
            for i in top_indices
        ]

        return EMOTIONS[predicted_class], confidence, prediction[0], top_predictions

    except Exception as e:
        return f"Prediction Error: {str(e)}"