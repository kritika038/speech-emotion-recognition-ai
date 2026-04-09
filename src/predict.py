import os
import numpy as np
import librosa
import joblib

# ==============================
# CONFIG
# ==============================
MODEL_PATH = "models/model.pkl"

# ==============================
# LOAD MODEL
# ==============================
if not os.path.exists(MODEL_PATH):
    raise Exception("❌ Model file not found. Run training first.")

model = joblib.load(MODEL_PATH)


# ==============================
# FEATURE EXTRACTION (MATCH TRAINING)
# ==============================
def extract_features(file_path):
    try:
        audio, sr = librosa.load(file_path, duration=3, offset=0.5)

        # MFCC (40)
        mfcc = np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40).T, axis=0)

        # Chroma (12)
        stft = np.abs(librosa.stft(audio))
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sr).T, axis=0)

        # Mel Spectrogram (~128)
        mel = np.mean(librosa.feature.melspectrogram(y=audio, sr=sr).T, axis=0)

        # Combine → ~180 features
        features = np.hstack([mfcc, chroma, mel])

        return features.reshape(1, -1)

    except Exception as e:
        raise Exception(f"Feature extraction error: {e}")


# ==============================
# PREDICTION
# ==============================
def predict_emotion(file_path):
    try:
        features = extract_features(file_path)

        # Prediction (returns STRING label)
        prediction = model.predict(features)[0]

        # Probabilities
        probabilities = model.predict_proba(features)[0]

        # Confidence
        confidence = float(np.max(probabilities))
        confidence = round(confidence, 2)

        # Top 2 predictions
        top_indices = np.argsort(probabilities)[-2:][::-1]
        top_predictions = [
            (model.classes_[i], float(probabilities[i]))
            for i in top_indices
        ]

        return (
            prediction,          # emotion (string)
            confidence,          # confidence score
            probabilities,       # full distribution
            top_predictions      # top 2 predictions
        )

    except Exception as e:
        return f"Prediction Error: {str(e)}"