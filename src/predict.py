import os
import joblib
import numpy as np
import librosa

MODEL_PATH = "models/model.pkl"

# Load model safely
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    model = None
    print("⚠️ Model not found. Train first.")


def extract_features(file_path):
    """
    Extract MFCC, Chroma, and Mel features
    """
    audio, sr = librosa.load(file_path, duration=3, offset=0.5)

    mfcc = np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=audio, sr=sr).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(y=audio, sr=sr).T, axis=0)

    return np.hstack([mfcc, chroma, mel])


def predict_emotion(file_path):
    """
    Predict emotion + confidence
    """
    if model is None:
        return {
            "emotion": "Model not trained",
            "confidence": 0
        }

    features = extract_features(file_path)

    probs = model.predict_proba([features])[0]
    idx = np.argmax(probs)

    emotion = model.classes_[idx]
    confidence = round(probs[idx] * 100, 2)

    return {
        "emotion": emotion,
        "confidence": confidence
    }