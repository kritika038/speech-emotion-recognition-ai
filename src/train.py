import os
import numpy as np
import librosa
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt
import seaborn as sns

DATA_PATH = "data"
MODEL_PATH = "models/model.pkl"


# ==============================
# FEATURE EXTRACTION
# ==============================
def extract_features(file_path):
    audio, sr = librosa.load(file_path, duration=3, offset=0.5)

    mfcc = np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=audio, sr=sr).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(y=audio, sr=sr).T, axis=0)

    return np.hstack([mfcc, chroma, mel])


# ==============================
# LOAD DATASET
# ==============================
def load_dataset():
    X, y = [], []

    for emotion in os.listdir(DATA_PATH):
        emotion_path = os.path.join(DATA_PATH, emotion)

        if not os.path.isdir(emotion_path):
            continue

        for file in os.listdir(emotion_path):
            file_path = os.path.join(emotion_path, file)

            try:
                features = extract_features(file_path)
                X.append(features)
                y.append(emotion)
            except:
                continue

    return np.array(X), np.array(y)


# ==============================
# TRAIN
# ==============================
def train():
    print("Loading dataset...")
    X, y = load_dataset()

    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("Training model...")

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(n_estimators=300))
    ])

    model.fit(X_train, y_train)

    print("Evaluating model...")
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc}")

    # Save model
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved at {MODEL_PATH}")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)

    os.makedirs("outputs/graphs", exist_ok=True)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=model.classes_,
                yticklabels=model.classes_)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig("outputs/graphs/confusion_matrix.png")
    plt.close()

    print("Confusion matrix saved")


if __name__ == "__main__":
    train()