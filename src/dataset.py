import os
import numpy as np
from src.preprocess import load_audio, extract_features

def load_dataset(data_path):
    """
    Load dataset from directory
    Expected structure:
    data/
        happy/
        sad/
        angry/
    """
    X = []
    y = []

    for emotion in os.listdir(data_path):
        emotion_path = os.path.join(data_path, emotion)

        if not os.path.isdir(emotion_path):
            continue

        for file in os.listdir(emotion_path):
            file_path = os.path.join(emotion_path, file)

            try:
                audio, sr = load_audio(file_path)
                features = extract_features(audio, sr)

                X.append(features)
                y.append(emotion)

            except Exception as e:
                print(f"Error processing {file_path}: {e}")

    return np.array(X), np.array(y)