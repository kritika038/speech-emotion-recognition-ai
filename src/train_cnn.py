import os
import numpy as np
import librosa
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping


# ==============================
# CONFIG
# ==============================
DATA_PATH = "data"
EMOTIONS = ["happy", "sad", "angry", "neutral"]
IMG_SIZE = 128


# ==============================
# FEATURE EXTRACTION
# ==============================
def extract_spectrogram(file_path):
    """
    Convert audio to Mel Spectrogram
    """
    audio, sr = librosa.load(file_path, duration=3, offset=0.5)

    spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr)
    spectrogram = librosa.power_to_db(spectrogram)

    # Resize to fixed size
    spectrogram = np.resize(spectrogram, (IMG_SIZE, IMG_SIZE))

    return spectrogram


# ==============================
# LOAD DATA
# ==============================
def load_data():
    X, y = [], []

    for idx, emotion in enumerate(EMOTIONS):
        folder = os.path.join(DATA_PATH, emotion)

        if not os.path.exists(folder):
            print(f"⚠️ Missing folder: {folder}")
            continue

        for file in os.listdir(folder):
            path = os.path.join(folder, file)

            try:
                spec = extract_spectrogram(path)
                X.append(spec)
                y.append(idx)
            except Exception as e:
                print(f"Error: {path} -> {e}")

    X = np.array(X)
    y = to_categorical(y, num_classes=len(EMOTIONS))

    # Normalize data
    X = X / np.max(X)

    # Add channel dimension
    X = X[..., np.newaxis]

    return X, y


# ==============================
# MODEL
# ==============================
def build_model():
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
        BatchNormalization(),
        MaxPooling2D((2,2)),

        Conv2D(64, (3,3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2,2)),

        Conv2D(128, (3,3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2,2)),

        Flatten(),

        Dense(128, activation='relu'),
        Dropout(0.5),

        Dense(64, activation='relu'),
        Dropout(0.3),

        Dense(len(EMOTIONS), activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


# ==============================
# TRAINING
# ==============================
def train():
    print("📊 Loading data...")
    X, y = load_data()

    print("🔀 Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("🧠 Building model...")
    model = build_model()

    # Early stopping to prevent overfitting
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )

    print("🚀 Training model...")
    history = model.fit(
        X_train, y_train,
        epochs=30,
        batch_size=16,
        validation_data=(X_test, y_test),
        callbacks=[early_stop]
    )

    # ==============================
    # SAVE MODEL
    # ==============================
    os.makedirs("models", exist_ok=True)
    model.save("models/cnn_model.keras")
    print("✅ Model saved at models/cnn_model.keras")

    # ==============================
    # SAVE GRAPH
    # ==============================
    os.makedirs("outputs/graphs", exist_ok=True)

    plt.figure()
    plt.plot(history.history['accuracy'], label='train_accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.legend()
    plt.title("CNN Accuracy")
    plt.savefig("outputs/graphs/cnn_accuracy.png")
    plt.close()

    print("📈 Accuracy graph saved")


# ==============================
# MAIN
# ==============================
if __name__ == "__main__":
    train()