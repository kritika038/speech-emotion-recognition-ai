import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

from src.dataset import load_dataset
from src.model import build_model


DATA_PATH = "data"


def train():
    print("Loading dataset...")
    X, y = load_dataset(DATA_PATH)

    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("Training model...")
    model = build_model()
    model.fit(X_train, y_train)

    print("Evaluating model...")
    predictions = model.predict(X_test)

    acc = accuracy_score(y_test, predictions)
    print(f"Accuracy: {acc}")

    # Save model
    joblib.dump(model, "models/model.pkl")
    print("Model saved at models/model.pkl")

    # Confusion Matrix
    cm = confusion_matrix(y_test, predictions)

    plt.figure(figsize=(6,6))
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.savefig("outputs/graphs/confusion_matrix.png")
    plt.close()

    print("Confusion matrix saved")


if __name__ == "__main__":
    train()