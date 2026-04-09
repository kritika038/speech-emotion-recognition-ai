import os

from audio_utils import extract_mel_spectrogram

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ONNX_MODEL_PATH = os.path.join(BASE_DIR, "model.onnx")
PYTORCH_MODEL_PATH = os.path.join(BASE_DIR, "emotion_model.pth")

emotion_map = {
    0: "Neutral",
    1: "Calm",
    2: "Happy",
    3: "Sad",
    4: "Angry",
    5: "Fearful",
    6: "Disgust",
    7: "Surprised",
}

session = None
device = None
torch = None
model = None

try:
    import onnxruntime as ort

    if os.path.exists(ONNX_MODEL_PATH):
        session = ort.InferenceSession(ONNX_MODEL_PATH, providers=["CPUExecutionProvider"])
except Exception:
    session = None

if session is None:
    import torch

    from model import EmotionCNN

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EmotionCNN().to(device)
    model.load_state_dict(torch.load(PYTORCH_MODEL_PATH, map_location=device))
    model.eval()


def predict_emotion(file_path):
    mel = extract_mel_spectrogram(file_path)

    if session is not None:
        input_name = session.get_inputs()[0].name
        output = session.run(None, {input_name: mel.astype("float32")[None, None, :, :]})[0]
        pred = int(output.argmax(axis=1)[0])
    else:
        mel_tensor = torch.tensor(mel).unsqueeze(0).unsqueeze(0).float().to(device)

        with torch.no_grad():
            output = model(mel_tensor)
            pred = torch.argmax(output, dim=1).item()

    return emotion_map[pred]
