from fastapi import FastAPI, UploadFile, File
import shutil
import os

from src.predict import predict_emotion

app = FastAPI()

TEMP_FILE = "temp.wav"


@app.get("/")
def home():
    return {"message": "Speech Emotion API Running (CNN Model)"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Save uploaded file
    with open(TEMP_FILE, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Predict emotion
    emotion = predict_emotion(TEMP_FILE)

    # Delete temp file
    os.remove(TEMP_FILE)

    return {
        "predicted_emotion": emotion
    }