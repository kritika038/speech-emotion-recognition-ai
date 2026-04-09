from fastapi import FastAPI, UploadFile, File
import shutil
import os

from src.predict import predict_emotion

app = FastAPI(title="Speech Emotion Recognition API")

TEMP_FILE = "temp.wav"


@app.get("/")
def home():
    return {"message": "API is running 🚀"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Upload audio file and get predicted emotion + confidence
    """
    try:
        # Save uploaded file
        with open(TEMP_FILE, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Predict
        result = predict_emotion(TEMP_FILE)

        # Remove temp file
        os.remove(TEMP_FILE)

        return {
            "filename": file.filename,
            "predicted_emotion": result["emotion"],
            "confidence": result["confidence"]
        }

    except Exception as e:
        return {"error": str(e)}