import os
import shutil

# CHANGE THIS if your folder name is different
SOURCE_DIR = "audio_speech_actors_01-24"
TARGET_DIR = "data"

emotion_map = {
    "01": "neutral",
    "03": "happy",
    "04": "sad",
    "05": "angry"
}

# Create target folders
os.makedirs(TARGET_DIR, exist_ok=True)

for emotion in emotion_map.values():
    os.makedirs(os.path.join(TARGET_DIR, emotion), exist_ok=True)

# Traverse dataset
for root, dirs, files in os.walk(SOURCE_DIR):
    for file in files:
        if file.endswith(".wav"):
            parts = file.split("-")
            
            if len(parts) < 3:
                continue

            emotion_code = parts[2]

            if emotion_code in emotion_map:
                emotion = emotion_map[emotion_code]

                src = os.path.join(root, file)
                dst = os.path.join(TARGET_DIR, emotion, file)

                shutil.copy(src, dst)

print("✅ Dataset organized successfully!")