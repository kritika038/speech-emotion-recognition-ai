import os

print("🚀 Starting Speech Emotion Recognition App...")

# Install dependencies
print("📦 Installing dependencies...")
os.system("pip install -r requirements.txt")

# Run Streamlit app
print("🌐 Launching Streamlit app...")
os.system("streamlit run streamlit_app.py")