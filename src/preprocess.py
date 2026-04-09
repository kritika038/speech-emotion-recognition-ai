import librosa
import numpy as np

def load_audio(file_path):
    """
    Load audio file and return signal and sample rate
    """
    audio, sr = librosa.load(file_path, duration=3, offset=0.5)
    return audio, sr


def extract_features(audio, sr):
    """
    Extract MFCC, Chroma, and Mel features
    """
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    mfcc = np.mean(mfcc.T, axis=0)

    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    chroma = np.mean(chroma.T, axis=0)

    mel = librosa.feature.melspectrogram(y=audio, sr=sr)
    mel = np.mean(mel.T, axis=0)

    features = np.hstack([mfcc, chroma, mel])
    return features