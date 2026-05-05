import numpy as np
import librosa
import tensorflow as tf
import os

def load_model(model_path="models/best_model.keras"):
    return tf.keras.models.load_model(model_path)

def load_norm_params(mean_path="models/X_mean.npy", std_path="models/X_std.npy"):
    X_mean = np.load(mean_path)
    X_std = np.load(std_path)
    return X_mean, X_std

def predict_ir(filepath, model, X_mean, X_std, threshold=0.5):
    y, sr = librosa.load(filepath, sr=None)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40, fmin=200, fmax=5000)
    S_db = librosa.power_to_db(S, ref=np.max)
    S_normalized = (S_db - X_mean) / X_std
    S_input = S_normalized[np.newaxis, ..., np.newaxis]
    prob = model.predict(S_input, verbose=0)[0][0]
    label = "sound" if prob > threshold else "silence"
    return {"file": filepath, "probability": round(float(prob), 3), "label": label}

def rolling_ir(audio_files, model, X_mean, X_std, clips_per_minute=30):
    results = []
    for i in range(0, len(audio_files), clips_per_minute):
        window = audio_files[i:i + clips_per_minute]
        if len(window) < clips_per_minute:
            break
        events = sum(1 for f in window if predict_ir(f, model, X_mean, X_std)["label"] == "sound")
        minute = (i // clips_per_minute) + 1
        status = "normal" if 5 <= events <= 30 else "abnormal"
        results.append({"minute": minute, "ir": events, "status": status})
    return results
