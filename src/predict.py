import numpy as np
import librosa
import tensorflow as tf

def load_model(model_path="models/best_model.keras"):
    return tf.keras.models.load_model(model_path)

def predict_ir(filepath, model, threshold=0.5):
    y, sr = librosa.load(filepath, sr=None)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40, fmin=200, fmax=5000)
    S_db = librosa.power_to_db(S, ref=np.max)
    S_db = S_db[np.newaxis, ..., np.newaxis]
    prob = model.predict(S_db, verbose=0)[0][0]
    label = "sound" if prob > threshold else "silence"
    return {"file": filepath, "probability": round(float(prob), 3), "label": label}

def rolling_ir(audio_files, model, clips_per_minute=30):
    results = []
    for i in range(0, len(audio_files), clips_per_minute):
        window = audio_files[i:i + clips_per_minute]
        if len(window) < clips_per_minute:
            break
        events = sum(1 for f in window if predict_ir(f, model)["label"] == "sound")
        minute = (i // clips_per_minute) + 1
        status = "normal" if 5 <= events <= 30 else "abnormal"
        results.append({"minute": minute, "ir": events, "status": status})
    return results
