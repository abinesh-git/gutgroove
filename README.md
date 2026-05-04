# GutGroove

Acoustic gut health monitoring — DH803 PoC project.

A wearable adhesive patch that listens to bowel sounds and computes an Intestinal Rate (IR) score to indicate gut motility.

---

## What's built so far

- Loaded and labelled 1606 bowel sound clips from the Kaggle dataset
- Generated log-Mel spectrograms for each clip
- Trained a CNN binary classifier (sound vs silence) — 81% test accuracy
- Built a rolling IR score calculator (events per minute)
- Flask web app — upload a WAV file, get an IR score

---

## Project structure

````
gutgroove/
├── notebooks/            # exploration and training (start here)
├── src/
│   └── predict.py        # core prediction functions
├── app/
│   ├── server.py         # Flask backend
│   └── templates/
│       └── index.html    # frontend
├── models/
│   └── best_model.keras  # trained CNN
└── data/                 # not in git — download from Kaggle
````

## Setup

````bash
git clone https://github.com/abinesh-git/gutgroove.git
cd gutgroove
python3 -m venv venv
source venv/bin/activate
pip install librosa tensorflow numpy matplotlib scikit-learn jupyter pandas flask seaborn
````

Download the dataset from https://www.kaggle.com/datasets/robertnowak/bowel-sounds and place files in `data/raw/`

## Run the web app

````bash
python app/server.py
````

Open http://127.0.0.1:5000, upload a `.wav` file from `data/raw/`, see the IR score.

## Run the notebook

````bash
jupyter notebook
````

Open `notebooks/01_explore.ipynb` and run top to bottom.

---

## IR score interpretation

| Range | Status |
|---|---|
| < 5 events/min | hypoactive — slow gut |
| 5–30 events/min | normal motility |
| > 30 events/min | hyperactive |

---

## Status

- [x] Data pipeline
- [x] Spectrogram generation
- [x] CNN training (81% accuracy)
- [x] IR score calculator
- [x] Web app
- [ ] Test on real hardware audio
- [ ] UI improvements
- [ ] Deployment
- [ ] BLE integration
````
````

Push it:

````bash
git add README.md
git commit -m "add README"
git push
````
## Run the notebook

```bash
jupyter notebook
```

Open `notebooks/01_explore.ipynb` and run top to bottom.

---

## IR score interpretation

| Range | Status |
|---|---|
| < 5 events/min | hypoactive — slow gut |
| 5–30 events/min | normal motility |
| > 30 events/min | hyperactive |

---

## Status

- [x] Data pipeline
- [x] Spectrogram generation
- [x] CNN training (81% accuracy)
- [x] IR score calculator
- [x] Web app
- [ ] Test on real hardware audio
- [ ] UI improvements
- [ ] Deployment
- [ ] BLE integration
