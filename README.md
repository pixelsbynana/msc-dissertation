# ECG Diagnostic Assistant

**Deep Learning for Automated Arrhythmia Detection using Digital ECG Signals**

MSc Computer Science (Conversion) dissertation project by **Arrada Wongkittiruk**, supervised by **Dr. Xu Chen**.

This project trains a deep learning model to read 12-lead ECGs (electrocardiograms) and predict 28 possible diagnoses at once — for example normal sinus rhythm, left bundle branch block, or anterior infarct. Because a single ECG can show more than one condition at the same time, the model is built for **multi-label classification** rather than picking just one diagnosis.

Alongside the prediction, the project also estimates **how confident the model is** in each prediction, and shows **which part of the ECG signal the model focused on** to reach that decision.

## Try it live

A working version of the app is hosted here:

**https://ecg-assistant.streamlit.app/**

> If the page says *"This app has gone to sleep due to inactivity"*, click **"Yes, get this app back up!"** and wait about 30–60 seconds for it to restart.

## What the project does

1. **Trains a model** — a 1D ResNet (a type of neural network good at learning patterns in signals) is trained on the [PTB-XL](https://physionet.org/content/ptb-xl/) dataset to recognise 28 diagnostic classes from raw ECG waveforms.
2. **Tunes per-class decision thresholds** — instead of one fixed cutoff for "yes, this diagnosis is present", each of the 28 classes gets its own tuned cutoff, which noticeably improves results.
3. **Estimates uncertainty** — using a technique called Monte Carlo Dropout, the model is run 50 times per ECG with slight randomness switched on, and how much the predictions disagree with each other is used as an uncertainty score.
4. **Explains its predictions** — using Grad-CAM, the model highlights which part of the ECG signal most influenced its decision, shown as a heatmap over the waveform.
5. **Is tested on unseen hospital data** — the model is also evaluated on the independent Harvard-Emory ECG Database (HEEDB) to check whether it still works well on ECGs from a completely different hospital, not just the dataset it was trained on.
6. **Is wrapped in a clinical-style app** — a Streamlit web app lets you upload an ECG recording and see the prediction, the uncertainty, and the Grad-CAM explanation together.

**This project is a research and educational demonstration only. It is not a certified medical device and must not be used to make real clinical decisions.**

## Repository structure

```
.
├── project-main.ipynb       # Main notebook: data prep, training, evaluation, explainability
├── 12sl_statements.csv      # 12SL diagnostic labels for PTB-XL
├── label_distribution.png   # Chart of how common each diagnosis is in the dataset
├── ecg-app/                 # The Streamlit web app
│   ├── app.py                #   App entry point — run this with `streamlit run`
│   ├── inference.py           #   Loads the trained model and runs predictions
│   ├── preprocessing.py       #   Cleans and prepares an uploaded ECG for the model
│   ├── visualization.py       #   Builds the charts and Grad-CAM plots shown in the app
│   ├── labels.py               #   Maps diagnosis codes to full names
│   ├── requirements.txt       #   Python packages needed to run the app
│   └── models/                #   Saved, trained model files (already included)
├── ptb-xl-og/                # PTB-XL raw ECG data (not included — see below)
└── heedb_10k/                # HEEDB raw ECG data (not included — see below)
```

`ptb-xl-og/` and `heedb_10k/` are excluded from this repository (see `.gitignore`) because they are several gigabytes of raw ECG recordings. You only need these folders if you want to **re-run the training notebook from scratch** — you do **not** need them to run the app, since the trained model is already saved in `ecg-app/models/`.

## Running the app yourself

You don't need any ECG datasets to do this — the trained model, scaler, and label files are already included in `ecg-app/models/`.

1. **Install Python 3.10+** if you don't already have it.
2. **Open a terminal and go into the app folder:**
   ```bash
   cd ecg-app
   ```
3. **(Recommended) Create a virtual environment**, so the app's packages don't clash with anything else on your machine:
   ```bash
   python -m venv venv
   source venv/bin/activate        # on Windows: venv\Scripts\activate
   ```
4. **Install the required packages:**
   ```bash
   pip install -r requirements.txt
   ```
5. **Start the app:**
   ```bash
   streamlit run app.py
   ```
   This opens the app in your browser, usually at `http://localhost:8501`.

### How to use the app

The app expects a **WFDB-format ECG record**, which is made up of two files with the same name:

- a **header file** (`.hea`)
- a **signal file** (`.dat`)

Upload both files together in the app. It will automatically:
- resample the signal to match what the model expects,
- clean the signal to remove noise,
- run the model and show the predicted diagnoses,
- show how confident the model is,
- show a Grad-CAM heatmap over the ECG explaining the prediction.

If you don't have a WFDB ECG record to test with, sample recordings can be downloaded from the public [PTB-XL dataset](https://physionet.org/content/ptb-xl/) on PhysioNet.

## Re-running the training notebook

Only do this if you want to reproduce the model training yourself from raw data — it requires downloading large datasets and can take a while to run.

1. **Download the datasets:**
   - [PTB-XL](https://physionet.org/content/ptb-xl/) — place it in a folder named `ptb-xl-og/` in the project root.
   - [Harvard-Emory ECG Database (HEEDB)](https://bdsp.io/content/heedb/5.0/) — a 10,000-record subset should go in a folder named `heedb_10k/` in the project root.
2. **Install the same packages** listed in `ecg-app/requirements.txt`, plus Jupyter:
   ```bash
   pip install -r ecg-app/requirements.txt jupyter
   ```
3. **Open the notebook:**
   ```bash
   jupyter notebook project-main.ipynb
   ```
4. **Run the cells from top to bottom.** The notebook is organised into clear sections: data loading and preprocessing, model architecture, training, evaluation and threshold tuning, Monte Carlo Dropout, Grad-CAM, and external validation on HEEDB.

Training was originally run on Apple Silicon (MPS), but the notebook automatically falls back to CUDA or CPU if MPS isn't available.
