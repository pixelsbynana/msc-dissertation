"""
WFDB loading and preprocessing.

Two signals are produced from every upload:

  raw_signal   — exactly what wfdb.rdsamp() returns (physical units, original
                 length/rate). Used ONLY for the "ECG Signal" display —
                 no filtering, no cleaning, no resampling, no normalisation.

  model_input  — resampled to 500 Hz, padded/truncated to 5000 samples,
                 cleaned with NeuroKit2 (nk.ecg_clean, method="neurokit"),
                 then z-scored with the training StandardScaler.
                 This exactly mirrors the preprocessing cells in
                 project-main.ipynb, so the model receives signals from
                 the same distribution it was trained on.
"""

import logging
from pathlib import Path
from typing import Tuple

import neurokit2 as nk
import numpy as np
import scipy.signal
import wfdb

logger = logging.getLogger(__name__)

SAMPLING_RATE = 500   # Hz — training rate
N_LEADS = 12
SIGNAL_LENGTH = 5000  # samples at 500 Hz = 10 seconds

LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF",
              "V1", "V2", "V3", "V4", "V5", "V6"]


class ECGLoadError(ValueError):
    """Raised when an uploaded record cannot be used for inference."""


def load_wfdb_record(record_path: str) -> Tuple[np.ndarray, int]:
    """
    Read a WFDB record (.hea + .dat) and return the raw physical signal.

    Args:
        record_path: path WITHOUT extension; wfdb opens <path>.hea and the
                     signal file it references.

    Returns:
        signal: float32 array (T, 12)
        fs:     sampling rate in Hz

    Raises:
        FileNotFoundError: header missing.
        ECGLoadError:      not a 12-lead recording, or unreadable.
    """
    hea_path = Path(record_path).with_suffix(".hea")
    if not hea_path.exists():
        raise FileNotFoundError(f"Header file not found: {hea_path}")

    try:
        signal, fields = wfdb.rdsamp(str(record_path))
    except Exception as exc:
        raise ECGLoadError(f"Could not read WFDB record: {exc}") from exc

    signal = signal.astype(np.float32)
    if signal.shape[1] != N_LEADS:
        raise ECGLoadError(
            f"Expected {N_LEADS} leads, got {signal.shape[1]}. "
            "Please upload a standard 12-lead ECG recording."
        )
    if np.isnan(signal).all():
        raise ECGLoadError("Signal contains no valid samples.")

    fs = int(fields["fs"])
    logger.info("Loaded record %s — shape %s, fs=%d Hz", record_path, signal.shape, fs)
    return signal, fs


def resample_signal(signal: np.ndarray, original_fs: int, target_fs: int = SAMPLING_RATE) -> np.ndarray:
    """Resample (T, C) signal from original_fs to target_fs via Fourier-method resampling."""
    if original_fs == target_fs:
        return signal
    num_target = int(round(len(signal) * target_fs / original_fs))
    resampled = scipy.signal.resample(signal, num_target, axis=0)
    return resampled.astype(np.float32)


def pad_or_truncate(signal: np.ndarray, target_length: int = SIGNAL_LENGTH) -> np.ndarray:
    """Zero-pad (at the end) or truncate (from the end) to exactly target_length samples."""
    T = signal.shape[0]
    if T == target_length:
        return signal
    if T > target_length:
        return signal[:target_length, :]
    pad = np.zeros((target_length - T, signal.shape[1]), dtype=np.float32)
    return np.vstack([signal, pad])


def clean_ecg_record(signal: np.ndarray, sampling_rate: int = SAMPLING_RATE) -> np.ndarray:
    """Clean every lead with NeuroKit2 — matches clean_ecg_record() in the training notebook."""
    return np.column_stack([
        nk.ecg_clean(signal[:, i], sampling_rate=sampling_rate, method="neurokit")
        for i in range(signal.shape[1])
    ]).astype(np.float32)


def normalize_signal(signal: np.ndarray, scaler) -> np.ndarray:
    """Apply the training StandardScaler (per-lead z-score, fit on cleaned training data)."""
    T, C = signal.shape
    return scaler.transform(signal.reshape(-1, C)).reshape(T, C).astype(np.float32)


def prepare_signals(record_path: str, scaler) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Run the full pipeline for one uploaded record.

    Returns:
        raw_signal:  (T_original, 12) float32 — untouched, for display only.
        model_input: (12, 5000) float32 — cleaned + normalised, ready for the model.
        fs_original: sampling rate of the uploaded recording (for the raw display axis).
    """
    raw_signal, fs = load_wfdb_record(record_path)

    working = resample_signal(raw_signal, original_fs=fs, target_fs=SAMPLING_RATE)
    working = pad_or_truncate(working, target_length=SIGNAL_LENGTH)
    working = clean_ecg_record(working, sampling_rate=SAMPLING_RATE)
    working = normalize_signal(working, scaler)

    model_input = working.T.copy()  # (12, 5000)
    return raw_signal, model_input, fs
