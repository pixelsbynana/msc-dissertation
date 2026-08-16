"""
12SL diagnostic statement label mapping.

The model (trained in project-main.ipynb) predicts 28 12SL statement codes,
filtered to those occurring >= 300 times in PTB-XL (MIN_FREQ=300), with
punctuation artefacts (LPAREN / RPAREN / COMMA) and non-clinically-relevant
tokens (grammatical fragments, generic ECG quality
labels, age metadata, and raw LVH voltage-criterion measurements) excluded.

FULL_NAMES maps each short 12SL acronym to its full human-readable diagnostic
text, sourced from the 12SL statement dictionary (heedb_10k/12sl_dictionary.csv).
"""

from typing import Dict

FULL_NAMES: Dict[str, str] = {
    "2ST":       "With Repolarization Abnormality",
    "AFB":       "Left Anterior Fascicular Block",
    "AFIB":      "Atrial Fibrillation",
    "AMI":       "Anterior Infarct",
    "ASMI":      "Anteroseptal Infarct",
    "FAV":       "1st Degree AV Block",
    "IMI":       "Inferior Infarct",
    "IRBBB":     "Incomplete Right Bundle Branch Block",
    "IT":        "T Wave Abnormality — Consider Inferior Ischemia",
    "LAD3":      "Left Axis Deviation",
    "LAE":       "Left Atrial Enlargement",
    "LBBB":      "Left Bundle Branch Block",
    "LNGQT":     "Prolonged QT",
    "LOWV":      "Low Voltage QRS",
    "LT":        "T Wave Abnormality — Consider Lateral Ischemia",
    "LVH2":      "Left Ventricular Hypertrophy",
    "NML":       "Normal ECG",
    "NSR":       "Normal Sinus Rhythm",
    "NST":       "Nonspecific ST Abnormality",
    "NSTT":      "Nonspecific ST and T Wave Abnormality",
    "NT":        "Nonspecific T Wave Abnormality",
    "PAC":       "Premature Atrial Complexes",
    "PVC":       "Premature Ventricular Complexes",
    "RBBB":      "Right Bundle Branch Block",
    "SBRAD":     "Sinus Bradycardia",
    "SMI":       "Septal Infarct",
    "SRTH":      "Sinus Rhythm",
    "STACH":     "Sinus Tachycardia",
}


def get_full_name(code: str) -> str:
    """Return the full diagnostic name for a 12SL code, or the raw code if unmapped."""
    return FULL_NAMES.get(code, code)
