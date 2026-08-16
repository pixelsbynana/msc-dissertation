"""
Model architecture, artefact loading, and deterministic inference for the
28-class 12SL ECGResNet.

Architecture and class ordering mirror project-main.ipynb exactly so the
checkpoint saved there (models/best_model_12sl.pth) loads without key
mismatches, and probabilities line up with models/optimal_thresholds_12sl.npy.
"""

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).parent / "models"
MODEL_PATH = MODEL_DIR / "best_model_12sl.pth"
SCALER_PATH = MODEL_DIR / "scaler_12sl.pkl"
MLB_PATH = MODEL_DIR / "mlb_12sl.pkl"
THRESHOLDS_PATH = MODEL_DIR / "optimal_thresholds_12sl.npy"


# ---------------------------------------------------------------------------
# Model architecture (identical to the training notebook)
# ---------------------------------------------------------------------------

class ResBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 9) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm1d(out_channels)

        if in_channels != out_channels:
            self.shortcut: nn.Module = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class ECGResNet(nn.Module):
    """
    Input:  (batch, 12, 5000) — 12 leads x 5000 samples at 500 Hz
    Output: (batch, num_classes) raw logits (apply sigmoid for probabilities)
    """

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.block1 = ResBlock1D(12, 32)
        self.pool1 = nn.MaxPool1d(2)
        self.block2 = ResBlock1D(32, 64)
        self.pool2 = nn.MaxPool1d(2)
        self.block3 = ResBlock1D(64, 128)
        self.pool3 = nn.MaxPool1d(2)
        self.block4 = ResBlock1D(128, 256)
        self.gap = nn.AdaptiveAvgPool1d(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(self.block1(x))
        x = self.pool2(self.block2(x))
        x = self.pool3(self.block3(x))
        x = self.gap(self.block4(x))
        return self.classifier(x)


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Artefact loading
# ---------------------------------------------------------------------------

def load_classes(mlb_path: Path = MLB_PATH) -> List[str]:
    """Load the 28 12SL class codes, in the exact order the model outputs them."""
    if not mlb_path.exists():
        raise FileNotFoundError(f"MultiLabelBinarizer not found at {mlb_path}")
    with open(mlb_path, "rb") as f:
        mlb = pickle.load(f)
    return list(mlb.classes_)


def load_thresholds(classes: List[str], thresholds_path: Path = THRESHOLDS_PATH) -> Dict[str, float]:
    """Load per-class tuned thresholds (fold-9 F1 optimum), keyed by class code."""
    if not thresholds_path.exists():
        raise FileNotFoundError(f"Thresholds not found at {thresholds_path}")
    values = np.load(thresholds_path)
    if len(values) != len(classes):
        raise ValueError(
            f"Threshold count ({len(values)}) does not match class count ({len(classes)})."
        )
    return {cls: float(t) for cls, t in zip(classes, values)}


def load_scaler(scaler_path: Path = SCALER_PATH):
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler not found at {scaler_path}")
    with open(scaler_path, "rb") as f:
        return pickle.load(f)


def load_model(
    num_classes: int,
    device: torch.device,
    model_path: Path = MODEL_PATH,
) -> ECGResNet:
    if not model_path.exists():
        raise FileNotFoundError(f"Model weights not found at {model_path}")
    model = ECGResNet(num_classes=num_classes)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model loaded from %s on %s (%s params)", model_path, device, f"{n_params:,}")
    return model


# ---------------------------------------------------------------------------
# Deterministic prediction
# ---------------------------------------------------------------------------

def predict(
    model_input: np.ndarray,
    model: ECGResNet,
    device: torch.device,
) -> np.ndarray:
    """
    Single deterministic forward pass.

    Args:
        model_input: (12, 5000) normalised float32 array.

    Returns:
        (num_classes,) sigmoid probabilities.
    """
    model.eval()
    x = torch.tensor(model_input, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.sigmoid(logits).cpu().numpy()[0]
    return probs


def apply_thresholds(
    probabilities: Dict[str, float],
    thresholds: Dict[str, float],
) -> List[Tuple[str, float]]:
    """Return (class, probability) pairs that clear their tuned threshold, sorted descending."""
    positives = [(c, p) for c, p in probabilities.items() if p >= thresholds[c]]
    return sorted(positives, key=lambda item: item[1], reverse=True)


# ---------------------------------------------------------------------------
# Monte Carlo Dropout uncertainty
# ---------------------------------------------------------------------------

N_MC_PASSES = 50  # fixed — matches the notebook's MC Dropout evaluation (no user control)


def set_mc_dropout_mode(model: ECGResNet) -> None:
    """
    Turns on Monte Carlo Dropout: Dropout stays active (so it keeps
    randomly zeroing activations), but BatchNorm is forced into eval mode so it
    uses its fixed, learned statistics instead of recalculating from whatever
    batch happens to pass through. Without this, BatchNorm's behavior would
    depend on batch size — especially with single-record batches — which
    would contaminate the uncertainty estimate with noise unrelated to the
    model itself. Mirrors set_mc_dropout_mode() in project-main.ipynb.
    """
    model.eval()
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()


def mc_dropout_predict(
    model_input: np.ndarray,
    model: ECGResNet,
    device: torch.device,
    n_samples: int = N_MC_PASSES,
) -> Dict[str, np.ndarray]:
    """
    Runs n_samples forward passes with dropout active (BatchNorm stays in eval
    mode — see set_mc_dropout_mode) to get a distribution of predictions per
    sample, instead of just one.
    
    Skips torch.no_grad() on purpose: if Grad-CAM hooks are still attached to
    the model elsewhere in the notebook, no_grad() breaks them. Detaching each
    output right after computing it saves the same memory without that conflict.
    
    Returns:
        mean_probs (N, C): average predicted probability across passes
        std_probs  (N, C): how much predictions varied across passes — this is
        the model's uncertainty per class
    """
    x = torch.tensor(model_input, dtype=torch.float32).unsqueeze(0).to(device)

    set_mc_dropout_mode(model)
    passes = []
    with torch.no_grad():
        for _ in range(n_samples):
            logits = model(x)
            passes.append(torch.sigmoid(logits).cpu().numpy()[0])
    model.eval()

    raw = np.stack(passes, axis=0)  # (n_samples, num_classes)
    mean_probs = raw.mean(axis=0)
    std_probs = raw.std(axis=0)

    eps = 1e-8
    entropy = -(
        mean_probs * np.log(mean_probs + eps)
        + (1 - mean_probs) * np.log(1 - mean_probs + eps)
    )

    return {
        "mean_probs": mean_probs,
        "std_probs": std_probs,
        "entropy": entropy,
        "raw_passes": raw,
    }


def confidence_label(std: float) -> str:
    """Heuristic mapping from MC Dropout std-dev to a plain-language confidence tier."""
    if std < 0.05:
        return "High confidence"
    if std < 0.15:
        return "Medium confidence"
    return "Low confidence"
