"""
Grad-CAM computation and all plotting helpers (Plotly + Matplotlib).

Nothing here calls st.* directly — app.py owns the Streamlit UI and passes
returned Figure objects to st.plotly_chart() / st.pyplot().
"""

import logging
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
import torch.nn.functional as F
from plotly.subplots import make_subplots

from inference import ECGResNet
from labels import get_full_name
from preprocessing import LEAD_NAMES

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Grad-CAM (1-D), target layer: model.block4
# ---------------------------------------------------------------------------

class GradCAM1D:
    """
    Grad-CAM (Selvaraju et al., 2017) for ECGResNet.

    Registers forward/backward hooks on model.block4 (output: batch x 256 x 625,
    the final residual block before global average pooling). Each channel's
    gradient is averaged over time to obtain a weight, the weighted feature
    maps are summed, and the result is ReLU'd and upsampled back to 5000
    samples for overlay on the input signal.
    """

    def __init__(self, model: ECGResNet) -> None:
        self.model = model
        self._activations: Optional[torch.Tensor] = None
        self._gradients: Optional[torch.Tensor] = None
        model.block4.register_forward_hook(self._forward_hook)
        model.block4.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inp, out) -> None:
        self._activations = out

    def _backward_hook(self, module, grad_inp, grad_out) -> None:
        self._gradients = grad_out[0]

    def compute_cam(
        self,
        model_input: np.ndarray,
        class_idx: int,
        device: torch.device,
        signal_len: int = 5000,
    ) -> np.ndarray:
        """
        Args:
            model_input: (12, 5000) normalised float32 array.
            class_idx:   index of the target 12SL class.

        Returns:
            (signal_len,) float32 array in [0, 1] — higher = more influential.
        """
        self.model.eval()
        x = torch.tensor(model_input, dtype=torch.float32).unsqueeze(0).to(device)
        x = x.detach().requires_grad_(True)

        logits = self.model(x)
        self.model.zero_grad()
        logits[0, class_idx].backward()

        weights = self._gradients[0].mean(dim=-1)                     # (256,)
        cam = (weights[:, None] * self._activations[0]).sum(dim=0)    # (625,)
        cam = F.relu(cam).detach().cpu().numpy()

        cam = np.interp(
            np.linspace(0, len(cam) - 1, signal_len),
            np.arange(len(cam)),
            cam,
        )
        cam_max = cam.max()
        if cam_max > 0:
            cam = cam / cam_max
        return cam.astype(np.float32)


def plot_gradcam_ecg(
    signal: np.ndarray,
    cam: np.ndarray,
    class_code: str,
    probability: float,
    sampling_rate: int = 500,
) -> plt.Figure:
    """
    Overlay the Grad-CAM heatmap on all 12 leads, stacked one above another.

    Args:
        signal: (12, 5000) array (the model-input signal, for time alignment).
        cam:    (5000,) Grad-CAM importance in [0, 1].
    """
    time = np.arange(signal.shape[1]) / sampling_rate
    cmap = plt.cm.YlOrRd

    fig, axes = plt.subplots(12, 1, figsize=(12, 16), sharex=True)
    fig.suptitle(
        f"Grad-CAM — {class_code}: {get_full_name(class_code)}  ·  Probability: {probability:.1%}",
        fontsize=16, fontweight="bold",
    )

    for lead_i, ax in enumerate(axes):
        sig = signal[lead_i]
        # Symmetric range centred on the isoelectric baseline (0), rather than
        # sig.min()/sig.max() independently -- otherwise leads with an inverted,
        # baseline-skewed morphology (e.g. aVR) show the baseline near the top of
        # the row while upright leads (e.g. II, V5) show it near the bottom, so
        # the baseline appears to wander from row to row for no clinical reason.
        y_abs = abs(sig).max() * 1.15
        ymin, ymax = -y_abs, y_abs

        ax.imshow(
            cam[np.newaxis, :], aspect="auto",
            extent=[time[0], time[-1], ymin, ymax],
            cmap=cmap, alpha=0.55, vmin=0, vmax=1,
            origin="lower", zorder=1,
        )
        ax.plot(time, sig, color="#1a1a2e", linewidth=0.65, zorder=2)
        ax.set_ylabel(
            LEAD_NAMES[lead_i], rotation=0, ha="right", va="center",
            fontsize=11, fontweight="bold", labelpad=10,
        )
        ax.set_xlim(time[0], time[-1])
        ax.set_ylim(ymin, ymax)
        ax.set_yticks([])
        ax.tick_params(axis="x", labelsize=8)
        ax.spines[["top", "right"]].set_visible(False)

    axes[-1].set_xlabel("Time (s)", fontsize=10)

    fig.tight_layout(rect=[0, 0, 0.92, 0.96])

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar_ax = fig.add_axes([0.94, 0.15, 0.015, 0.7])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Grad-CAM Importance", fontsize=10)
    cbar.ax.tick_params(labelsize=7)

    return fig


# ---------------------------------------------------------------------------
# ECG raw signal viewer (Plotly)
# ---------------------------------------------------------------------------

def plot_raw_ecg(signal: np.ndarray, fs: int) -> go.Figure:
    """
    Interactive 12-lead ECG viewer — one row per lead, no filtering applied.

    Args:
        signal: (T, 12) float32 array, exactly as returned by wfdb.rdsamp().
        fs:     sampling rate in Hz (from the uploaded header).
    """
    time = np.arange(signal.shape[0]) / fs

    fig = make_subplots(
        rows=12, cols=1, shared_xaxes=True, vertical_spacing=0.004,
        subplot_titles=LEAD_NAMES,
    )
    for row_i, lead_name in enumerate(LEAD_NAMES, start=1):
        fig.add_trace(
            go.Scatter(
                x=time, y=signal[:, row_i - 1], mode="lines", name=lead_name,
                line=dict(width=0.7, color="#1a1a2e"), showlegend=False,
            ),
            row=row_i, col=1,
        )
        fig.update_yaxes(row=row_i, col=1, showticklabels=False, showgrid=False, zeroline=False)

    fig.update_xaxes(title_text="Time (s)", showgrid=True, gridcolor="#f0f0f0", row=12, col=1)
    fig.update_layout(
        title=f"Raw ECG Signal — All 12 Leads (unfiltered, {fs} Hz)",
        height=1150,
        margin=dict(l=50, r=20, t=50, b=40),
        plot_bgcolor="white", paper_bgcolor="white",
    )
    return fig


# ---------------------------------------------------------------------------
# Probability table
# ---------------------------------------------------------------------------

def build_probability_table(
    probabilities: Dict[str, float],
    thresholds: Dict[str, float],
) -> pd.DataFrame:
    rows = [
        {
            "Code": cls,
            "Diagnosis": get_full_name(cls),
            "Probability": probabilities[cls],
            "Threshold": thresholds[cls],
            "Positive": probabilities[cls] >= thresholds[cls],
        }
        for cls in probabilities
    ]
    return pd.DataFrame(rows).sort_values("Probability", ascending=False).reset_index(drop=True)


def plot_top_probability_bar(
    positives: List[tuple],
    max_bars: int = 15,
) -> go.Figure:
    """Horizontal bar chart of thresholded-positive classes, full names, sorted by probability."""
    items = positives[:max_bars][::-1]  # reverse so highest prob is at top of horizontal chart
    names = [f"{get_full_name(c)} ({c})" for c, _ in items]
    probs = [p for _, p in items]

    fig = go.Figure(go.Bar(
        x=probs, y=names, orientation="h",
        marker_color="#c0392b",
        text=[f"{p:.1%}" for p in probs], textposition="outside", cliponaxis=False,
    ))
    fig.update_layout(
        title="Top Predicted Diagnoses (thresholded)",
        xaxis=dict(title="Probability", range=[0, 1.15], tickformat=".0%"),
        yaxis=dict(title=""),
        height=max(300, 40 * len(items) + 100),
        margin=dict(l=280, r=60, t=50, b=40),
        plot_bgcolor="white", paper_bgcolor="white",
    )
    return fig


# ---------------------------------------------------------------------------
# MC Dropout uncertainty
# ---------------------------------------------------------------------------

def plot_mc_uncertainty(
    classes: List[str],
    mean_probs: np.ndarray,
    std_probs: np.ndarray,
) -> go.Figure:
    """Mean +/- std bar chart for the classes shown in the top-predictions section."""
    order = np.argsort(mean_probs)[::-1]
    classes_sorted = [classes[i] for i in order]
    means = mean_probs[order]
    stds = std_probs[order]
    labels = [f"{get_full_name(c)} ({c})" for c in classes_sorted]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=means, y=labels, orientation="h",
        error_x=dict(type="data", array=stds, visible=True, color="#2c3e50"),
        marker_color="#2980b9",
    ))
    fig.update_layout(
        title=f"MC Dropout — Mean Probability +/- Std Dev",
        xaxis=dict(title="Probability", range=[0, 1.1]),
        yaxis=dict(title="", autorange="reversed"),
        height=max(320, 40 * len(labels) + 100),
        margin=dict(l=280, r=40, t=50, b=40),
        plot_bgcolor="white", paper_bgcolor="white",
    )
    return fig
