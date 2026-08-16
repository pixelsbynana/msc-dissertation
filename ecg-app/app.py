"""
ECG Diagnostic Assistant — Streamlit application entry point.

12SL multi-label 12-lead ECG classifier (28 statement codes), with Grad-CAM
explainability and Monte Carlo Dropout uncertainty estimation.

Run with:
    streamlit run app.py

Required artefacts in models/ :
    best_model_12sl.pth          ECGResNet state-dict
    scaler_12sl.pkl              StandardScaler fit on cleaned training signals
    mlb_12sl.pkl                 MultiLabelBinarizer (defines class order)
    optimal_thresholds_12sl.npy  per-class thresholds tuned on fold 9
"""

import logging
import os
import tempfile
from pathlib import Path

import numpy as np
import streamlit as st

st.set_page_config(
    page_title="ECG Diagnostic Assistant",
    page_icon="\U0001FAC0",
    layout="wide",
    initial_sidebar_state="collapsed",
)

from inference import (
    N_MC_PASSES,
    apply_thresholds,
    confidence_label,
    get_device,
    load_classes,
    load_model,
    load_scaler,
    load_thresholds,
    mc_dropout_predict,
    predict,
)
from labels import get_full_name
from preprocessing import ECGLoadError, prepare_signals
from visualization import (
    GradCAM1D,
    build_probability_table,
    plot_gradcam_ecg,
    plot_mc_uncertainty,
    plot_raw_ecg,
    plot_top_probability_bar,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(name)s — %(message)s")
logger = logging.getLogger(__name__)

TOP_K_GRADCAM = 3  # number of top predicted classes to render individual Grad-CAM plots for


# ---------------------------------------------------------------------------
# Cached resource loading
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading model…")
def load_resources():
    device = get_device()
    try:
        classes = load_classes()
        thresholds = load_thresholds(classes)
        model = load_model(num_classes=len(classes), device=device)
        scaler = load_scaler()
        gradcam = GradCAM1D(model)
        logger.info("All resources loaded on %s (%d classes)", device, len(classes))
        return model, scaler, classes, thresholds, gradcam, device
    except FileNotFoundError as exc:
        logger.error("Resource loading failed: %s", exc)
        return None, None, None, None, None, device


def _write_upload_to_tmpdir(tmpdir: str, hea_bytes: bytes, dat_bytes: bytes) -> str:
    """
    WFDB requires the file stem referenced inside the .hea file to match the
    actual filename on disk, so the stem is parsed from the header's first
    line rather than the uploaded filenames.
    """
    first_line = hea_bytes.decode("utf-8", errors="ignore").splitlines()[0]
    record_stem = Path(first_line.split()[0]).name

    hea_path = os.path.join(tmpdir, f"{record_stem}.hea")
    dat_path = os.path.join(tmpdir, f"{record_stem}.dat")
    with open(hea_path, "wb") as f:
        f.write(hea_bytes)
    with open(dat_path, "wb") as f:
        f.write(dat_bytes)
    return os.path.join(tmpdir, record_stem)


# ---------------------------------------------------------------------------
# UI sections
# ---------------------------------------------------------------------------

def _show_prediction_section(probabilities, thresholds, positives) -> None:
    st.header("Prediction")

    if positives:
        st.success(
            f"**{len(positives)}** statement(s) exceeded their tuned threshold.", icon="✅"
        )
        st.plotly_chart(plot_top_probability_bar(positives), use_container_width=True)

        cols = st.columns(min(3, len(positives)))
        for i, (code, prob) in enumerate(positives[:3]):
            with cols[i % len(cols)]:
                st.metric(get_full_name(code), f"{prob:.1%}", help=f"12SL code: {code}")
    else:
        st.warning(
            "No 12SL statement exceeded its tuned decision threshold for this recording.",
            icon="⚠️",
        )

    with st.expander("Full probability table (all 28 classes)"):
        df = build_probability_table(probabilities, thresholds)
        st.dataframe(
            df.style.format({"Probability": "{:.1%}", "Threshold": "{:.3f}"}),
            use_container_width=True, hide_index=True,
        )


def _show_signal_section(raw_signal: np.ndarray, fs: int) -> None:
    st.header("ECG Signal (raw, unfiltered)")
    st.caption(
        "Exactly as read from the uploaded WFDB record — no NeuroKit cleaning, "
        "filtering, or normalisation applied. All 12 leads shown."
    )
    st.plotly_chart(plot_raw_ecg(raw_signal, fs), use_container_width=True)


def _show_uncertainty_section(classes, mc_results, top_classes) -> None:
    st.header("Uncertainty Estimation")
    st.caption(
        f"Monte Carlo Dropout: {N_MC_PASSES} stochastic forward passes with Dropout active. "
        "Higher standard deviation indicates greater epistemic uncertainty."
    )

    mean_probs = mc_results["mean_probs"]
    std_probs = mc_results["std_probs"]
    entropy = mc_results["entropy"]
    class_idx = {c: i for i, c in enumerate(classes)}

    if not top_classes:
        top_classes = [classes[int(np.argmax(mean_probs))]]

    display_classes = top_classes[:5]
    idxs = [class_idx[c] for c in display_classes]

    st.plotly_chart(
        plot_mc_uncertainty(display_classes, mean_probs[idxs], std_probs[idxs]),
        use_container_width=True,
    )

    st.subheader("Confidence Summary")
    cols = st.columns(min(3, len(display_classes)))
    for i, code in enumerate(display_classes):
        j = class_idx[code]
        with cols[i % len(cols)]:
            tier = confidence_label(std_probs[j])
            icon = "\U0001F7E2" if tier.startswith("High") else ("\U0001F7E1" if tier.startswith("Medium") else "\U0001F534")
            st.metric(get_full_name(code), f"{mean_probs[j]:.1%} ± {std_probs[j]:.3f}")
            st.caption(f"{icon} {tier}")

    st.caption(f"Total predictive entropy across all 28 classes: {entropy.sum():.3f}")


def _show_gradcam_section(model_input, gradcam, device, classes, probabilities, top_classes) -> None:
    st.header("Explainability — Grad-CAM")
    st.markdown(
        "Highlights which **time regions** of the (model-preprocessed) ECG signal most "
        "influenced each predicted diagnosis. Red regions received the highest gradient "
        "weight in the final residual block (`block4`)."
    )

    if not top_classes:
        st.info("No thresholded prediction available to explain.")
        return

    tabs = st.tabs([f"{code} — {get_full_name(code)}" for code in top_classes])
    for tab, code in zip(tabs, top_classes):
        with tab:
            class_idx = classes.index(code)
            probability = probabilities[code]
            cam = gradcam.compute_cam(model_input, class_idx=class_idx, device=device)
            fig = plot_gradcam_ecg(model_input, cam, code, probability)
            st.pyplot(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    st.title("\U0001FAC0 ECG Diagnostic Assistant")
    st.markdown(
        "_Explainable Deep Learning for 12-Lead ECG Classification "
        "(PTB-XL · 12SL Statements · 1-D ResNet · Grad-CAM · MC Dropout)_"
    )
    st.divider()

    model, scaler, classes, thresholds, gradcam, device = load_resources()
    if model is None:
        st.error(
            "**Model artefacts not found.** Ensure `best_model_12sl.pth`, `scaler_12sl.pkl`, "
            "`mlb_12sl.pkl`, and `optimal_thresholds_12sl.npy` are present in `models/`."
        )
        st.stop()
    st.caption(f"Running on: `{device}`  ·  {len(classes)} 12SL classes")

    st.subheader("Upload ECG Recording")
    st.markdown("Upload the two files that make up a WFDB record: the **header** (`.hea`) and the **signal data** (`.dat`).")

    col_hea, col_dat = st.columns(2)
    with col_hea:
        hea_file = st.file_uploader("Header file (.hea)", type=["hea"])
    with col_dat:
        dat_file = st.file_uploader("Signal file (.dat)", type=["dat"])

    both_uploaded = hea_file is not None and dat_file is not None
    analyse_btn = st.button("\U0001F52C Analyse ECG", type="primary", disabled=not both_uploaded)

    if not both_uploaded:
        st.info("Upload both a `.hea` and a `.dat` file to begin.")
        return
    if not analyse_btn:
        return

    hea_bytes, dat_bytes = hea_file.read(), dat_file.read()

    with st.spinner("Loading and preprocessing ECG signal…"):
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                record_path = _write_upload_to_tmpdir(tmpdir, hea_bytes, dat_bytes)
                raw_signal, model_input, fs = prepare_signals(record_path, scaler)
        except FileNotFoundError as exc:
            st.error(f"File error: {exc}")
            st.stop()
        except ECGLoadError as exc:
            st.error(f"Invalid ECG file: {exc}")
            st.stop()
        except Exception as exc:
            st.error(f"Preprocessing failed: {exc}")
            logger.exception("Preprocessing error")
            st.stop()

    with st.spinner("Running prediction…"):
        try:
            probs = predict(model_input, model, device)
            probabilities = {c: float(p) for c, p in zip(classes, probs)}
            positives = apply_thresholds(probabilities, thresholds)
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
            logger.exception("Prediction error")
            st.stop()

    with st.spinner(f"Estimating uncertainty ({N_MC_PASSES} MC Dropout passes)…"):
        try:
            mc_results = mc_dropout_predict(model_input, model, device)
        except Exception as exc:
            st.error(f"Uncertainty estimation failed: {exc}")
            logger.exception("MC Dropout error")
            st.stop()

    top_classes = [code for code, _ in positives[:TOP_K_GRADCAM]]

    st.success("Analysis complete.", icon="✅")
    st.divider()

    _show_prediction_section(probabilities, thresholds, positives)
    st.divider()

    _show_uncertainty_section(classes, mc_results, top_classes)
    st.divider()

    _show_signal_section(raw_signal, fs)
    st.divider()

    with st.spinner("Computing Grad-CAM…"):
        try:
            _show_gradcam_section(model_input, gradcam, device, classes, probabilities, top_classes)
        except Exception as exc:
            st.error(f"Grad-CAM failed: {exc}")
            logger.exception("Grad-CAM error")


if __name__ == "__main__":
    main()
