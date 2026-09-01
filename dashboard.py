"""
Interactive research dashboard for RF modulation classification.

Compares:
    - CNN1D (Raw IQ)
    - CNN-Transformer (Raw IQ)
    - ResNet-18 (2D STFT)
    - STFT-RADN (2D STFT, when correct checkpoint is available)

Metrics:
    - Overall accuracy
    - Accuracy vs SNR
    - Peak accuracy
    - Peak SNR
    - Confusion matrices
    - Per-class accuracy
    - CPU inference latency
    - Trainable parameter count

Run with:
    streamlit run dashboard.py
"""

from __future__ import annotations

import json
import importlib.util
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import torch
from sklearn.metrics import confusion_matrix


# ============================================================================
# PATH SETUP
# ============================================================================

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


# ============================================================================
# MODEL IMPORTS
# ============================================================================

from models_2d import ResNet18_2D, STFTRADN


_CLASSIFIERS_SPEC = importlib.util.spec_from_file_location(
    "dashboard_classifiers_1d",
    ROOT / "subsystem2_1d" / "classifiers_1d.py",
)

_CLASSIFIERS = importlib.util.module_from_spec(_CLASSIFIERS_SPEC)

assert _CLASSIFIERS_SPEC.loader is not None

_CLASSIFIERS_SPEC.loader.exec_module(_CLASSIFIERS)

build_model = _CLASSIFIERS.build_model


# ============================================================================
# CONSTANTS
# ============================================================================

CLASSES = [
    "8PSK",
    "AM-DSB",
    "AM-SSB",
    "BPSK",
    "CPFSK",
    "GFSK",
    "PAM4",
    "QAM16",
    "QAM64",
    "QPSK",
    "WBFM",
]

SNR_LEVELS = list(range(-20, 20, 2))

MODEL_FILES = {
    "ResNet-18 (2D)": ROOT / "results" / "resnet18.pt",
    "STFT-RADN (2D)": ROOT / "results" / "stft.pt",
    "CNN1D": ROOT / "subsystem2_1d" / "checkpoints" / "best_model.pt",
    "CNN-Transformer (1D)": (
        ROOT
        / "subsystem2_1d"
        / "checkpoints"
        / "best_cnn_transformer.pt"
    ),
}


# ============================================================================
# CHECKPOINT LOADING
# ============================================================================

def _checkpoint(path: Path) -> Any:
    """Load a checkpoint safely onto CPU."""
    return torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )


@st.cache_resource(show_spinner=False)
def load_model(name: str):
    """
    Load and initialise one of the four classifier artifacts.
    """

    path = MODEL_FILES[name]
    checkpoint = _checkpoint(path)

    # ----------------------------------------------------------------------
    # 2D RESNET
    # ----------------------------------------------------------------------

    if name == "ResNet-18 (2D)":

        model = ResNet18_2D(
            num_classes=11,
            in_channels=checkpoint.get("in_channels", 1),
        )

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

    # ----------------------------------------------------------------------
    # 2D STFT-RADN
    # ----------------------------------------------------------------------

    elif name == "STFT-RADN (2D)":

        architecture = checkpoint.get(
            "model_name",
            "stftradn",
        )

        # Automatically respect checkpoint metadata
        model_class = (
            ResNet18_2D
            if architecture == "resnet18"
            else STFTRADN
        )

        model = model_class(
            num_classes=11,
            in_channels=checkpoint.get("in_channels", 1),
        )

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

    # ----------------------------------------------------------------------
    # 1D MODELS
    # ----------------------------------------------------------------------

    else:

        model_type = (
            "cnn1d"
            if name == "CNN1D"
            else "cnn_transformer"
        )

        model = build_model(
            model_type,
            num_classes=11,
        )

        state_dict = (
            checkpoint["model_state_dict"]
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
            else checkpoint
        )
        model.load_state_dict(state_dict)

    model.eval()

    return model


# ============================================================================
# PARAMETER COUNT
# ============================================================================

def count_parameters(model: torch.nn.Module) -> int:
    """Return total trainable parameter count."""

    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


# ============================================================================
# LATENCY BENCHMARK
# ============================================================================

@st.cache_data(show_spinner=False)
def measure_latency(name: str) -> float:
    """
    Measure CPU inference latency.

    Protocol:
        - CPU
        - Batch size = 1
        - Evaluation mode
        - 20 warm-up runs
        - 200 timed inference runs

    Returns:
        Mean latency in milliseconds per sample.
    """

    model = load_model(name)

    # Ensure CPU comparison is fair
    model = model.cpu()
    model.eval()

    if "2D" in name:
        input_shape = (1, 1, 64, 5)
    else:
        input_shape = (1, 2, 128)

    sample = torch.randn(*input_shape)

    warmup_runs = 20
    measurement_runs = 200

    # ----------------------------------------------------------------------
    # WARMUP
    # ----------------------------------------------------------------------

    with torch.inference_mode():

        for _ in range(warmup_runs):
            _ = model(sample)

    # ----------------------------------------------------------------------
    # TIMED INFERENCE
    # ----------------------------------------------------------------------

    start_time = time.perf_counter()

    with torch.inference_mode():

        for _ in range(measurement_runs):
            _ = model(sample)

    end_time = time.perf_counter()

    total_time_ms = (
        end_time - start_time
    ) * 1000

    latency_per_sample = (
        total_time_ms / measurement_runs
    )

    return float(latency_per_sample)


# ============================================================================
# LOAD PERSISTED 1D RESULTS
# ============================================================================

@st.cache_data(show_spinner=False)
def load_1d_metrics(name: str) -> dict[str, Any]:
    """
    Load persisted locked-test evaluation results for 1D models.
    """

    key = (
        "cnn1d"
        if name == "CNN1D"
        else "cnn_transformer"
    )

    results_dir = (
        ROOT
        / "subsystem2_1d"
        / "results"
    )

    accuracy_path = (
        results_dir
        / f"accuracy_vs_snr_{key}.json"
    )

    with accuracy_path.open(
        encoding="utf-8"
    ) as stream:

        metrics = json.load(stream)

    matrices = {}

    # Load comparable SNR-specific confusion matrices
    for snr in (-10, 0, 10):

        matrix_path = (
            results_dir
            / f"confusion_matrix_{snr}db_{key}.npy"
        )

        if matrix_path.exists():

            matrices[snr] = np.load(
                matrix_path
            )

    return {
        "overall": metrics["overall_accuracy"],

        "snr": {
            int(key): value
            for key, value
            in metrics["per_snr_accuracy"].items()
        },

        "matrices": matrices,
    }


# ============================================================================
# EVALUATE 2D MODELS
# ============================================================================

@st.cache_data(
    show_spinner="Evaluating 2D models on locked test split..."
)
def evaluate_2d(name: str) -> dict[str, Any] | None:
    """
    Evaluate a 2D model on the locked test partition.

    Generates:
        - Overall accuracy
        - Accuracy at every SNR
        - Overall confusion matrix
        - Confusion matrices at -10, 0 and +10 dB
    """

    data_dir = ROOT / "data"

    required_files = [
        "spectrograms_all.npy",
        "y_all.npy",
        "snrs_all.npy",
        "test_idx.npy",
    ]

    # Don't fabricate metrics if processed test data is absent
    if not all(
        (data_dir / item).exists()
        for item in required_files
    ):
        return None

    from dataset_loader import (
        get_spectrogram_dataloader
    )

    model = load_model(name)

    model = model.cpu()
    model.eval()

    # Detect expected number of input channels
    first_conv_weight = next(
        parameter
        for parameter in model.parameters()
        if parameter.ndim == 4
    )

    in_channels = first_conv_weight.shape[1]

    test_loader = (
        get_spectrogram_dataloader(
            str(data_dir),
            split="test",
            batch_size=256,
            in_channels=in_channels,
            num_workers=0,
        )
    )

    truth = []
    predictions = []
    snrs = []

    # ----------------------------------------------------------------------
    # INFERENCE
    # ----------------------------------------------------------------------

    with torch.inference_mode():

        for signals, labels, batch_snrs in test_loader:

            outputs = model(signals)

            predicted = outputs.argmax(1)

            predictions.extend(
                predicted.numpy()
            )

            truth.extend(
                labels.numpy()
            )

            snrs.extend(
                batch_snrs.numpy()
            )

    truth = np.array(truth)
    predictions = np.array(predictions)
    snrs = np.array(snrs)

    # ----------------------------------------------------------------------
    # OVERALL CONFUSION MATRIX
    # ----------------------------------------------------------------------

    overall_matrix = confusion_matrix(
        truth,
        predictions,
        labels=range(len(CLASSES)),
    )

    matrices = {
        "Overall": overall_matrix
    }

    # ----------------------------------------------------------------------
    # SNR-SPECIFIC CONFUSION MATRICES
    # ----------------------------------------------------------------------

    for target_snr in (-10, 0, 10):

        mask = snrs == target_snr

        if np.any(mask):

            matrices[target_snr] = (
                confusion_matrix(
                    truth[mask],
                    predictions[mask],
                    labels=range(len(CLASSES)),
                )
            )

    # ----------------------------------------------------------------------
    # ACCURACY VS SNR
    # ----------------------------------------------------------------------

    snr_accuracy = {}

    for snr in np.unique(snrs):

        mask = snrs == snr

        if np.any(mask):

            snr_accuracy[int(snr)] = float(
                (
                    predictions[mask]
                    == truth[mask]
                ).mean()
            )

    return {

        "overall": float(
            (truth == predictions).mean()
        ),

        "snr": snr_accuracy,

        "matrices": matrices,
    }


# ============================================================================
# UNIFIED METRICS ACCESS
# ============================================================================

def metrics_for(
    name: str,
) -> dict[str, Any] | None:
    """
    Return evaluation metrics for any dashboard model.
    """

    if name in (
        "CNN1D",
        "CNN-Transformer (1D)",
    ):
        return load_1d_metrics(name)

    return evaluate_2d(name)


# ============================================================================
# CLASS-WISE ACCURACY
# ============================================================================

def accuracy_by_class(
    matrix: np.ndarray,
) -> list[float]:
    """
    Compute recall / per-class accuracy
    from a confusion matrix.
    """

    totals = matrix.sum(axis=1)

    values = []

    for index in range(len(CLASSES)):

        if totals[index] == 0:
            values.append(0.0)

        else:
            values.append(
                float(
                    matrix[index, index]
                    / totals[index]
                )
            )

    return values


# ============================================================================
# ACCURACY VS SNR FIGURE
# ============================================================================

def accuracy_snr_chart(
    all_metrics: dict[str, dict[str, Any] | None]
) -> go.Figure:

    figure = go.Figure()

    for name, metric in all_metrics.items():

        if metric is None:
            continue

        points = sorted(
            metric["snr"].items()
        )

        figure.add_trace(
            go.Scatter(
                x=[
                    point[0]
                    for point in points
                ],

                y=[
                    point[1] * 100
                    for point in points
                ],

                mode="lines+markers",

                name=name,
            )
        )

    figure.update_layout(

        height=450,

        xaxis_title="SNR (dB)",

        yaxis_title="Classification Accuracy (%)",

        yaxis=dict(
            range=[0, 100]
        ),

        hovermode="x unified",

        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.3,
            xanchor="center",
            x=0.5,
        ),

        margin=dict(
            l=20,
            r=20,
            t=30,
            b=70,
        ),
    )

    return figure


# ============================================================================
# MAIN DASHBOARD
# ============================================================================

def main() -> None:

    # ----------------------------------------------------------------------
    # PAGE CONFIGURATION
    # ----------------------------------------------------------------------

    st.set_page_config(
        page_title="RF Signal Classifier Lab",
        layout="wide",
    )

    st.title(
        "RF Signal Classifier Lab"
    )

    st.caption(
        "Research comparison of raw I/Q and STFT-based "
        "modulation classification models."
    )

    # ----------------------------------------------------------------------
    # CHECK MODEL ARTIFACTS
    # ----------------------------------------------------------------------

    missing_models = [

        name

        for name, path in MODEL_FILES.items()

        if not path.exists()

    ]

    if missing_models:

        st.error(
            "Missing model artifact(s): "
            + ", ".join(missing_models)
        )

        st.stop()

    # ----------------------------------------------------------------------
    # CHECK STFT CHECKPOINT METADATA
    # ----------------------------------------------------------------------

    stft_metadata = _checkpoint(
        MODEL_FILES["STFT-RADN (2D)"]
    )

    if (
        stft_metadata.get("model_name")
        == "resnet18"
    ):

        st.warning(
            "Checkpoint warning: stft.pt identifies itself "
            "as a ResNet-18 checkpoint. It is loaded according "
            "to its saved metadata. Do not interpret this artifact "
            "as an independent STFT-RADN experiment until the "
            "correct STFT-RADN checkpoint replaces it."
        )

    # ----------------------------------------------------------------------
    # LOAD METRICS
    # ----------------------------------------------------------------------

    model_names = list(
        MODEL_FILES.keys()
    )

    all_metrics = {

        name: metrics_for(name)

        for name in model_names

    }

    # ======================================================================
    # CREATE DASHBOARD TABS
    # ======================================================================

    performance_tab, analysis_tab, efficiency_tab = st.tabs(

        [
            "Performance",
            "Classification Analysis",
            "Efficiency",
        ]

    )

    # ======================================================================
    # TAB 1 — PERFORMANCE
    # ======================================================================

    with performance_tab:

        st.subheader(
            "Performance Overview"
        )

        rows = []

        for name in model_names:

            model = load_model(name)

            metric = all_metrics[name]

            if metric:

                snr_items = list(
                    metric["snr"].items()
                )

                peak_snr, peak_accuracy = max(
                    snr_items,
                    key=lambda item: item[1],
                )

                overall_accuracy = (
                    metric["overall"] * 100
                )

                peak_accuracy_pct = (
                    peak_accuracy * 100
                )

                peak_snr_display = (
                    f"{peak_snr:+d} dB"
                )

            else:

                overall_accuracy = None

                peak_accuracy_pct = None

                peak_snr_display = (
                    "Unavailable"
                )

            rows.append(

                {

                    "Model": name,

                    "Overall Accuracy": (
                        f"{overall_accuracy:.2f}%"
                        if overall_accuracy is not None
                        else "Unavailable"
                    ),

                    "Peak Accuracy": (
                        f"{peak_accuracy_pct:.2f}%"
                        if peak_accuracy_pct is not None
                        else "Unavailable"
                    ),

                    "Peak SNR": (
                        peak_snr_display
                    ),

                    "Latency (ms/sample)": (
                        f"{measure_latency(name):.3f}"
                    ),

                    "Trainable Parameters": (
                        f"{count_parameters(model):,}"
                    ),
                }

            )

        # --------------------------------------------------------------
        # OVERVIEW TABLE
        # --------------------------------------------------------------

        st.dataframe(
            rows,
            use_container_width=True,
            hide_index=True,
        )

        # --------------------------------------------------------------
        # DATA AVAILABILITY WARNING
        # --------------------------------------------------------------

        unavailable = [

            name

            for name, metric
            in all_metrics.items()

            if metric is None

        ]

        if unavailable:

            missing_data = [

                item

                for item in [

                    "spectrograms_all.npy",
                    "y_all.npy",
                    "snrs_all.npy",
                    "test_idx.npy",

                ]

                if not (
                    ROOT
                    / "data"
                    / item
                ).exists()

            ]

            st.info(
                "2D locked-test metrics are unavailable because "
                "the following processed data files are missing: "
                + ", ".join(missing_data)
                + ". Run dataset_agent.py followed by "
                "precompute_stft.py."
            )

        # --------------------------------------------------------------
        # ACCURACY VS SNR
        # --------------------------------------------------------------

        st.subheader(
            "Robustness Across Signal-to-Noise Ratio"
        )

        st.plotly_chart(

            accuracy_snr_chart(all_metrics),

            use_container_width=True,

        )

    # ======================================================================
    # TAB 2 — CLASSIFICATION ANALYSIS
    # ======================================================================

    with analysis_tab:

        st.subheader(
            "Class-Wise Classification Behaviour"
        )

        left, right = st.columns(2)

        selected_matrix = None
        selected_snr = None

        # --------------------------------------------------------------
        # MODEL AND MATRIX SELECTION
        # --------------------------------------------------------------

        with left:

            selected_model = st.selectbox(

                "Select Model",

                model_names,

                key="matrix_model",

            )

            selected_metric = (
                all_metrics[selected_model]
            )

            if selected_metric:

                matrix_options = list(
                    selected_metric[
                        "matrices"
                    ].keys()
                )

            else:

                matrix_options = []

            if matrix_options:

                selected_snr = st.selectbox(

                    "Evaluation Condition",

                    matrix_options,

                    format_func=lambda value: (
                        f"{value} dB"
                        if isinstance(value, int)
                        else str(value)
                    ),

                )

                selected_matrix = (
                    selected_metric[
                        "matrices"
                    ][selected_snr]
                )

                # ------------------------------------------------------
                # NORMALIZE CONFUSION MATRIX
                # ------------------------------------------------------

                normalized_matrix = (

                    selected_matrix

                    / np.maximum(

                        selected_matrix.sum(
                            axis=1,
                            keepdims=True,
                        ),

                        1,

                    )

                )

                figure = go.Figure(

                    go.Heatmap(

                        z=normalized_matrix,

                        x=CLASSES,

                        y=CLASSES,

                        colorscale="Blues",

                        zmin=0,

                        zmax=1,

                        hovertemplate=(
                            "True: %{y}<br>"
                            "Predicted: %{x}<br>"
                            "Recall: %{z:.1%}"
                            "<extra></extra>"
                        ),

                    )

                )

                condition_label = (

                    f"{selected_snr} dB"

                    if isinstance(
                        selected_snr,
                        int,
                    )

                    else "Overall"

                )

                figure.update_layout(

                    title=(
                        f"{selected_model} "
                        f"Confusion Matrix — "
                        f"{condition_label}"
                    ),

                    height=550,

                    xaxis_title=(
                        "Predicted Class"
                    ),

                    yaxis_title=(
                        "True Class"
                    ),

                    margin=dict(
                        l=20,
                        r=20,
                        t=60,
                        b=100,
                    ),

                )

                st.plotly_chart(

                    figure,

                    use_container_width=True,

                )

            else:

                st.warning(
                    "No confusion matrix is available "
                    "for this model."
                )

        # --------------------------------------------------------------
        # PER CLASS ACCURACY
        # --------------------------------------------------------------

        with right:

            st.subheader(
                "Per-Class Accuracy"
            )

            if selected_matrix is not None:

                class_accuracy = (

                    accuracy_by_class(
                        selected_matrix
                    )

                )

                figure = go.Figure(

                    go.Bar(

                        x=CLASSES,

                        y=[
                            value * 100
                            for value
                            in class_accuracy
                        ],

                    )

                )

                condition_label = (

                    f"{selected_snr} dB"

                    if isinstance(
                        selected_snr,
                        int,
                    )

                    else "Overall"

                )

                figure.update_layout(

                    title=(
                        f"{selected_model} "
                        f"Per-Class Accuracy — "
                        f"{condition_label}"
                    ),

                    height=550,

                    yaxis_title=(
                        "Accuracy (%)"
                    ),

                    yaxis=dict(
                        range=[0, 100]
                    ),

                    xaxis_tickangle=-45,

                    margin=dict(
                        l=20,
                        r=20,
                        t=60,
                        b=120,
                    ),

                )

                st.plotly_chart(

                    figure,

                    use_container_width=True,

                )

            else:

                st.warning(
                    "Per-class accuracy is unavailable "
                    "because no confusion matrix was found."
                )

    # ======================================================================
    # TAB 3 — EFFICIENCY
    # ======================================================================

    with efficiency_tab:

        st.subheader(
            "Deployment Efficiency Comparison"
        )

        efficiency_rows = []

        for name in model_names:

            model = load_model(name)

            latency = measure_latency(name)

            parameters = count_parameters(model)

            efficiency_rows.append(

                {

                    "Model": name,

                    "Latency (ms/sample)": latency,

                    "Trainable Parameters": parameters,

                    "Parameters (Millions)": (
                        parameters / 1_000_000
                    ),

                }

            )

        # --------------------------------------------------------------
        # EFFICIENCY TABLE
        # --------------------------------------------------------------

        st.dataframe(

            efficiency_rows,

            use_container_width=True,

            hide_index=True,

        )

        # --------------------------------------------------------------
        # LATENCY GRAPH
        # --------------------------------------------------------------

        latency_chart = go.Figure(

            go.Bar(

                x=model_names,

                y=[

                    row[
                        "Latency (ms/sample)"
                    ]

                    for row
                    in efficiency_rows

                ],

            )

        )

        latency_chart.update_layout(

            title=(
                "CPU Inference Latency"
            ),

            xaxis_title="Model",

            yaxis_title=(
                "Latency (ms/sample)"
            ),

            height=400,

        )

        st.plotly_chart(

            latency_chart,

            use_container_width=True,

        )

        # --------------------------------------------------------------
        # PARAMETER GRAPH
        # --------------------------------------------------------------

        parameter_chart = go.Figure(

            go.Bar(

                x=model_names,

                y=[

                    row[
                        "Parameters (Millions)"
                    ]

                    for row
                    in efficiency_rows

                ],

            )

        )

        parameter_chart.update_layout(

            title=(
                "Trainable Parameter Footprint"
            ),

            xaxis_title="Model",

            yaxis_title=(
                "Parameters (Millions)"
            ),

            height=400,

        )

        st.plotly_chart(

            parameter_chart,

            use_container_width=True,

        )

        st.caption(
            "Latency protocol: CPU inference, batch size = 1, "
            "evaluation mode, 20 warm-up runs and 200 timed runs. "
            "Latency values are hardware-dependent."
        )


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":

    main()