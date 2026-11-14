"""Local data access and inference for the fNIRS prototype."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "fnirs_high_freq_supine" / "data" / "BH"
RESULTS_DIR = ROOT / "breathback_results" / "all_channels_mean"
WEIGHTS_PATH = ROOT / "wagi-dziala.json"

FEATURE_NAMES = [
    "slope_before", "slope_after", "slope_change", "abs_slope_change",
    "level_change", "abs_level_change", "local_variability",
    "log_variance_change", "abs_log_variance_change", "log_pulse_change",
    "abs_log_pulse_change", "direction_agreement", "channel_coherence",
    "relative_level",
]


class PrototypeDataError(RuntimeError):
    """Raised when the local prototype data is missing or inconsistent."""


def load_prototype_data() -> dict[str, object]:
    """Load the local participant/event catalogue, features, metrics and weights."""
    required = [
        RESULTS_DIR / "event_targets.csv",
        RESULTS_DIR / "candidate_features.csv",
        RESULTS_DIR / "lopo_predictions.csv",
        RESULTS_DIR / "lopo_metrics.csv",
        WEIGHTS_PATH,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise PrototypeDataError(
            "Brakuje plików prototypu: " + ", ".join(missing)
            + ". Przygotuj dane poleceniem `python -m breathback.mean_channels --stage all`."
        )

    events = pd.read_csv(required[0], encoding="utf-8-sig")
    features = pd.read_csv(required[1], encoding="utf-8-sig")
    predictions = pd.read_csv(required[2], encoding="utf-8-sig")
    metrics = pd.read_csv(required[3], encoding="utf-8-sig")
    weights = json.loads(WEIGHTS_PATH.read_text(encoding="utf-8"))

    for frame in (events, features, predictions):
        frame["subject_id"] = pd.to_numeric(frame["subject_id"], errors="raise").astype(int)
        frame["trial"] = pd.to_numeric(frame["trial"], errors="raise").astype(int)
    numeric_columns = [
        "onset_s", "reference_end_s", "duration_s", "candidate_time_s", "elapsed_s",
        "pairs_used", "pairs_total", *FEATURE_NAMES,
    ]
    for column in numeric_columns:
        if column in events.columns:
            events[column] = pd.to_numeric(events[column], errors="coerce")
        if column in features.columns:
            features[column] = pd.to_numeric(features[column], errors="coerce")
    for column in [
        "onset_s", "reference_end_s", "predicted_end_s", "error_s",
        "duration_baseline_end_s", "score", "pairs_at_prediction",
        "available_candidates", "search_lower_s", "search_upper_s",
    ]:
        if column in predictions.columns:
            predictions[column] = pd.to_numeric(predictions[column], errors="coerce")

    events = events.sort_values(["subject_id", "trial"]).reset_index(drop=True)
    features = features.sort_values(["subject_id", "trial", "candidate_time_s"]).reset_index(drop=True)
    predictions = predictions.sort_values(["subject_id", "trial", "method"]).reset_index(drop=True)
    if weights.get("model_type") != "mlp" or weights.get("feature_names") != FEATURE_NAMES:
        raise PrototypeDataError("Plik wagi-dziala.json nie pasuje do wejść prototypu (oczekiwany MLP z 14 cechami).")
    return dict(events=events, features=features, predictions=predictions, metrics=metrics, weights=weights)


def infer_from_weight_file(candidate_rows: pd.DataFrame, weights: dict) -> dict[str, object]:
    """Reproduce the exported MLP directly from its JSON weights."""
    lower, upper = map(float, weights["search_bounds_elapsed_s"])
    selected = candidate_rows[
        candidate_rows["elapsed_s"].between(lower, upper, inclusive="both")
    ].copy()
    matrix = selected[FEATURE_NAMES].to_numpy(dtype=float)
    finite = np.isfinite(matrix).all(axis=1)
    selected = selected.loc[finite].reset_index(drop=True)
    matrix = matrix[finite]
    if not len(matrix):
        return dict(status="no_usable_candidates", prediction_s=np.nan, score=np.nan,
                    pairs_used=0, available_candidates=0, scores=pd.DataFrame())

    center = np.asarray(weights["scaler_center"], dtype=float)
    scale = np.asarray(weights["scaler_scale"], dtype=float)
    scaled = np.clip((matrix-center)/scale, -10, 10)
    activation = scaled
    coefficients = weights["layer_coefficients"]
    intercepts = weights["layer_intercepts"]
    for index, (coefficient, intercept) in enumerate(zip(coefficients, intercepts)):
        activation = activation @ np.asarray(coefficient, dtype=float) + np.asarray(intercept, dtype=float)
        if index < len(coefficients)-1:
            activation = np.maximum(activation, 0.0)
    logits = np.clip(activation.reshape(-1), -40.0, 40.0)
    probabilities = 1.0/(1.0+np.exp(-logits))
    smooth_window = max(1, int(weights.get("best_params", {}).get("smooth_window", 1)))
    smoothed = uniform_filter1d(probabilities, size=smooth_window, mode="nearest")
    winner = int(np.argmax(smoothed))
    scores = pd.DataFrame({
        "elapsed_s": selected["elapsed_s"].to_numpy(dtype=float),
        "probability_score": probabilities,
        "smoothed_score": smoothed,
    })
    pairs_used = int(selected.iloc[winner]["pairs_used"]) if "pairs_used" in selected else 0
    return dict(
        status="estimated", prediction_s=float(selected.iloc[winner]["candidate_time_s"]),
        score=float(smoothed[winner]), pairs_used=pairs_used,
        available_candidates=len(selected), scores=scores,
    )


def mean_slow_od_trace(filename: str) -> pd.DataFrame:
    """Read one local SNIRF and average valid slow OD across all montage pairs."""
    path = DATA_DIR / Path(filename).name
    if not path.is_file():
        raise PrototypeDataError(f"Nie znaleziono lokalnego nagrania: {path}")

    from breathback.signals import prepare

    recording = prepare(path, channel_region="all")
    n_samples = len(recording.time)
    signal_sum = np.zeros(n_samples, dtype=float)
    pair_count = np.zeros(n_samples, dtype=int)
    for pair in recording.pairs:
        columns = [pair["a"], pair["b"]]
        valid = ~recording.invalid[:, columns].any(axis=1)
        pair_signal = recording.slow[:, columns].mean(axis=1)
        signal_sum[valid] += pair_signal[valid]
        pair_count[valid] += 1
    mean_signal = np.divide(
        signal_sum, pair_count, out=np.full(n_samples, np.nan), where=pair_count > 0
    )
    return pd.DataFrame({
        "time_s": recording.time,
        "slow_od_mean": mean_signal,
        "pairs_available": pair_count,
    })
