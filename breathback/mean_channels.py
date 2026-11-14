"""Prepare all-channel mean features and train a participant-held-out model.

Run from the repository root:
    python -m breathback.mean_channels --stage prepare
    python -m breathback.mean_channels --stage train

The preparation stage writes candidate features and reference endpoints to
separate files. The training stage uses leave-one-participant-out evaluation.
"""
import argparse
import csv
import json
from pathlib import Path

import joblib
import numpy as np
import optuna
from scipy.ndimage import uniform_filter1d
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import RobustScaler
import warnings

from .signals import FEATURES, candidate_features, event_table, prepare


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "fnirs_high_freq_supine" / "data" / "BH"
DEFAULT_OUTPUT = ROOT / "breathback_results" / "all_channels_mean"
SEARCH_MIN_S = 4.0
SEARCH_MAX_S = 120.0
PRIOR_MARGIN_S = 10.0
POSITIVE_TOLERANCE_S = 3.0
NEGATIVE_TOLERANCE_S = 6.0
MODEL_TYPES = ("logistic", "mlp")
DEFAULT_OPTUNA_TRIALS = 12
INNER_FOLDS = 3
AUGMENTATION_SIGMAS = (0.0, 0.02, 0.05, 0.1)


def _value(value):
    if value is None:
        return ""
    if isinstance(value, (float, np.floating)):
        return "" if not np.isfinite(value) else f"{float(value):.10g}"
    if isinstance(value, (int, np.integer)):
        return int(value)
    return value


def write_csv(path, rows, fieldnames=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _value(row.get(key)) for key in fieldnames})


def mean_pair_features(features):
    """Arithmetic mean across valid source-detector pairs at each candidate."""
    times = np.asarray(features["times"], dtype=float)
    pair_features = np.asarray(features["pair_features"], dtype=float)
    valid = np.asarray(features["valid"], dtype=bool)
    if pair_features.ndim != 3 or pair_features.shape[2] != 12:
        raise ValueError("Unexpected pair feature array; regenerate the prepared data")
    result = np.full((len(times), len(FEATURES)), np.nan)
    counts = np.zeros(len(times), dtype=int)
    for index, pair_values in enumerate(pair_features):
        usable = valid[index] & np.isfinite(pair_values).all(axis=1)
        selected = pair_values[usable]
        counts[index] = len(selected)
        if not len(selected):
            continue
        average = np.mean(selected, axis=0)
        agreement = abs(np.mean(np.sign(selected[:, 4])))
        coherence = abs(np.mean(selected[:, 2])) / (np.mean(abs(selected[:, 2])) + 1e-6)
        result[index] = np.r_[average[:11], agreement, coherence, average[11]]
    return times, result, counts


def prepare_dataset(data_dir, output_dir):
    files = sorted(data_dir.glob("*.snirf"))
    if not files:
        raise ValueError(f"No SNIRF files found in {data_dir}")

    feature_rows, target_rows, presentation_rows, montage_rows = [], [], [], []
    pair_counts_by_subject = {}
    for file_index, path in enumerate(files, 1):
        recording = prepare(path, channel_region="all")
        pair_counts_by_subject[recording.subject] = len(recording.pairs)
        montage_rows.extend(
            dict(subject_id=recording.subject, file=recording.file, **row)
            for row in recording.montage_audit
        )
        event_rows = event_table(path)
        for event in event_rows:
            target_rows.append(dict(
                subject_id=event["subject_id"], trial=event["trial"],
                onset_s=event["onset_s"], reference_end_s=event["end_s"],
                duration_s=event["duration_s"], reference=event["reference"],
                file=event["file"],
            ))
            features = candidate_features(recording, event["onset_s"])
            times, averaged, counts = mean_pair_features(features)
            for index, candidate_time in enumerate(times):
                row = dict(
                    subject_id=event["subject_id"], trial=event["trial"],
                    onset_s=event["onset_s"], candidate_time_s=float(candidate_time),
                    elapsed_s=float(candidate_time-event["onset_s"]),
                    pairs_used=int(counts[index]), pairs_total=len(recording.pairs),
                    file=event["file"],
                )
                row.update({name: averaged[index, j] for j, name in enumerate(FEATURES)})
                feature_rows.append(row)
                if not np.isfinite(event["end_s"]):
                    presentation_rows.append(row.copy())
        print(f"Prepare {file_index}/{len(files)}: {path.name} ({len(recording.pairs)} par)", flush=True)

    subjects = sorted({int(row["subject_id"]) for row in target_rows
                       if np.isfinite(row["reference_end_s"])})
    folds = []
    for held_out in subjects:
        train_subjects = [subject for subject in subjects if subject != held_out]
        folds.append(dict(
            held_out_subject_id=held_out,
            train_subject_ids=";".join(map(str, train_subjects)),
            n_train_subjects=len(train_subjects),
            n_test_labelled_trials=sum(
                int(row["subject_id"] == held_out and np.isfinite(row["reference_end_s"]))
                for row in target_rows
            ),
        ))

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir/"candidate_features.csv", feature_rows)
    write_csv(output_dir/"event_targets.csv", target_rows)
    write_csv(output_dir/"presentation_features.csv", presentation_rows)
    write_csv(output_dir/"lopo_folds.csv", folds)
    write_csv(output_dir/"all_channel_montage.csv", montage_rows)
    manifest = dict(
        channel_region="all complete source-detector pairs in the SNIRF montage, including pairs outside the forehead-only ROI",
        channel_aggregation="arithmetic mean of valid pair-level features per candidate; two wavelengths averaged within each pair",
        invalid_data="pair-candidates with >5% invalid samples are excluded by the existing feature pipeline",
        target="marker 2 to the next marker 1; endpoint labels are stored separately from candidate features",
        split="leave-one-participant-out; both labelled trials of the held-out participant are test data",
        n_recordings=len(files), n_labelled_trials=int(sum(np.isfinite(r["reference_end_s"]) for r in target_rows)),
        n_unmarked_trials=int(sum(not np.isfinite(r["reference_end_s"]) for r in target_rows)),
        participants=subjects, source_pair_counts=pair_counts_by_subject,
        candidate_feature_names=FEATURES,
        note="The preparation stage does not fit a model. No test participant is mixed into training.",
    )
    (output_dir/"dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Saved {len(feature_rows)} candidate rows; {len(subjects)} participant-held-out folds.", flush=True)


def read_prepared(output_dir):
    features_by_event = {}
    with (output_dir/"candidate_features.csv").open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            key = (int(row["subject_id"]), int(row["trial"]))
            parsed = {name: float(row[name]) if row[name] else np.nan for name in FEATURES}
            parsed.update(candidate_time_s=float(row["candidate_time_s"]),
                          elapsed_s=float(row["elapsed_s"]),
                          pairs_used=int(row["pairs_used"]))
            features_by_event.setdefault(key, []).append(parsed)

    targets = {}
    with (output_dir/"event_targets.csv").open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            key = (int(row["subject_id"]), int(row["trial"]))
            targets[key] = dict(
                subject_id=key[0], trial=key[1], onset_s=float(row["onset_s"]),
                end_s=float(row["reference_end_s"]) if row["reference_end_s"] else np.nan,
                duration_s=float(row["duration_s"]) if row["duration_s"] else np.nan,
                file=row["file"],
            )
    return features_by_event, targets


def duration_bounds(train_events, margin_s=PRIOR_MARGIN_S):
    durations = np.asarray([event["duration_s"] for event in train_events], dtype=float)
    lower = max(SEARCH_MIN_S, float(np.min(durations)-margin_s))
    upper = min(SEARCH_MAX_S, float(np.max(durations)+margin_s))
    if upper < lower:
        raise ValueError("Training-duration prior produced an empty search interval")
    return lower, upper


def training_arrays(train_events, features_by_event, bounds, model_type, params):
    xs, ys, weights, event_keys = [], [], [], []
    lower, upper = bounds
    for event in train_events:
        rows = features_by_event.get((event["subject_id"], event["trial"]), [])
        selected_x, selected_y = [], []
        for row in rows:
            x = np.asarray([row[name] for name in FEATURES], dtype=float)
            if not lower <= row["elapsed_s"] <= upper or not np.isfinite(x).all():
                continue
            distance = abs(row["candidate_time_s"]-event["end_s"])
            if distance <= POSITIVE_TOLERANCE_S:
                selected_x.append(x); selected_y.append(1)
            elif distance >= NEGATIVE_TOLERANCE_S:
                selected_x.append(x); selected_y.append(0)
        y = np.asarray(selected_y, dtype=int)
        if not len(y) or not y.any() or np.all(y):
            continue
        x = np.asarray(selected_x, dtype=float)
        w = np.where(y == 1, .5/max(1, int(y.sum())), .5/max(1, int((y == 0).sum())))
        xs.append(x); ys.append(y); weights.append(w)
        event_keys.append((event["subject_id"], event["trial"]))
    if not xs:
        raise ValueError("No trainable events in this participant split")
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    w = np.concatenate(weights)
    scaler = RobustScaler().fit(x)
    z = np.clip(scaler.transform(x), -10, 10)
    augmentation_sigma = float(params.get("augmentation_sigma", 0.0))
    if augmentation_sigma > 0:
        # One lightly jittered view per training event. A shared shift/scale
        # across that event's candidate windows preserves its temporal pattern.
        augmented, augmented_y, augmented_w = [], [], []
        original_w = w.copy()
        cursor = 0
        for event_y, event_w, (subject_id, trial_number) in zip(ys, weights, event_keys):
            count = len(event_y)
            event_z = z[cursor:cursor+count]
            cursor += count
            rng = np.random.default_rng(42017 + 1009*int(subject_id) + int(trial_number))
            scale = 1.0 + rng.normal(0.0, augmentation_sigma/2, size=(1, z.shape[1]))
            shift = rng.normal(0.0, augmentation_sigma, size=(1, z.shape[1]))
            augmented.append(np.clip(event_z*scale+shift, -10, 10))
            augmented_y.append(event_y)
            augmented_w.append(event_w)
        z = np.concatenate([z, *augmented])
        y = np.concatenate([y, *augmented_y])
        w = np.concatenate([original_w*.5, *[part*.5 for part in augmented_w]])
    w *= (sum(len(part) for part in xs))/w.sum()
    if model_type == "logistic":
        model = LogisticRegression(C=float(params["C"]), max_iter=1000,
                                   solver="liblinear", random_state=42)
    elif model_type == "mlp":
        model = MLPClassifier(
            hidden_layer_sizes=(int(params["hidden_units"]),),
            alpha=float(params["alpha"]),
            learning_rate_init=float(params["learning_rate"]),
            max_iter=300, tol=1e-3, solver="adam", early_stopping=False,
            random_state=42,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        model.fit(z, y, sample_weight=w)
    return scaler, model


def export_final_augmented_dataset(output_dir):
    """Export the final fit's original and synthetic candidate rows for inspection."""
    features_by_event, targets = read_prepared(output_dir)
    labelled = [event for event in targets.values() if np.isfinite(event["end_s"])]
    feature_columns = [f"scaled_{name}" for name in FEATURES]
    exported = []
    trainable_events_by_model = {}
    for model_type in MODEL_TYPES:
        model_path = output_dir/f"mean_all_channels_{model_type}_optuna.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"Train the final model first: {model_path}")
        bundle = joblib.load(model_path)
        params = bundle["best_params"]
        bounds = tuple(bundle["search_bounds_elapsed_s"])
        scaler = bundle["scaler"]
        event_chunks = []
        for event in labelled:
            xs, ys, raw_weights, source_rows = [], [], [], []
            for row in features_by_event.get((event["subject_id"], event["trial"]), []):
                x = np.asarray([row[name] for name in FEATURES], dtype=float)
                if not bounds[0] <= row["elapsed_s"] <= bounds[1] or not np.isfinite(x).all():
                    continue
                distance = abs(row["candidate_time_s"]-event["end_s"])
                if distance <= POSITIVE_TOLERANCE_S:
                    ys.append(1)
                elif distance >= NEGATIVE_TOLERANCE_S:
                    ys.append(0)
                else:
                    continue
                xs.append(x)
                source_rows.append(row)
            y = np.asarray(ys, dtype=int)
            if not len(y) or not y.any() or np.all(y):
                continue
            raw_weights = np.where(y == 1, .5/max(1, int(y.sum())),
                                   .5/max(1, int((y == 0).sum())))
            event_chunks.append((event, np.asarray(xs), y, raw_weights, source_rows))

        if not event_chunks:
            raise ValueError(f"No exportable training rows for {model_type}")
        trainable_events_by_model[model_type] = [
            [int(chunk[0]["subject_id"]), int(chunk[0]["trial"])] for chunk in event_chunks
        ]
        raw_x = np.concatenate([chunk[1] for chunk in event_chunks])
        original_z = np.clip(scaler.transform(raw_x), -10, 10)
        n_original = len(original_z)
        total_weight = sum(chunk[3].sum() for chunk in event_chunks)
        weight_scale = n_original/total_weight
        sigma = float(params.get("augmentation_sigma", 0.0))
        cursor = 0
        for event, event_x, y, raw_weights, source_rows in event_chunks:
            n = len(y)
            event_z = original_z[cursor:cursor+n]
            cursor += n
            base_weights = raw_weights*weight_scale
            for view, values, weights in [("original", event_z, base_weights*(.5 if sigma > 0 else 1.0))]:
                for j, (row, vector) in enumerate(zip(source_rows, values)):
                    record = dict(model=model_type, subject_id=event["subject_id"],
                                  trial=event["trial"], view=view,
                                  candidate_time_s=row["candidate_time_s"],
                                  elapsed_s=row["elapsed_s"], pairs_used=row["pairs_used"],
                                  label=int(y[j]), sample_weight=float(weights[j]))
                    record.update({name: float(vector[k]) for k, name in enumerate(feature_columns)})
                    exported.append(record)
            if sigma > 0:
                rng = np.random.default_rng(42017+1009*int(event["subject_id"])+int(event["trial"]))
                scale = 1.0+rng.normal(0.0, sigma/2, size=(1, original_z.shape[1]))
                shift = rng.normal(0.0, sigma, size=(1, original_z.shape[1]))
                augmented_z = np.clip(event_z*scale+shift, -10, 10)
                for j, (row, vector) in enumerate(zip(source_rows, augmented_z)):
                    record = dict(model=model_type, subject_id=event["subject_id"],
                                  trial=event["trial"], view="augmented",
                                  candidate_time_s=row["candidate_time_s"],
                                  elapsed_s=row["elapsed_s"], pairs_used=row["pairs_used"],
                                  label=int(y[j]), sample_weight=float(base_weights[j]*.5))
                    record.update({name: float(vector[k]) for k, name in enumerate(feature_columns)})
                    exported.append(record)

    write_csv(output_dir/"final_training_augmented.csv", exported)
    manifest_path = output_dir/"augmentation_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    manifest["export"] = "final_training_augmented.csv contains the final full-data training rows only; do not use it to report test performance"
    all_labelled_keys = sorted([[int(event["subject_id"]), int(event["trial"])] for event in labelled])
    manifest["export_rows"] = len(exported)
    manifest["trainable_labelled_events_by_model"] = {
        key: len(value) for key, value in trainable_events_by_model.items()
    }
    manifest["labelled_events_without_trainable_candidates_by_model"] = {
        key: [event_key for event_key in all_labelled_keys if event_key not in value]
        for key, value in trainable_events_by_model.items()
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return len(exported)


def predict_event(event, rows, scaler, model, bounds, smooth_window=3):
    lower, upper = bounds
    in_band = [row for row in rows if lower <= row["elapsed_s"] <= upper]
    valid_rows = []
    valid_x = []
    for row in in_band:
        x = np.asarray([row[name] for name in FEATURES], dtype=float)
        if np.isfinite(x).all():
            valid_rows.append(row)
            valid_x.append(x)
    if not valid_rows:
        return dict(predicted_end_s=np.nan, score=np.nan, pairs_at_prediction=0,
                    available_candidates=0, status="no_usable_candidates")
    probabilities = model.predict_proba(np.clip(scaler.transform(np.asarray(valid_x)), -10, 10))[:, 1]
    times = np.asarray([row["candidate_time_s"] for row in valid_rows])
    # A short uniform smoother avoids choosing a single noisy one-second peak.
    numerator = uniform_filter1d(probabilities, size=int(smooth_window), mode="nearest")
    index = int(np.argmax(numerator))
    return dict(predicted_end_s=float(times[index]), score=float(numerator[index]),
                pairs_at_prediction=int(valid_rows[index]["pairs_used"]),
                available_candidates=len(valid_rows), status="estimated")


def suggest_parameters(trial, model_type):
    params = {
        "prior_margin_s": trial.suggest_int("prior_margin_s", 5, 25, step=5),
        "smooth_window": trial.suggest_categorical("smooth_window", [1, 3, 5]),
        "augmentation_sigma": trial.suggest_categorical("augmentation_sigma", AUGMENTATION_SIGMAS),
    }
    if model_type == "logistic":
        params["C"] = trial.suggest_float("C", 1e-3, 10., log=True)
    elif model_type == "mlp":
        params.update(
            hidden_units=trial.suggest_categorical("hidden_units", [4, 8, 16, 32]),
            alpha=trial.suggest_float("alpha", 1e-4, 1e-1, log=True),
            learning_rate=trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    return params


def inner_grouped_mae(events, features_by_event, model_type, params):
    """Tune only on participant-grouped inner folds of the current training set."""
    groups = np.asarray([event["subject_id"] for event in events])
    n_splits = min(INNER_FOLDS, len(np.unique(groups)))
    if n_splits < 2:
        raise ValueError("Optuna needs at least two training participants")
    splitter = GroupKFold(n_splits=n_splits)
    errors = []
    for train_indices, valid_indices in splitter.split(np.arange(len(events)), groups=groups):
        inner_train = [events[i] for i in train_indices]
        inner_valid = [events[i] for i in valid_indices]
        bounds = duration_bounds(inner_train, params["prior_margin_s"])
        scaler, model = training_arrays(inner_train, features_by_event, bounds,
                                        model_type, params)
        for event in inner_valid:
            result = predict_event(
                event, features_by_event.get((event["subject_id"], event["trial"]), []),
                scaler, model, bounds, params["smooth_window"],
            )
            predicted = result["predicted_end_s"]
            errors.append(abs(predicted-event["end_s"]) if np.isfinite(predicted) else 120.)
    return float(np.mean(errors))


def inner_grouped_within_10s(events, features_by_event, model_type, params):
    """Tune only on participant-grouped inner folds for the ±10 s hit rate."""
    groups = np.asarray([event["subject_id"] for event in events])
    n_splits = min(INNER_FOLDS, len(np.unique(groups)))
    if n_splits < 2:
        raise ValueError("Optuna needs at least two training participants")
    splitter = GroupKFold(n_splits=n_splits)
    hits = []
    for train_indices, valid_indices in splitter.split(np.arange(len(events)), groups=groups):
        inner_train = [events[i] for i in train_indices]
        inner_valid = [events[i] for i in valid_indices]
        bounds = duration_bounds(inner_train, params["prior_margin_s"])
        scaler, model = training_arrays(inner_train, features_by_event, bounds,
                                        model_type, params)
        for event in inner_valid:
            result = predict_event(
                event, features_by_event.get((event["subject_id"], event["trial"]), []),
                scaler, model, bounds, params["smooth_window"],
            )
            predicted = result["predicted_end_s"]
            hits.append(bool(np.isfinite(predicted) and abs(predicted-event["end_s"]) <= 10.0))
    return 1.0-float(np.mean(hits))


def optimize_parameters(events, features_by_event, model_type, n_trials, seed,
                        objective_name="mae"):
    def objective(trial):
        params = suggest_parameters(trial, model_type)
        if objective_name == "within_10s":
            return inner_grouped_within_10s(events, features_by_event, model_type, params)
        return inner_grouped_mae(events, features_by_event, model_type, params)

    sampler = optuna.samplers.TPESampler(seed=seed, n_startup_trials=min(8, n_trials))
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, n_jobs=1, show_progress_bar=False)
    return dict(study.best_params), float(study.best_value), study


def summarize(rows, prediction_key):
    errors = np.asarray([row[prediction_key]-row["reference_end_s"] for row in rows], dtype=float)
    ok = np.isfinite(errors)
    ae = np.abs(errors[ok])
    return dict(
        n_trials=len(rows), n_predictions=int(ok.sum()),
        coverage_pct=100*float(ok.mean()) if len(rows) else np.nan,
        median_abs_error_s=float(np.median(ae)) if len(ae) else np.nan,
        mean_abs_error_s=float(np.mean(ae)) if len(ae) else np.nan,
        median_signed_error_s=float(np.median(errors[ok])) if len(ae) else np.nan,
        within_5s_pct=100*float(np.sum(ae <= 5)/len(rows)) if len(rows) else np.nan,
        within_10s_pct=100*float(np.sum(ae <= 10)/len(rows)) if len(rows) else np.nan,
        n_no_prediction=int((~ok).sum()),
    )


def train_and_evaluate(output_dir, optuna_trials=DEFAULT_OPTUNA_TRIALS,
                       optuna_objective="mae"):
    features_by_event, targets = read_prepared(output_dir)
    labelled = [event for event in targets.values() if np.isfinite(event["end_s"])]
    subjects = sorted({event["subject_id"] for event in labelled})
    if len(subjects) < 2:
        raise ValueError("At least two labelled participants are required")

    if optuna_trials < 1:
        raise ValueError("--optuna-trials must be at least 1")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    predictions, demonstrations, trial_audit, parameter_audit = [], [], [], []
    for fold, held_out in enumerate(subjects, 1):
        train_events = [event for event in labelled if event["subject_id"] != held_out]
        test_events = [event for event in labelled if event["subject_id"] == held_out]
        assert not ({event["subject_id"] for event in train_events} & {held_out})
        typical_duration = float(np.median([event["duration_s"] for event in train_events]))
        for event in test_events:
            predictions.append(dict(
                fold=fold, subject_id=held_out, trial=event["trial"], method="duration_baseline",
                onset_s=event["onset_s"], reference_end_s=event["end_s"],
                predicted_end_s=event["onset_s"]+typical_duration,
                error_s=typical_duration-event["duration_s"],
                duration_baseline_end_s=event["onset_s"]+typical_duration,
                inner_cv_mae_s=np.nan, params_json="", score=np.nan,
                pairs_at_prediction=0, available_candidates=0, status="estimated",
                search_lower_s=np.nan, search_upper_s=np.nan,
            ))

        for model_index, model_type in enumerate(MODEL_TYPES):
            seed = fold*100+model_index
            params, inner_loss, study = optimize_parameters(
                train_events, features_by_event, model_type, optuna_trials, seed,
                optuna_objective,
            )
            inner_mae = (inner_loss if optuna_objective == "mae" else
                         inner_grouped_mae(train_events, features_by_event, model_type, params))
            bounds = duration_bounds(train_events, params["prior_margin_s"])
            scaler, model = training_arrays(train_events, features_by_event, bounds,
                                            model_type, params)
            params_json = json.dumps(params, sort_keys=True)
            parameter_audit.append(dict(
                fold=fold, held_out_subject_id=held_out, model=model_type,
                inner_cv_mae_s=inner_mae, optimization_objective=optuna_objective,
                objective_value=inner_loss, best_params_json=params_json,
            ))
            for optuna_trial in study.trials:
                trial_audit.append(dict(
                    fold=fold, held_out_subject_id=held_out, model=model_type,
                    trial_number=optuna_trial.number, optimization_objective=optuna_objective,
                    objective_value=optuna_trial.value,
                    params_json=json.dumps(optuna_trial.params, sort_keys=True),
                ))
            for event in test_events:
                result = predict_event(
                    event, features_by_event.get((held_out, event["trial"]), []),
                    scaler, model, bounds, params["smooth_window"],
                )
                prediction = result["predicted_end_s"]
                predictions.append(dict(
                    fold=fold, subject_id=held_out, trial=event["trial"],
                    method=f"{model_type}_optuna", onset_s=event["onset_s"],
                    reference_end_s=event["end_s"], predicted_end_s=prediction,
                    error_s=prediction-event["end_s"] if np.isfinite(prediction) else np.nan,
                    duration_baseline_end_s=event["onset_s"]+typical_duration,
                    inner_cv_mae_s=inner_mae, params_json=params_json,
                    score=result["score"], pairs_at_prediction=result["pairs_at_prediction"],
                    available_candidates=result["available_candidates"], status=result["status"],
                    search_lower_s=bounds[0], search_upper_s=bounds[1],
                ))

            for event in targets.values():
                if event["subject_id"] != held_out or np.isfinite(event["end_s"]):
                    continue
                result = predict_event(
                    event, features_by_event.get((held_out, event["trial"]), []),
                    scaler, model, bounds, params["smooth_window"],
                )
                demonstrations.append(dict(
                    fold=fold, subject_id=held_out, trial=event["trial"],
                    method=f"{model_type}_optuna", onset_s=event["onset_s"],
                    duration_baseline_end_s=event["onset_s"]+typical_duration,
                    search_lower_s=bounds[0], search_upper_s=bounds[1],
                    inner_cv_mae_s=inner_mae, params_json=params_json, **result,
                ))

        for event in targets.values():
            if event["subject_id"] != held_out or np.isfinite(event["end_s"]):
                continue
            demonstrations.append(dict(
                fold=fold, subject_id=held_out, trial=event["trial"], method="duration_baseline",
                onset_s=event["onset_s"], predicted_end_s=event["onset_s"]+typical_duration,
                duration_baseline_end_s=event["onset_s"]+typical_duration,
                search_lower_s=np.nan, search_upper_s=np.nan,
                inner_cv_mae_s=np.nan, params_json="", score=np.nan,
                pairs_at_prediction=0, available_candidates=0, status="estimated",
            ))
        print(f"LOPO {fold}/{len(subjects)}: participant {held_out} complete; Optuna tuned logistic + MLP", flush=True)

    prediction_fields = [
        "fold", "subject_id", "trial", "method", "onset_s", "reference_end_s",
        "predicted_end_s", "error_s", "duration_baseline_end_s", "inner_cv_mae_s",
        "params_json", "score", "pairs_at_prediction", "available_candidates", "status",
        "search_lower_s", "search_upper_s",
    ]
    demo_fields = [
        "fold", "subject_id", "trial", "method", "onset_s", "predicted_end_s",
        "duration_baseline_end_s", "search_lower_s", "search_upper_s", "inner_cv_mae_s",
        "params_json", "score", "pairs_at_prediction", "available_candidates", "status",
    ]
    write_csv(output_dir/"lopo_predictions.csv", predictions, prediction_fields)
    write_csv(output_dir/"presentation_predictions.csv", demonstrations, demo_fields)
    write_csv(output_dir/"optuna_trials_lopo.csv", trial_audit)
    write_csv(output_dir/"optuna_best_params_lopo.csv", parameter_audit)
    metric_rows = []
    for method in ["duration_baseline", "logistic_optuna", "mlp_optuna"]:
        subset = [row for row in predictions if row["method"] == method]
        metric_rows.append(dict(method=method, **summarize(subset, "predicted_end_s")))
    metrics = metric_rows
    write_csv(output_dir/"lopo_metrics.csv", metrics)

    final_audit = []
    for model_index, model_type in enumerate(MODEL_TYPES):
        params, inner_loss, study = optimize_parameters(
            labelled, features_by_event, model_type, optuna_trials, 9000+model_index,
            optuna_objective,
        )
        inner_mae = (inner_loss if optuna_objective == "mae" else
                     inner_grouped_mae(labelled, features_by_event, model_type, params))
        final_bounds = duration_bounds(labelled, params["prior_margin_s"])
        scaler, model = training_arrays(labelled, features_by_event, final_bounds,
                                        model_type, params)
        bundle = dict(
            scaler=scaler, model=model, model_type=model_type, feature_names=FEATURES,
            channel_region="all montage pairs, including pairs outside the forehead-only ROI",
            aggregation="arithmetic mean of valid pair-level features",
            search_bounds_elapsed_s=list(final_bounds), best_params=params,
            inner_cv_mae_s=inner_mae, optimization_objective=optuna_objective,
            objective_value=inner_loss, trained_subjects=subjects,
        )
        joblib.dump(bundle, output_dir/f"mean_all_channels_{model_type}_optuna.joblib", compress=3)
        weights = dict(
            model_type=model_type, feature_names=FEATURES, best_params=params,
            inner_cv_mae_s=inner_mae, optimization_objective=optuna_objective,
            objective_value=inner_loss, search_bounds_elapsed_s=list(final_bounds),
            scaler_center=scaler.center_.tolist(), scaler_scale=scaler.scale_.tolist(),
            channel_region=bundle["channel_region"], aggregation=bundle["aggregation"],
        )
        if model_type == "logistic":
            weights.update(coefficients=model.coef_.tolist(), intercept=model.intercept_.tolist())
        else:
            weights.update(
                layer_coefficients=[layer.tolist() for layer in model.coefs_],
                layer_intercepts=[layer.tolist() for layer in model.intercepts_],
            )
        (output_dir/f"mean_all_channels_{model_type}_optuna_weights.json").write_text(
            json.dumps(weights, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        final_audit.append(dict(
            model=model_type, inner_cv_mae_s=inner_mae,
            optimization_objective=optuna_objective, objective_value=inner_loss,
            best_params_json=json.dumps(params, sort_keys=True),
            search_lower_s=final_bounds[0], search_upper_s=final_bounds[1],
        ))
        for optuna_trial in study.trials:
            trial_audit.append(dict(
                fold="final_all_subjects", held_out_subject_id="", model=model_type,
                trial_number=optuna_trial.number, optimization_objective=optuna_objective,
                objective_value=optuna_trial.value,
                params_json=json.dumps(optuna_trial.params, sort_keys=True),
            ))

    write_csv(output_dir/"optuna_final_models.csv", final_audit)
    write_csv(output_dir/"optuna_trials_lopo.csv", trial_audit)
    (output_dir/"augmentation_manifest.json").write_text(json.dumps({
        "method": "one feature-space jittered copy per training event",
        "features": "RobustScaler-transformed candidate features",
        "event_pattern": "one shared per-feature shift and scale across all candidate windows in an event",
        "sigma_search": list(AUGMENTATION_SIGMAS),
        "labels": "copied from the original candidate; endpoints are not perturbed",
        "fold_policy": "augmentation is applied only after participant-grouped train/validation splitting; held-out people are never augmented",
        "selection": "Optuna may select sigma 0.0, which disables augmentation",
        "export": "final_training_augmented.csv contains the final full-data training rows only; do not use it to report test performance",
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    exported_count = export_final_augmented_dataset(output_dir)
    (output_dir/"training_objective.json").write_text(json.dumps({
        "optimization_objective": optuna_objective,
        "within_10s_definition": "absolute endpoint error <= 10 seconds",
        "optuna_trials_per_fit": optuna_trials,
        "evaluation": "outer leave-one-participant-out; hyperparameters selected only in inner participant-grouped folds",
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved nested-LOPO predictions, Optuna audits, final models, and {exported_count} augmented training rows.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA,
                        help="Folder with source SNIRF recordings")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="Folder for prepared data and model artifacts")
    parser.add_argument("--stage", choices=["prepare", "train", "export-augmented", "all"], default="all")
    parser.add_argument("--optuna-trials", type=int, default=DEFAULT_OPTUNA_TRIALS,
                        help=f"Optuna trials per model and CV fit (default: {DEFAULT_OPTUNA_TRIALS})")
    parser.add_argument("--optuna-objective", choices=["mae", "within_10s"], default="mae",
                        help="Inner-CV objective: minimize MAE or maximize the fraction within ±10 s")
    args = parser.parse_args()
    if args.stage in {"prepare", "all"}:
        prepare_dataset(args.data, args.output)
    if args.stage in {"train", "all"}:
        train_and_evaluate(args.output, args.optuna_trials, args.optuna_objective)
    elif args.stage == "export-augmented":
        count = export_final_augmented_dataset(args.output)
        print(f"Saved {count} rows to {args.output/'final_training_augmented.csv'}")


if __name__ == "__main__":
    main()
