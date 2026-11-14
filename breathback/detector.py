"""Nested subject-grouped fitting. Ground-truth endpoints never define search windows."""
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import uniform_filter1d
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import RobustScaler

from .signals import aggregate, SETTINGS


@dataclass
class Trial:
    subject: int
    number: int
    onset: float
    end: float
    file: str
    features: dict
    cache: dict

    def data(self, kind, threshold):
        key = (kind, threshold)
        if key not in self.cache:
            self.cache[key] = aggregate(self.features, kind, threshold)
        return self.cache[key]


@dataclass
class Fitted:
    kind: str
    threshold: float | None
    method: str
    parameter: object
    offset_s: float
    scaler: RobustScaler
    model: object
    channel_region: str = "forehead"


class NoTrainableEvents(ValueError):
    pass


def fit(trials, kind, threshold, method, parameter):
    xs, ys, ws = [], [], []
    for t in trials:
        x, _ = t.data(kind, threshold)
        distance = abs(t.features["times"]-t.end)
        valid = np.isfinite(x).all(axis=1)
        # Ambiguous 3--6 s transition band is excluded ONLY from training labels.
        valid &= (distance <= 3) | (distance >= 6)
        y = (distance[valid] <= 3).astype(int)
        if not y.any() or np.all(y):
            continue
        w = np.where(y, .5/max(1, y.sum()), .5/max(1, (1-y).sum()))
        xs.append(x[valid]); ys.append(y); ws.append(w)
    if not xs:
        raise NoTrainableEvents("No trainable events for this quality/channel variant")
    x, y, w = np.concatenate(xs), np.concatenate(ys), np.concatenate(ws)
    scaler = RobustScaler().fit(x)
    model = None
    if method == "logistic":
        # Scale weights to sample count; each trial and each class carry equal total weight.
        w *= len(w)/w.sum()
        model = LogisticRegression(C=float(parameter), max_iter=1000, solver="liblinear", random_state=42)
        model.fit(np.clip(scaler.transform(x), -10, 10), y, sample_weight=w)
    return Fitted(kind, threshold, method, parameter, 0., scaler, model)


def corrected_time(trial, kind, threshold, raw_time, offset):
    x, _ = trial.data(kind, threshold)
    valid_indices = np.flatnonzero(np.isfinite(x).all(axis=1))
    times = trial.features["times"]
    if not len(valid_indices):
        return np.nan, None
    index = int(valid_indices[np.argmin(abs(times[valid_indices]-(raw_time+offset)))])
    return float(times[index]), index


def predict(fitted, trial):
    x, counts = trial.data(fitted.kind, fitted.threshold)
    valid = np.isfinite(x).all(axis=1)
    scores = np.full(len(x), np.nan)
    if not valid.any() or fitted.scaler is None:
        return dict(predicted_end_s=np.nan, score=np.nan, available_candidates=0,
                    pairs_at_prediction=0, peak_time_s=np.nan, status="no_usable_candidates"), scores
    z = np.clip(fitted.scaler.transform(x[valid]), -10, 10)
    if fitted.method == "logistic":
        scores[valid] = fitted.model.predict_proba(z)[:, 1]
    else:
        indices = {"slope": [3], "pulse": [10], "combined": [3, 5, 8, 10]}[fitted.parameter]
        scores[valid] = np.mean(z[:, indices], axis=1)
    # Fixed 3-candidate (3 s) smoothing, with explicit handling of missing candidates.
    numerator = uniform_filter1d(np.nan_to_num(scores), 3, mode="nearest")
    denominator = uniform_filter1d(valid.astype(float), 3, mode="nearest")
    smooth = np.divide(numerator, denominator, out=np.full(len(x), np.nan), where=denominator > 0)
    smooth[~valid] = np.nan
    index = int(np.nanargmax(smooth))
    times = trial.features["times"]
    endpoint, corrected_index = corrected_time(trial, fitted.kind, fitted.threshold, times[index], fitted.offset_s)
    return dict(predicted_end_s=endpoint, score=float(smooth[index]),
                available_candidates=int(valid.sum()), pairs_at_prediction=int(counts[corrected_index]),
                peak_time_s=float(times[index]), status="estimated"), smooth


def tune(trials, kind, quality, method):
    """All tuning, including lag correction and SCI threshold, uses training people only."""
    subjects = np.array([t.subject for t in trials])
    splits = list(GroupKFold(n_splits=min(3, len(set(subjects)))).split(trials, groups=subjects))
    thresholds = SETTINGS["sci_threshold_grid"] if quality else [None]
    parameters = [.1, 1.] if method == "logistic" else ["slope", "pulse", "combined"]
    best = None
    audit = []
    for threshold in thresholds:
        for parameter in parameters:
            refs, predictions, validation_trials = [], [], []
            failed = False
            for train, val in splits:
                training = [trials[i] for i in train]
                validation = [trials[i] for i in val]
                assert not set(t.subject for t in training) & set(t.subject for t in validation)
                try:
                    fitted = fit(training, kind, threshold, method, parameter)
                except NoTrainableEvents:
                    failed = True
                    break
                for t in validation:
                    prediction, _ = predict(fitted, t)
                    refs.append(t.end)
                    predictions.append(prediction["predicted_end_s"])
                    validation_trials.append(t)
            if failed:
                audit.append(dict(threshold=threshold, parameter=parameter, offset_s=0.,
                                  inner_loss_s=120., status="not_trainable_in_inner_fold"))
                continue
            refs, predictions = np.array(refs), np.array(predictions)
            ok = np.isfinite(predictions)
            # Lag is a training-only calibration, NOT an assumed physiological constant.
            offset = float(np.clip(np.median(refs[ok]-predictions[ok]), -8, 8)) if ok.any() else 0.
            corrected = np.array([corrected_time(t, kind, threshold, p, offset)[0] if np.isfinite(p) else np.nan
                                  for t, p in zip(validation_trials, predictions)])
            errors = np.where(ok, np.minimum(abs(corrected-refs), 120), 120.)
            loss = float(np.mean(errors))  # Abstentions cannot improve the tuning loss.
            row = dict(threshold=threshold, parameter=parameter, offset_s=offset, inner_loss_s=loss)
            audit.append(row)
            if best is None or loss < best["inner_loss_s"]:
                best = row
    if best is None:
        best = dict(threshold=thresholds[0], parameter=parameters[0], offset_s=0.,
                    inner_loss_s=120., status="no_trainable_configuration")
        fitted = Fitted(kind, thresholds[0], method, parameters[0], 0., None, None)
        return fitted, best, audit
    fitted = fit(trials, kind, best["threshold"], method, best["parameter"])
    fitted.offset_s = best["offset_s"]
    return fitted, best, audit


def metrics(rows):
    errors = np.array([r["error_s"] for r in rows], dtype=float)
    available = np.isfinite(errors)
    ae = abs(errors[available])
    return dict(n_trials=len(rows), n_predictions=int(available.sum()),
                coverage_pct=100*float(available.mean()),
                median_abs_error_s=float(np.median(ae)) if len(ae) else np.nan,
                median_signed_error_s=float(np.median(errors[available])) if len(ae) else np.nan,
                mean_abs_error_s=float(np.mean(ae)) if len(ae) else np.nan,
                within_5s_pct=100*float(np.sum(ae <= 5)/len(rows)),
                within_10s_pct=100*float(np.sum(ae <= 10)/len(rows)),
                n_no_prediction=int((~available).sum()),
                n_outside_10s=int(np.sum(ae > 10)),
                n_not_correct_10s=int(len(rows)-np.sum(ae <= 10)))
