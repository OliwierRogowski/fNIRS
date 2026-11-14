"""Label-blind SNIRF preparation and candidate-window features."""
from dataclasses import dataclass, field
from pathlib import Path
import re

import h5py
import numpy as np
from scipy import signal


FEATURES = ["slope_before", "slope_after", "slope_change", "abs_slope_change",
            "level_change", "abs_level_change", "local_variability", "log_variance_change",
            "abs_log_variance_change", "log_pulse_change", "abs_log_pulse_change",
            "direction_agreement", "channel_coherence", "relative_level"]
# Prespecified engineering settings, not optimized on test labels.
SETTINGS = dict(min_search_s=4., max_search_s=120., step_s=1., half_window_s=4.,
                short_distance_mm=15., max_invalid_fraction=.05,
                slow_lowpass_hz=.4, pulse_band_hz=[.7, 2.5], sci_band_hz=[.7, 1.5],
                downsample_factor=7, sci_threshold_grid=[.5, .75, .9],
                channel_region="forehead", forehead_landmark_max_distance_mm=25.,
                forehead_landmark_pattern=r"(?:Fp|AFp|AF)(?:\d+|z)")


@dataclass
class Recording:
    subject: int
    file: str
    time: np.ndarray
    od: np.ndarray
    slow: np.ndarray
    cardiac: np.ndarray
    pulse: np.ndarray
    sci_signal: np.ndarray
    invalid: np.ndarray
    motion: np.ndarray
    pairs: list
    onsets: np.ndarray
    breathing: np.ndarray
    sample_flags: np.ndarray
    original_fs: float
    montage_audit: list = field(default_factory=list)


def forehead_optodes(positions, landmarks, labels):
    """Operational forehead ROI using the nearest of ALL nominal landmarks.

    AFF/F/FC are intentionally excluded. This checks file geometry, not actual
    contact or individually measured anatomical placement.
    """
    if not len(landmarks) or len(landmarks) != len(labels):
        raise ValueError("Forehead selection requires labelled 3D landmarks")
    result = []
    for position in positions:
        distances = np.linalg.norm(landmarks-position, axis=1)
        index = int(np.argmin(distances))
        label, distance = str(labels[index]), float(distances[index])
        accepted = (np.isfinite(distance) and
                    re.fullmatch(SETTINGS["forehead_landmark_pattern"], label) is not None and
                    distance <= SETTINGS["forehead_landmark_max_distance_mm"])
        result.append(dict(nearest_landmark=label, distance_to_landmark_mm=distance, forehead=bool(accepted)))
    return result


def select_forehead_pairs(pairs, source_tags, detector_tags):
    kept, audit = [], []
    for pair in pairs:
        source = source_tags[pair["source"]-1]
        detector = detector_tags[pair["detector"]-1]
        selected = source["forehead"] and detector["forehead"]
        audit.append(dict(source=pair["source"], detector=pair["detector"],
                          original_a=pair["a"], original_b=pair["b"],
                          kind=pair["kind"], distance_mm=pair["distance_mm"],
                          source_landmark=source["nearest_landmark"],
                          source_landmark_distance_mm=source["distance_to_landmark_mm"],
                          detector_landmark=detector["nearest_landmark"],
                          detector_landmark_distance_mm=detector["distance_to_landmark_mm"],
                          selected=bool(selected), reason="forehead_both_endpoints" if selected else "outside_forehead"))
        if selected:
            kept.append(dict(pair))
    if not kept:
        raise ValueError("No source-detector pair has BOTH optodes in the forehead ROI")
    return kept, audit


def subject_id(filename):
    match = re.search(r"V2_(\d+)_", str(filename))
    if not match:
        raise ValueError(f"Cannot parse supine participant ID: {filename}")
    return int(match[1])


def read_events(path):
    with h5py.File(path, "r") as f:
        n = f["nirs"]
        events = {}
        for name in n:
            if re.fullmatch(r"stim\d+", name):
                g = n[name]
                events[g["name"].asstr()[()]] = np.sort(g["data"][()][:, 0])
        return events.get("2", np.array([])), events.get("1", np.array([]))


def event_table(path):
    starts, ends = read_events(path)
    rows = []
    for trial, start in enumerate(starts, 1):
        next_start = starts[trial] if trial < len(starts) else np.inf
        possible = ends[(ends > start) & (ends < next_start)]
        end = float(possible[0]) if possible.size else np.nan
        rows.append(dict(subject_id=subject_id(path.name), trial=trial, onset_s=float(start),
                         end_s=end, duration_s=end-float(start), file=path.name,
                         reference="marker_1" if possible.size else "unmarked"))
    return rows


def filt(x, fs, cutoffs, kind):
    return signal.sosfiltfilt(signal.butter(3, cutoffs, btype=kind, fs=fs, output="sos"), x, axis=0)


def prepare(path, channel_region="forehead"):
    with h5py.File(path, "r") as f:
        n = f["nirs"]
        d = n["data1"]
        raw = d["dataTimeSeries"][()].astype(float)
        t = d["time"][()].ravel().astype(float)
        if t.size == 2 and raw.shape[0] != 2:
            t = t[0] + np.arange(raw.shape[0])*t[1]
        if len(t) != len(raw) or np.any(np.diff(t) <= 0):
            raise ValueError(f"Invalid time axis: {path}")
        fs = 1/np.median(np.diff(t))
        flags = n["metaDataTags/missing_sample"][()].ravel() != 0
        if len(flags) != len(raw):
            raise ValueError("Unexpected missing_sample shape")
        unit = n["metaDataTags/LengthUnit"].asstr()[()]
        factor = {"mm": 1., "cm": 10., "m": 1000.}[unit]
        source = n["probe/sourcePos3D"][()] * factor
        detector = n["probe/detectorPos3D"][()] * factor
        landmarks = n["probe/landmarkPos3D"][()] * factor
        labels = n["probe/landmarkLabels"].asstr()[()].ravel()
        source_tags = forehead_optodes(source, landmarks, labels)
        detector_tags = forehead_optodes(detector, landmarks, labels)
        grouped = {}
        for name in d:
            match = re.fullmatch(r"measurementList(\d+)", name)
            if match:
                m = d[name]
                s, det, w = (int(m[k][()]) for k in ("sourceIndex", "detectorIndex", "wavelengthIndex"))
                grouped.setdefault((s, det), {})[w] = int(match[1])-1
        pairs = []
        for (s, det), indices in sorted(grouped.items()):
            if 1 not in indices or 2 not in indices:
                continue
            dist = float(np.linalg.norm(source[s-1]-detector[det-1]))
            pairs.append(dict(source=s, detector=det, a=indices[1], b=indices[2], distance_mm=dist,
                              kind="short" if dist < SETTINGS["short_distance_mm"] else "long"))
        if channel_region == "forehead":
            pairs, montage_audit = select_forehead_pairs(pairs, source_tags, detector_tags)
        elif channel_region == "all":
            # Keep every complete source-detector pair in the recorded montage,
            # including pairs omitted by the forehead-only selection. The
            # per-pair audit makes this broader selection explicit.
            montage_audit = []
            for pair in pairs:
                source_tag = source_tags[pair["source"]-1]
                detector_tag = detector_tags[pair["detector"]-1]
                montage_audit.append(dict(
                    source=pair["source"], detector=pair["detector"],
                    original_a=pair["a"], original_b=pair["b"],
                    kind=pair["kind"], distance_mm=pair["distance_mm"],
                    source_landmark=source_tag["nearest_landmark"],
                    source_landmark_distance_mm=source_tag["distance_to_landmark_mm"],
                    detector_landmark=detector_tag["nearest_landmark"],
                    detector_landmark_distance_mm=detector_tag["distance_to_landmark_mm"],
                    selected=True, reason="all_montage_pairs_including_hair"))
        else:
            raise ValueError("channel_region must be 'forehead' or 'all'")
        # Physically discard other signal columns before interpolation, OD, filtering,
        # feature extraction, quality estimation and plotting.
        columns = sorted({p[k] for p in pairs for k in ["a", "b"]})
        raw = raw[:, columns]
        remap = {old: new for new, old in enumerate(columns)}
        for pair in pairs:
            pair["original_a"], pair["original_b"] = pair["a"], pair["b"]
            pair["a"], pair["b"] = remap[pair["a"]], remap[pair["b"]]
        motion = np.column_stack([n[f"aux{i}/dataTimeSeries"][()].ravel() for i in range(1, 7)])
    starts, ends = read_events(path)
    invalid = (~np.isfinite(raw)) | (raw <= 0) | flags[:, None]
    for c in range(raw.shape[1]):
        ok = ~invalid[:, c]
        if ok.sum() < 2:
            raw[:, c] = 1.
            invalid[:, c] = True
        else:
            raw[:, c] = np.interp(t, t[ok], raw[ok, c])
    # The constant I0 only changes the OD intercept, not derivatives or local changes.
    od = -np.log(raw / np.median(raw, axis=0))
    factor = SETTINGS["downsample_factor"]
    od = signal.resample_poly(od, 1, factor, axis=0)
    motion = signal.resample_poly(motion, 1, factor, axis=0)
    td = t[0] + np.arange(len(od))*factor/fs
    invd = np.empty(od.shape, dtype=bool)
    for i in range(len(od)):
        invd[i] = invalid[i*factor:min((i+1)*factor, len(invalid))].any(axis=0)
    fsd = fs/factor
    cardiac = filt(od, fsd, SETTINGS["pulse_band_hz"], "bandpass")
    pulse = filt(np.abs(signal.hilbert(cardiac, axis=0)), fsd, .4, "lowpass")
    return Recording(subject_id(path.name), path.name, td, od,
                     filt(od, fsd, SETTINGS["slow_lowpass_hz"], "lowpass"), cardiac, pulse,
                     filt(od, fsd, SETTINGS["sci_band_hz"], "bandpass"), invd, motion,
                     pairs, starts, ends, invalid, fs, montage_audit)


def correlation(a, b):
    a, b = a-a.mean(axis=0), b-b.mean(axis=0)
    denom = np.sqrt(np.sum(a*a, axis=0)*np.sum(b*b, axis=0))
    return np.divide(np.sum(a*b, axis=0), denom, out=np.full(denom.shape, np.nan), where=denom > 1e-20)


def search_times(onset, recording_end):
    """No endpoint, following marker or duration is accepted by this interface."""
    upper = min(onset+SETTINGS["max_search_s"], recording_end-SETTINGS["half_window_s"])
    return np.arange(onset+SETTINGS["min_search_s"], upper+1e-8, SETTINGS["step_s"])


def candidate_features(rec, onset):
    times = search_times(onset, rec.time[-1])
    baseline = (rec.time >= onset-30) & (rec.time < onset-5)
    if baseline.sum() < 10:
        raise ValueError("Insufficient pre-onset reference")
    scale = np.maximum(np.std(rec.slow[baseline], axis=0), 1e-5)
    z = (rec.slow - np.median(rec.slow[baseline], axis=0))/scale
    a = np.array([p["a"] for p in rec.pairs])
    b = np.array([p["b"] for p in rec.pairs])
    channel_features, scis, valid = [], [], []
    for candidate in times:
        left = (rec.time >= candidate-4) & (rec.time < candidate)
        right = (rec.time >= candidate) & (rec.time < candidate+4)
        window = left | right
        def slope(mask):
            t = rec.time[mask]-np.mean(rec.time[mask])
            return (t @ z[mask])/np.sum(t*t)
        sl, sr = slope(left), slope(right)
        delta = z[right].mean(axis=0)-z[left].mean(axis=0)
        variance = np.log((np.std(z[right], axis=0)+.01)/(np.std(z[left], axis=0)+.01))
        pl = np.maximum(rec.pulse[left].mean(axis=0), 1e-9)
        pr = np.maximum(rec.pulse[right].mean(axis=0), 1e-9)
        pulse = np.log(pr/pl)
        f = np.column_stack([sl, sr, sr-sl, np.abs(sr-sl), delta, np.abs(delta),
                             np.std(z[window], axis=0), variance, np.abs(variance), pulse, np.abs(pulse),
                             z[window].mean(axis=0)])
        # One feature vector per physical source-detector pair (average wavelengths).
        channel_features.append((f[a]+f[b])/2)
        scis.append(correlation(rec.sci_signal[window][:, a], rec.sci_signal[window][:, b]))
        bad_fraction = rec.invalid[window].mean(axis=0)
        valid.append((bad_fraction[a] <= SETTINGS["max_invalid_fraction"]) &
                     (bad_fraction[b] <= SETTINGS["max_invalid_fraction"]))
    return dict(times=times, pair_features=np.array(channel_features), sci=np.array(scis),
                valid=np.array(valid), kinds=np.array([p["kind"] for p in rec.pairs]))


def aggregate(features, kind, sci_threshold=None):
    """Identical aggregation/detector inputs for each channel variant."""
    result = np.full((len(features["times"]), len(FEATURES)), np.nan)
    counts = np.zeros(len(result), dtype=int)
    for i, f in enumerate(features["pair_features"]):
        good = features["valid"][i].copy()
        if kind != "both":
            good &= features["kinds"] == kind
        if sci_threshold is not None:
            good &= features["sci"][i] >= sci_threshold
        good &= np.isfinite(f).all(axis=1)
        counts[i] = good.sum()
        if counts[i] == 0:
            continue
        selected = f[good]
        v = np.median(selected, axis=0)
        agreement = abs(np.mean(np.sign(selected[:, 4])))
        # Agreement of normalized slope changes; defined also for one available pair.
        coherence = abs(np.mean(selected[:, 2]))/(np.mean(abs(selected[:, 2]))+1e-6)
        result[i] = np.r_[v[:11], agreement, coherence, v[11]]
    return np.clip(result, -100, 100), counts
