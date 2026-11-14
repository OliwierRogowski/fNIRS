#!/usr/bin/env python3
"""Prepare BH fNIRS signals while preserving raw data and quality flags.

Run from the repository root:
    python fnirs_high_freq_supine/prepare_bh_signals.py

Dependencies: h5py, numpy, scipy.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

import h5py
import numpy as np
from scipy.signal import butter, sosfiltfilt


ROOT = Path(__file__).resolve().parent
INPUT_DIR = ROOT / "data" / "BH"
OUTPUT_DIR = ROOT / "derived" / "BH_signals"
QUALITY_CSV = ROOT / "derived" / "BH_channel_quality.csv"

LOWPASS_HZ = 0.5
PULSE_BAND_HZ = (0.5, 2.5)
FILTER_ORDER = 4
SCI_THRESHOLD = 0.75
SNR_THRESHOLD = 5.0
SHORT_MAX_MM = 15.0


def _decode(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _participant_and_initials(filename: str) -> tuple[int, str]:
    match = re.search(r"V2_(\d+)_([A-Za-z]+)_", filename)
    if not match:
        raise ValueError(f"Cannot parse participant ID from {filename}")
    return int(match.group(1)), match.group(2)


def _read_channels(nirs: h5py.Group) -> list[dict]:
    wavelengths = np.asarray(nirs["probe/wavelengths"], dtype=float)
    source_pos = np.asarray(nirs["probe/sourcePos3D"], dtype=float)
    detector_pos = np.asarray(nirs["probe/detectorPos3D"], dtype=float)
    length_unit = _decode(nirs["metaDataTags/LengthUnit"][()])
    unit_scale = {"m": 1000.0, "cm": 10.0, "mm": 1.0}.get(length_unit)
    if unit_scale is None:
        raise ValueError(f"Unsupported probe length unit: {length_unit}")

    data_group = nirs["data1"]
    data = data_group["dataTimeSeries"]
    channel_count = data.shape[1]
    channels = []
    for column in range(channel_count):
        ml = data_group[f"measurementList{column + 1}"]
        source = int(ml["sourceIndex"][()])
        detector = int(ml["detectorIndex"][()])
        wavelength_index = int(ml["wavelengthIndex"][()]) - 1
        distance_mm = float(
            np.linalg.norm(source_pos[source - 1] - detector_pos[detector - 1])
            * unit_scale
        )
        separation = "short" if distance_mm <= SHORT_MAX_MM else "long"
        channels.append(
            {
                "column": column,
                "source": source,
                "detector": detector,
                "wavelength_nm": float(wavelengths[wavelength_index]),
                "distance_mm": distance_mm,
                "separation": separation,
                "label": f"S{source}-D{detector}-{wavelengths[wavelength_index]:g}nm",
            }
        )
    return channels


def _fill_invalid_for_filter(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Interpolate only for filtering; invalid source rows are remasked after."""
    positions = np.flatnonzero(valid)
    if positions.size == 0:
        return np.full(values.shape, np.nan, dtype=float)
    if positions.size == 1:
        return np.full(values.shape, values[positions[0]], dtype=float)
    return np.interp(np.arange(values.size), positions, values[positions])


def _correlation(a: np.ndarray, b: np.ndarray, valid: np.ndarray) -> float:
    if valid.sum() < 3:
        return float("nan")
    a = a[valid]
    b = b[valid]
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _process_file(source_path: Path) -> list[dict]:
    with h5py.File(source_path, "r") as source:
        nirs = source["nirs"]
        data_group = nirs["data1"]
        intensity = np.asarray(data_group["dataTimeSeries"], dtype=float)
        time = np.asarray(data_group["time"], dtype=float)
        channels = _read_channels(nirs)
        missing_raw = np.asarray(nirs["metaDataTags/missing_sample"], dtype=float)
        sample_index = np.asarray(nirs["metaDataTags/sample_index"], dtype=float)

    if intensity.shape[0] != time.size or missing_raw.size != time.size:
        raise ValueError(f"Time/sample dimensions do not match in {source_path.name}")
    if not np.all(np.isfinite(time)) or time.size < 32:
        raise ValueError(f"Invalid or insufficient time vector in {source_path.name}")

    dt = float(np.median(np.diff(time)))
    if dt <= 0:
        raise ValueError(f"Non-increasing time vector in {source_path.name}")
    sampling_rate = 1.0 / dt
    if PULSE_BAND_HZ[1] >= sampling_rate / 2:
        raise ValueError(f"Pulse band exceeds Nyquist frequency in {source_path.name}")

    missing_sample = missing_raw > 0
    nonfinite_or_nonpositive = ~np.isfinite(intensity) | (intensity <= 0)
    invalid_sample = missing_sample[:, None] | nonfinite_or_nonpositive
    valid = ~invalid_sample

    # A sample-index discontinuity is kept as a separate flag. It does not
    # overwrite the uniform time vector or cause a sample to be removed.
    sample_index_discontinuity_after = np.zeros(time.size, dtype=bool)
    if sample_index.size == time.size and time.size > 1:
        sample_index_discontinuity_after[1:] = np.diff(sample_index) != 1

    means = np.full(intensity.shape[1], np.nan, dtype=float)
    raw_snr = np.full(intensity.shape[1], np.nan, dtype=float)
    od = np.full(intensity.shape, np.nan, dtype=float)
    for column in range(intensity.shape[1]):
        channel_valid = valid[:, column]
        if not channel_valid.any():
            continue
        values = intensity[channel_valid, column]
        means[column] = float(np.mean(values))
        std = float(np.std(values))
        raw_snr[column] = means[column] / std if std > 0 else float("inf")
        od[channel_valid, column] = -np.log10(values / means[column])

    slow_sos = butter(
        FILTER_ORDER, LOWPASS_HZ, btype="lowpass", fs=sampling_rate, output="sos"
    )
    pulse_sos = butter(
        FILTER_ORDER,
        PULSE_BAND_HZ,
        btype="bandpass",
        fs=sampling_rate,
        output="sos",
    )
    od_slow = np.full_like(od, np.nan)
    od_pulse = np.full_like(od, np.nan)
    for column in range(od.shape[1]):
        channel_valid = valid[:, column]
        if channel_valid.sum() < 32:
            continue
        filled = _fill_invalid_for_filter(od[:, column], channel_valid)
        try:
            od_slow[:, column] = sosfiltfilt(slow_sos, filled)
            od_pulse[:, column] = sosfiltfilt(pulse_sos, filled)
        except ValueError as exc:
            raise ValueError(f"Filtering failed for {source_path.name}, column {column}") from exc
        od_slow[~channel_valid, column] = np.nan
        od_pulse[~channel_valid, column] = np.nan

    # SCI is calculated for each source-detector pair by correlating the two
    # wavelength traces after cardiac-band filtering. The missing/invalid rows
    # are interpolated only for filtering and omitted from the correlation.
    pair_columns: dict[tuple[int, int], dict[float, int]] = {}
    for channel in channels:
        pair_columns.setdefault((channel["source"], channel["detector"]), {})[
            channel["wavelength_nm"]
        ] = channel["column"]
    sci_by_pair = {}
    for pair, wavelength_columns in pair_columns.items():
        if len(wavelength_columns) != 2:
            sci_by_pair[pair] = float("nan")
            continue
        columns = [wavelength_columns[w] for w in sorted(wavelength_columns)]
        filtered = []
        for column in columns:
            filled = _fill_invalid_for_filter(intensity[:, column], valid[:, column])
            filtered.append(sosfiltfilt(pulse_sos, filled))
        pair_valid = valid[:, columns[0]] & valid[:, columns[1]]
        sci_by_pair[pair] = _correlation(filtered[0], filtered[1], pair_valid)

    participant_id, _ = _participant_and_initials(source_path.name)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    target_path = OUTPUT_DIR / f"{source_path.stem}.h5"
    with h5py.File(target_path, "w") as target:
        target.attrs["source_file"] = source_path.name
        target.attrs["participant_id"] = participant_id
        target.attrs["sampling_rate_hz"] = sampling_rate
        target.attrs["od_formula"] = "-log10(I / I0); I0 is mean valid intensity per channel"
        target.attrs["invalid_rows"] = "missing_sample metadata or nonfinite/nonpositive intensity"
        target.attrs["temporary_gap_handling"] = "linear interpolation for filtering; original invalid rows restored as NaN"
        target.attrs["slow_filter"] = f"order {FILTER_ORDER} Butterworth zero-phase low-pass < {LOWPASS_HZ} Hz"
        target.attrs["pulse_filter"] = (
            f"order {FILTER_ORDER} Butterworth zero-phase band-pass "
            f"{PULSE_BAND_HZ[0]}-{PULSE_BAND_HZ[1]} Hz"
        )
        target.attrs["short_long_split"] = f"short <= {SHORT_MAX_MM} mm; long > {SHORT_MAX_MM} mm"
        target.create_dataset("time_s", data=time, compression="gzip", compression_opts=4)
        target.create_dataset("missing_sample", data=missing_sample, compression="gzip", compression_opts=4)
        target.create_dataset("sample_index", data=sample_index, compression="gzip", compression_opts=4)
        target.create_dataset(
            "nonfinite_or_nonpositive_intensity",
            data=nonfinite_or_nonpositive,
            compression="gzip",
            compression_opts=4,
            shuffle=True,
        )
        target.create_dataset(
            "sample_index_discontinuity_after",
            data=sample_index_discontinuity_after,
            compression="gzip",
            compression_opts=4,
        )

        string_type = h5py.string_dtype(encoding="utf-8")
        target_channels = target.create_group("channels")
        channel_columns = (
            ("original_column", "column", np.int32),
            ("source_index", "source", np.int16),
            ("detector_index", "detector", np.int16),
            ("wavelength_nm", "wavelength_nm", np.float32),
            ("distance_mm", "distance_mm", np.float32),
        )
        for dataset_name, channel_key, dtype in channel_columns:
            target_channels.create_dataset(
                dataset_name,
                data=np.asarray([channel[channel_key] for channel in channels], dtype=dtype),
            )
        target_channels.create_dataset(
            "label", data=np.asarray([channel["label"] for channel in channels], dtype=object), dtype=string_type
        )
        target_channels.create_dataset("mean_valid_intensity_I0", data=means.astype(np.float32))
        target_channels.create_dataset("raw_intensity_snr", data=raw_snr.astype(np.float32))
        target_channels.create_dataset(
            "separation",
            data=np.asarray([channel["separation"] for channel in channels], dtype=object),
            dtype=string_type,
        )

        signals = target.create_group("signals")
        for separation in ("short", "long"):
            indices = np.asarray(
                [channel["column"] for channel in channels if channel["separation"] == separation],
                dtype=int,
            )
            group = signals.create_group(separation)
            group.attrs["channel_count"] = int(indices.size)
            group.create_dataset("channel_index", data=indices.astype(np.int32))
            for name, values in (
                ("od", od),
                ("od_slow", od_slow),
                ("od_pulse", od_pulse),
            ):
                group.create_dataset(
                    name,
                    data=values[:, indices].astype(np.float32),
                    chunks=(min(4096, len(time)), max(1, len(indices))),
                    compression="gzip",
                    compression_opts=4,
                    shuffle=True,
                )
            group.create_dataset(
                "invalid_sample_mask",
                data=invalid_sample[:, indices],
                chunks=(min(4096, len(time)), max(1, len(indices))),
                compression="gzip",
                compression_opts=4,
                shuffle=True,
            )

    quality_rows = []
    for pair, wavelength_columns in sorted(pair_columns.items()):
        pair_channels = [
            channel for channel in channels
            if (channel["source"], channel["detector"]) == pair
        ]
        distance_mm = float(pair_channels[0]["distance_mm"])
        separation = pair_channels[0]["separation"]
        sci = sci_by_pair[pair]
        per_wavelength = {}
        valid_fractions = []
        for channel in pair_channels:
            col = channel["column"]
            per_wavelength[channel["wavelength_nm"]] = raw_snr[col]
            valid_fractions.append(float(valid[:, col].mean()))
        min_snr = min(per_wavelength.values()) if per_wavelength else float("nan")
        valid_fraction = min(valid_fractions) if valid_fractions else float("nan")
        if not np.isfinite(sci):
            status = "not_assessable"
        elif sci < SCI_THRESHOLD or min_snr < SNR_THRESHOLD or valid_fraction < 0.95:
            status = "review"
        else:
            status = "good"
        quality_rows.append(
            {
                "ID osoby": participant_id,
                "Nagranie": source_path.name,
                "Źródło": pair[0],
                "Detektor": pair[1],
                "Odległość [mm]": f"{distance_mm:.2f}",
                "Typ kanału": separation,
                "SCI": "" if not np.isfinite(sci) else f"{sci:.4f}",
                "Próg SCI": SCI_THRESHOLD,
                "SNR 760 nm": _fmt(per_wavelength.get(760.0, float("nan"))),
                "SNR 850 nm": _fmt(per_wavelength.get(850.0, float("nan"))),
                "Udział próbek prawidłowych": _fmt(valid_fraction),
                "Ocena kontaktu": status,
            }
        )
    return quality_rows


def _fmt(value: float) -> str:
    return "" if not np.isfinite(value) else f"{value:.4f}"


def main() -> None:
    files = sorted(INPUT_DIR.glob("*.snirf"), key=lambda p: _participant_and_initials(p.name)[0])
    if not files:
        raise FileNotFoundError(f"No SNIRF files found under {INPUT_DIR}")
    all_quality_rows = []
    for path in files:
        all_quality_rows.extend(_process_file(path))

    QUALITY_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "ID osoby",
        "Nagranie",
        "Źródło",
        "Detektor",
        "Odległość [mm]",
        "Typ kanału",
        "SCI",
        "Próg SCI",
        "SNR 760 nm",
        "SNR 850 nm",
        "Udział próbek prawidłowych",
        "Ocena kontaktu",
    ]
    with QUALITY_CSV.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_quality_rows)

    print(f"Processed {len(files)} recordings into {OUTPUT_DIR}")
    print(f"Wrote {len(all_quality_rows)} channel-pair quality rows to {QUALITY_CSV}")


if __name__ == "__main__":
    main()
