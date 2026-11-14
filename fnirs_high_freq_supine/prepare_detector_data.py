#!/usr/bin/env python3
"""Build labeled sliding-window features and participant-wise data splits.

This prepares data only. It does not fit, tune, or save a detector.

Run from the repository root:
    python fnirs_high_freq_supine/prepare_detector_data.py
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parent
DERIVED = ROOT / "derived"
SIGNALS_DIR = DERIVED / "BH_signals"
EVENTS_CSV = DERIVED / "BH_event_table.csv"
OUTPUT_TRAIN = DERIVED / "BH_detector_training_LOSO.csv"
OUTPUT_TEST = DERIVED / "BH_detector_test_LOSO.csv"
OUTPUT_DEMO = DERIVED / "BH_detector_presentation.csv"
OUTPUT_FOLDS = DERIVED / "BH_detector_LOSO_folds.csv"
OUTPUT_NOTES = DERIVED / "BH_detector_data_notes.md"
OUTPUT_CHANNELS = DERIVED / "BH_forehead_channels.csv"
QUALITY_CSV = DERIVED / "BH_channel_quality.csv"

WINDOW_SECONDS = 5.0
HOP_SECONDS = 1.0
MIN_VALID_FRACTION = 0.80
VARIANTS = ("short", "long", "both")

FEATURE_FIELDS = [
    "slow_slope_median_od_per_s",
    "slope_change_median_od_per_s",
    "slow_local_sd_median_od",
    "pulse_rms_median_od",
    "channel_change_agreement_median_corr",
]
IDENTIFIER_FIELDS = [
    "cv_test_subject_id",
    "subject_id",
    "trial",
    "channel_variant",
    "window_start_s",
    "window_end_s",
    "elapsed_since_hold_start_s",
    "time_to_end_s",
    "channels_available",
    "channels_used",
]
CSV_FIELDS = IDENTIFIER_FIELDS + FEATURE_FIELDS


def _read_events() -> tuple[list[dict], list[dict]]:
    labeled, presentation = [], []
    with EVENTS_CSV.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            event = {
                "subject_id": int(row["ID osoby"]),
                "trial": int(row["Numer próby"]),
                "start_s": float(row["Początek [s]"]),
            }
            if row["Koniec [s]"] and row["Czas trwania [s]"]:
                event["end_s"] = float(row["Koniec [s]"])
                labeled.append(event)
            else:
                presentation.append(event)
    if not labeled or not presentation:
        raise ValueError("Event table must contain labeled and presentation events")
    return labeled, presentation


def _load_signal_variants(path: Path) -> tuple[np.ndarray, dict[str, tuple[np.ndarray, np.ndarray]]]:
    with h5py.File(path, "r") as processed:
        time_s = np.asarray(processed["time_s"], dtype=float)
        variants = {}
        for separation in ("short", "long"):
            group = processed[f"signals/{separation}"]
            indices = np.asarray(group["channel_index"], dtype=int)
            variants[separation] = (
                np.asarray(group["od_slow"], dtype=float),
                np.asarray(group["od_pulse"], dtype=float),
                indices,
            )

    all_indices = np.concatenate([variants["short"][2], variants["long"][2]])
    order = np.argsort(all_indices)
    both_slow = np.concatenate([variants["short"][0], variants["long"][0]], axis=1)[:, order]
    both_pulse = np.concatenate([variants["short"][1], variants["long"][1]], axis=1)[:, order]
    return time_s, {
        "short": (variants["short"][0], variants["short"][1]),
        "long": (variants["long"][0], variants["long"][1]),
        "both": (both_slow, both_pulse),
    }


def _slope(time_s: np.ndarray, values: np.ndarray, valid: np.ndarray) -> float:
    if valid.sum() < 3:
        return float("nan")
    x = time_s[valid]
    y = values[valid]
    x = x - np.mean(x)
    y = y - np.mean(y)
    denom = float(np.dot(x, x))
    return float(np.dot(x, y) / denom) if denom > 0 else float("nan")


def _window_features(
    time_s: np.ndarray,
    slow: np.ndarray,
    pulse: np.ndarray,
    window_start_s: float,
    window_end_s: float,
) -> dict:
    left = int(np.searchsorted(time_s, window_start_s, side="left"))
    right = int(np.searchsorted(time_s, window_end_s, side="left"))
    sample_time = time_s[left:right]
    slow_window = slow[left:right, :]
    pulse_window = pulse[left:right, :]
    n_samples, n_channels = slow_window.shape
    if n_samples < 3:
        return {"channels_available": n_channels, "channels_used": 0, **{key: float("nan") for key in FEATURE_FIELDS}}

    min_count = max(3, int(np.ceil(MIN_VALID_FRACTION * n_samples)))
    midpoint = (window_start_s + window_end_s) / 2.0
    first_half = sample_time < midpoint
    second_half = ~first_half
    min_half_count = max(3, int(np.ceil(MIN_VALID_FRACTION * max(first_half.sum(), second_half.sum()))))

    channel_slopes = []
    channel_slope_changes = []
    channel_sds = []
    channel_pulse_rms = []
    usable_columns = []
    for column in range(n_channels):
        y = slow_window[:, column]
        p = pulse_window[:, column]
        valid_slow = np.isfinite(y)
        valid_pulse = np.isfinite(p)
        if valid_slow.sum() < min_count or valid_pulse.sum() < min_count:
            continue

        slope_full = _slope(sample_time, y, valid_slow)
        first_valid = valid_slow & first_half
        second_valid = valid_slow & second_half
        slope_first = _slope(sample_time, y, first_valid) if first_valid.sum() >= min_half_count else float("nan")
        slope_second = _slope(sample_time, y, second_valid) if second_valid.sum() >= min_half_count else float("nan")
        channel_slopes.append(slope_full)
        channel_slope_changes.append(slope_second - slope_first if np.isfinite(slope_first) and np.isfinite(slope_second) else float("nan"))
        channel_sds.append(float(np.std(y[valid_slow])))
        channel_pulse_rms.append(float(np.sqrt(np.mean(np.square(p[valid_pulse])))))
        usable_columns.append(column)

    agreement_values = []
    if len(usable_columns) >= 2:
        derivatives = np.diff(slow_window[:, usable_columns], axis=0)
        min_pair_count = max(3, int(np.ceil(MIN_VALID_FRACTION * derivatives.shape[0])))
        for first in range(derivatives.shape[1]):
            for second in range(first + 1, derivatives.shape[1]):
                a = derivatives[:, first]
                b = derivatives[:, second]
                pair_valid = np.isfinite(a) & np.isfinite(b)
                if pair_valid.sum() < min_pair_count:
                    continue
                a, b = a[pair_valid], b[pair_valid]
                if np.std(a) > 0 and np.std(b) > 0:
                    agreement_values.append(float(np.corrcoef(a, b)[0, 1]))

    return {
        "channels_available": n_channels,
        "channels_used": len(usable_columns),
        "slow_slope_median_od_per_s": float(np.nanmedian(channel_slopes)) if channel_slopes else float("nan"),
        "slope_change_median_od_per_s": float(np.nanmedian(channel_slope_changes)) if np.isfinite(channel_slope_changes).any() else float("nan"),
        "slow_local_sd_median_od": float(np.nanmedian(channel_sds)) if channel_sds else float("nan"),
        "pulse_rms_median_od": float(np.nanmedian(channel_pulse_rms)) if channel_pulse_rms else float("nan"),
        "channel_change_agreement_median_corr": float(np.nanmedian(agreement_values)) if agreement_values else float("nan"),
    }


def _format(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(numeric):
        return ""
    return f"{numeric:.9g}"


def _make_event_feature_rows(
    event: dict,
    time_s: np.ndarray,
    variants: dict[str, tuple[np.ndarray, np.ndarray]],
    limit_s: float,
    labeled_end_s: float | None,
) -> list[dict]:
    rows = []
    steps = int(np.floor((limit_s - event["start_s"] - WINDOW_SECONDS) / HOP_SECONDS)) + 1
    for step in range(max(0, steps)):
        window_end = event["start_s"] + WINDOW_SECONDS + step * HOP_SECONDS
        # For labeled holds the sample window ends strictly before the marked
        # end, so no post-end sample can leak into the features.
        if labeled_end_s is not None and window_end >= labeled_end_s:
            continue
        window_start = window_end - WINDOW_SECONDS
        if window_start < time_s[0] or window_end > time_s[-1] + HOP_SECONDS:
            continue
        target = None if labeled_end_s is None else labeled_end_s - window_end
        for variant in VARIANTS:
            slow, pulse = variants[variant]
            features = _window_features(time_s, slow, pulse, window_start, window_end)
            rows.append(
                {
                    "subject_id": event["subject_id"],
                    "trial": event["trial"],
                    "channel_variant": variant,
                    "window_start_s": window_start,
                    "window_end_s": window_end,
                    "elapsed_since_hold_start_s": window_end - event["start_s"],
                    "time_to_end_s": target,
                    **features,
                }
            )
    return rows


def _find_processed_files() -> dict[int, Path]:
    output = {}
    for path in SIGNALS_DIR.glob("*.h5"):
        match = re.search(r"V2_(\d+)_", path.name)
        if match:
            output[int(match.group(1))] = path
    if not output:
        raise FileNotFoundError(f"No prepared signal files found in {SIGNALS_DIR}")
    return output


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _format(row.get(key)) for key in CSV_FIELDS})


def _write_forehead_channel_inventory() -> None:
    """Write a descriptive inventory; it does not filter channels or model inputs."""
    grouped: dict[tuple[int, int, float, str], list[float]] = {}
    with QUALITY_CSV.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            key = (
                int(row["Źródło"]),
                int(row["Detektor"]),
                float(row["Odległość [mm]"]),
                row["Typ kanału"],
            )
            grouped.setdefault(key, []).append(float(row["SCI"]))

    inventory = []
    for (source, detector, distance_mm, channel_type), sci_values in sorted(
        grouped.items(), key=lambda item: (item[0][3], item[0][0], item[0][1])
    ):
        sorted_sci = sorted(sci_values)
        middle = len(sorted_sci) // 2
        median_sci = (
            sorted_sci[middle]
            if len(sorted_sci) % 2
            else (sorted_sci[middle - 1] + sorted_sci[middle]) / 2
        )
        inventory.append({
            "Źródło": source,
            "Detektor": detector,
            "Kanały": f"S{source}-D{detector} przy 760 i 850 nm",
            "Odległość [mm]": f"{distance_mm:.2f}",
            "Typ": channel_type,
            "Mediana SCI (opisowo)": f"{median_sci:.4f}",
            "Nagrania z SCI >= 0,75": sum(value >= 0.75 for value in sci_values),
            "Liczba nagrań": len(sci_values),
            "Obszar": "przedczołowy (czoło)",
            "Uwzględniony": "tak",
        })
    with OUTPUT_CHANNELS.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(inventory[0]))
        writer.writeheader()
        writer.writerows(inventory)


def main() -> None:
    labeled_events, presentation_events = _read_events()
    signal_files = _find_processed_files()
    if set(signal_files) != {event["subject_id"] for event in labeled_events}:
        raise ValueError("Processed signals and labeled event IDs do not match")

    events_by_subject: dict[int, list[dict]] = {}
    for event in labeled_events:
        events_by_subject.setdefault(event["subject_id"], []).append(event)
    demo_by_subject = {event["subject_id"]: event for event in presentation_events}
    subjects = sorted(events_by_subject)
    if set(demo_by_subject) != set(subjects):
        raise ValueError("Expected one unlabeled presentation hold for every labeled participant")

    labeled_features: list[dict] = []
    presentation_features: list[dict] = []
    for subject_id in subjects:
        time_s, variants = _load_signal_variants(signal_files[subject_id])
        for event in events_by_subject[subject_id]:
            labeled_features.extend(
                _make_event_feature_rows(
                    event,
                    time_s,
                    variants,
                    limit_s=event["end_s"],
                    labeled_end_s=event["end_s"],
                )
            )
        demo_event = demo_by_subject[subject_id]
        presentation_features.extend(
            _make_event_feature_rows(
                demo_event,
                time_s,
                variants,
                limit_s=float(time_s[-1]),
                labeled_end_s=None,
            )
        )

    training_rows: list[dict] = []
    test_rows: list[dict] = []
    fold_rows = []
    for held_out_subject in subjects:
        training_subjects = [subject for subject in subjects if subject != held_out_subject]
        fold_rows.append(
            {
                "cv_test_subject_id": held_out_subject,
                "training_subject_ids": ";".join(map(str, training_subjects)),
                "training_subject_count": len(training_subjects),
                "test_event_count": len(events_by_subject[held_out_subject]),
            }
        )
        for row in labeled_features:
            split_row = {
                **row,
                "cv_test_subject_id": held_out_subject,
            }
            if row["subject_id"] == held_out_subject:
                test_rows.append(split_row)
            else:
                training_rows.append(split_row)

    for row in presentation_features:
        row["cv_test_subject_id"] = row["subject_id"]

    _write_csv(OUTPUT_TRAIN, training_rows)
    _write_csv(OUTPUT_TEST, test_rows)
    _write_csv(OUTPUT_DEMO, presentation_features)
    with OUTPUT_FOLDS.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["cv_test_subject_id", "training_subject_ids", "training_subject_count", "test_event_count"],
        )
        writer.writeheader()
        writer.writerows(fold_rows)

    counts_by_split = {
        "training": len(training_rows),
        "test": len(test_rows),
        "presentation": len(presentation_features),
    }
    _write_forehead_channel_inventory()
    OUTPUT_NOTES.write_text(
        "# Dane do pierwszego detektora\n\n"
        "Dane zostały przygotowane do modelowania, ale **żaden model nie został utworzony ani wytrenowany**.\n\n"
        "## Podział\n\n"
        "- Próby 1–2 (56 zdarzeń) są etykietowanym materiałem do uczenia i oceny.\n"
        "- Zastosowano 28 foldów leave-one-participant-out. W każdym foldzie wszystkie wiersze danej osoby są testowe, a jej próby nie występują w treningu. W `BH_detector_training_LOSO.csv` wiersze treningowe powtarzają się między foldami; wybór folda robi się przez `cv_test_subject_id`.\n"
        "- Próba 3 każdej osoby jest materiałem prezentacyjnym. Zapisano okna od początku próby do końca nagrania bez etykiety końca. Dla spójności fold prezentacyjny wyklucza tę osobę z listy treningowej.\n"
        "- `BH_detector_LOSO_folds.csv` zawiera listę osób treningowych i testową dla każdego foldu.\n\n"
        "## Okna i cechy\n\n"
        f"- Okno: {WINDOW_SECONDS:.0f} s; krok: {HOP_SECONDS:.0f} s. Każde okno kończy się przed znacznikiem końca w próbach etykietowanych. Cechy korzystają tylko z próbek dostępnych do końca okna.\n"
        "- Etykieta w próbach 1–2: `time_to_end_s`, czyli czas od końca okna do oznaczonego końca wstrzymania. Nie jest cechą wejściową.\n"
        "- Cechy: mediana nachylenia OD wolnego, mediana zmiany nachylenia między połówkami okna, mediana lokalnego SD OD wolnego, mediana RMS OD pulsacyjnego oraz mediana korelacji zmian między kanałami. Cechy kanałowe agreguje się medianą.\n"
        "- Wszystkie kanały w tym zbiorze pochodzą z montażu przedczołowego (czoło); nie ma tu kanałów z innych obszarów głowy. Warianty `short`, `long`, `both` porównują wyłącznie kanały czołowe. Ich spis jest w `BH_forehead_channels.csv`.\n"
        "- Każdy wariant daje te same 5 zagregowanych cech fNIRS na okno, zamiast osobnej cechy dla każdego kanału. To utrzymuje mały wymiar wejścia; model nie został jeszcze zbudowany ani trenowany.\n"
        "- Kanały nie są globalnie odrzucane według SCI, aby nie wybierać ich na podstawie osób testowych. `channels_used` liczy kanały z co najmniej 80% prawidłowych próbek w oknie. Podsumowanie SCI jest opisowe; późniejsze porównanie odrzucania słabych kanałów trzeba wykonać osobno wewnątrz foldów treningowych.\n"
        "- Opisowo, mediany SCI par kanałów mieszczą się w zakresie 0,9847–0,9974; próg SCI ≥ 0,75 spełnia 25–28 z 28 nagrań dla każdej pary. Nie oznacza to, że jakość jest dobra w każdej próbie i u każdej osoby.\n"
        "- `elapsed_since_hold_start_s` i indeksy kanałów są metadanymi. Model może użyć wyłącznie fNIRS cech, jeśli celem jest sprawdzenie informacji sygnałowej; kolumna czasu od początku pozwala jawnie ocenić wariant korzystający także z czasu.\n"
        "- W plikach prezentacyjnych kolumna `time_to_end_s` jest pusta. Żaden znacznik końca nie został dopisany ani wywnioskowany.\n\n"
        "## Pliki\n\n"
        "- `BH_detector_training_LOSO.csv` — wiersze do uczenia we wszystkich foldach.\n"
        "- `BH_detector_test_LOSO.csv` — wiersze testowe, każda osoba występuje tylko w swoim foldzie.\n"
        "- `BH_detector_presentation.csv` — okna trzecich prób bez etykiet.\n"
        "- `BH_detector_LOSO_folds.csv` — manifest foldów.\n\n"
        "- `BH_forehead_channels.csv` — kanały czołowe używane w porównaniu oraz ich opisowa jakość SCI.\n\n"
        f"Liczba wierszy: trening {counts_by_split['training']:,}, test {counts_by_split['test']:,}, prezentacja {counts_by_split['presentation']:,}.\n"
        "Liczby wierszy nie są liczbą niezależnych przykładów: 5-sekundowe okna nakładają się co 1 s, a wiersze treningowe powtarzają się między foldami. Dane obejmują 56 oznaczonych wstrzymań u 28 osób.\n"
        "Punkt odniesienia do porównania to LOSO MAE 8,210071 s. Detektor należy oceniać na tych samych foldach; jego MAE powinno być niższe.\n",
        encoding="utf-8",
    )

    print(f"Prepared {len(training_rows):,} training rows, {len(test_rows):,} test rows, {len(presentation_features):,} presentation rows")
    print(f"Wrote data only; no model was trained. Details: {OUTPUT_NOTES}")


if __name__ == "__main__":
    main()
