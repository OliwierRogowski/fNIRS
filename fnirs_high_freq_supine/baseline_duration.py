#!/usr/bin/env python3
"""Median-duration reference predictor for the BH event table.

Run from the repository root:
    python fnirs_high_freq_supine/baseline_duration.py

The evaluation uses leave-one-participant-out predictions: neither labeled
hold from the participant being predicted is used to calculate their median.
"""

from __future__ import annotations

import csv
from decimal import Decimal
from pathlib import Path
from statistics import median


ROOT = Path(__file__).resolve().parent
DERIVED = ROOT / "derived"
EVENT_TABLE = DERIVED / "BH_event_table.csv"
EVALUATION_CSV = DERIVED / "BH_baseline_evaluation.csv"
DEMO_CSV = DERIVED / "BH_baseline_demo.csv"
SUMMARY_MD = DERIVED / "BH_baseline_summary.md"
PRECISION = Decimal("0.000001")


def as_decimal(value: str) -> Decimal:
    return Decimal(value.strip())


def fmt(value: Decimal) -> str:
    return str(value.quantize(PRECISION))


def load_events() -> tuple[list[dict], list[dict]]:
    with EVENT_TABLE.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))

    labeled = []
    demo = []
    for row in rows:
        row["person_id"] = int(row["ID osoby"])
        row["trial"] = int(row["Numer próby"])
        row["start"] = as_decimal(row["Początek [s]"])
        if row["Koniec [s]"] and row["Czas trwania [s]"]:
            row["end"] = as_decimal(row["Koniec [s]"])
            row["duration"] = as_decimal(row["Czas trwania [s]"])
            labeled.append(row)
        else:
            demo.append(row)

    if not labeled or not demo:
        raise ValueError("Event table must include labeled holds and unlabeled demo holds")
    return labeled, demo


def main() -> None:
    labeled, demo = load_events()
    participants = sorted({row["person_id"] for row in labeled})
    evaluation_rows = []
    demo_rows = []
    absolute_errors: dict[int, list[Decimal]] = {1: [], 2: []}

    for participant_id in participants:
        training_rows = [row for row in labeled if row["person_id"] != participant_id]
        if not training_rows:
            raise ValueError(f"No training events available for participant {participant_id}")
        typical_duration = median(row["duration"] for row in training_rows)
        training_subject_count = len({row["person_id"] for row in training_rows})

        for row in labeled:
            if row["person_id"] != participant_id:
                continue
            predicted_end = row["start"] + typical_duration
            absolute_error = abs(predicted_end - row["end"])
            absolute_errors.setdefault(row["trial"], []).append(absolute_error)
            evaluation_rows.append(
                {
                    "ID osoby": participant_id,
                    "Numer próby": row["trial"],
                    "Początek [s]": fmt(row["start"]),
                    "Mediana treningowa [s]": fmt(typical_duration),
                    "Koniec rzeczywisty [s]": fmt(row["end"]),
                    "Koniec przewidziany [s]": fmt(predicted_end),
                    "Błąd bezwzględny [s]": fmt(absolute_error),
                    "Liczba osób treningowych": training_subject_count,
                }
            )

        for row in demo:
            if row["person_id"] != participant_id:
                continue
            predicted_end = row["start"] + typical_duration
            demo_rows.append(
                {
                    "ID osoby": participant_id,
                    "Numer próby": row["trial"],
                    "Początek [s]": fmt(row["start"]),
                    "Mediana treningowa [s]": fmt(typical_duration),
                    "Koniec przewidziany [s]": fmt(predicted_end),
                    "Liczba osób treningowych": training_subject_count,
                }
            )

    evaluation_fields = [
        "ID osoby",
        "Numer próby",
        "Początek [s]",
        "Mediana treningowa [s]",
        "Koniec rzeczywisty [s]",
        "Koniec przewidziany [s]",
        "Błąd bezwzględny [s]",
        "Liczba osób treningowych",
    ]
    demo_fields = [
        "ID osoby",
        "Numer próby",
        "Początek [s]",
        "Mediana treningowa [s]",
        "Koniec przewidziany [s]",
        "Liczba osób treningowych",
    ]
    with EVALUATION_CSV.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=evaluation_fields)
        writer.writeheader()
        writer.writerows(evaluation_rows)
    with DEMO_CSV.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=demo_fields)
        writer.writeheader()
        writer.writerows(demo_rows)

    all_errors = [error for errors in absolute_errors.values() for error in errors]
    mae = sum(all_errors, Decimal(0)) / Decimal(len(all_errors))
    median_error = median(all_errors)
    lines = [
        "# Prosta metoda odniesienia: typowy czas wstrzymania",
        "",
        "Predykcja końca: `początek wstrzymania + mediana czasu trwania w danych treningowych`. ",
        "Mediana jest liczona z oznaczonych prób 1–2; próby 3 nie uczestniczą w treningu.",
        "",
        "## Ocena na oznaczonych końcach",
        "",
        f"- Podział: leave-one-participant-out — obie oznaczone próby danej osoby są wyłączone z jej treningu.",
        f"- Uczestników: {len(participants)}; ocenianych zdarzeń: {len(evaluation_rows)}.",
        f"- MAE: **{mae.quantize(PRECISION)} s**.",
        f"- Mediana błędu bezwzględnego: **{median_error.quantize(PRECISION)} s**.",
        "",
        "MAE jest główną miarą porównania. Detektor powinien uzyskać niższe MAE na tych samych foldach uczestników.",
        "",
        "## Predykcje prób 3",
        "",
        f"- Zapisano {len(demo_rows)} predykcji bez błędu, ponieważ koniec próby 3 nie jest oznaczony.",
        "- Dla każdej osoby użyto mediany treningowej z pozostałych uczestników, zgodnie z oceną leave-one-participant-out.",
        "",
        "## Pliki",
        "",
        "- `BH_baseline_evaluation.csv` — predykcje i błędy dla prób 1–2.",
        "- `BH_baseline_demo.csv` — przewidywane końce prób 3.",
        "",
    ]
    for trial in sorted(absolute_errors):
        errors = absolute_errors[trial]
        if errors:
            trial_mae = sum(errors, Decimal(0)) / Decimal(len(errors))
            lines.append(f"- MAE dla próby {trial}: {trial_mae.quantize(PRECISION)} s ({len(errors)} zdarzeń).")
    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Created {len(evaluation_rows)} evaluation predictions; MAE={mae.quantize(PRECISION)} s")
    print(f"Created {len(demo_rows)} third-trial demo predictions")


if __name__ == "__main__":
    main()
