"""Estimate endpoints in a SNIRF recording using a saved fitted detector."""
import argparse
from pathlib import Path

import joblib
import numpy as np

from .signals import prepare, candidate_features
from .detector import Trial, predict
from .run import ROOT, write_csv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file", type=Path)
    parser.add_argument("--onset", type=float, action="append", help="Known onset in seconds; repeat for multiple holds. Defaults to marker 2.")
    parser.add_argument("--model", type=Path, default=ROOT/"breathback_results/models/logistic_both_qc.joblib")
    parser.add_argument("--output", type=Path, default=ROOT/"breathback_results/new_predictions.csv")
    args = parser.parse_args()
    fitted = joblib.load(args.model)
    if getattr(fitted, "__dict__", {}).get("channel_region") != "forehead":
        parser.error("Model predates forehead-only selection. Retrain with --stage all.")
    recording = prepare(args.file)
    starts = args.onset if args.onset is not None else recording.onsets
    if not len(starts):
        parser.error("No marker 2; supply --onset SECONDS")
    rows = []
    for trial, onset in enumerate(starts, 1):
        features = candidate_features(recording, float(onset))
        event = Trial(recording.subject, trial, float(onset), np.nan, recording.file, features, {})
        result, _ = predict(fitted, event)
        rows.append(dict(subject_id=recording.subject, trial=trial, onset_s=float(onset), **result))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_csv(args.output, rows)
    print(f"Saved {len(rows)} estimates to {args.output}")


if __name__ == "__main__":
    main()
