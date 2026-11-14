"""Run with: python -m fNIRS.breathback.run --stage all"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from .signals import prepare, event_table, candidate_features, SETTINGS, FEATURES
from .detector import Trial, tune, predict, metrics

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = [(kind, quality) for kind in ["short", "long", "both"] for quality in [False, True]]


def write_csv(path, rows):
    if rows:
        with path.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)


def write_json(path, obj):
    def convert(value):
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(type(value).__name__)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=convert), encoding="utf-8")


def variant_name(kind, quality):
    return kind + ("_qc" if quality else "_all")


def preparation(data, out):
    files = sorted(data.glob("*.snirf"))
    if not files:
        raise ValueError(f"No SNIRF files in {data}")
    events = [row for p in files for row in event_table(p)]
    known = [r for r in events if np.isfinite(r["end_s"])]
    # Reserve the three visually inspected participants from the main evaluation summary.
    ordered = sorted(known, key=lambda r: r["duration_s"])
    examples = []
    for target in [0, len(ordered)//2, len(ordered)-1]:
        options = sorted(range(len(ordered)), key=lambda i: abs(i-target))
        examples.append(next(ordered[i] for i in options if ordered[i]["subject_id"] not in {r["subject_id"] for r in examples}))
    development = sorted({r["subject_id"] for r in examples})
    for r in events:
        r["development_subject"] = r["subject_id"] in development
    write_csv(out/"events.csv", events)
    write_csv(out/"development_examples.csv", examples)
    (out/"signals").mkdir(exist_ok=True)
    trials, quality_rows, sources, montage_rows = [], [], [], []
    example_records = {}
    for index, path in enumerate(files, 1):
        print(f"Prepare {index}/{len(files)}: {path.name}", flush=True)
        rec = prepare(path)
        montage_rows.extend(dict(subject_id=rec.subject, file=rec.file, **row) for row in rec.montage_audit)
        sources.append(dict(file=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        np.savez_compressed(out/"signals"/f"subject_{rec.subject:02d}.npz", time=rec.time,
                            od=rec.od, slow=rec.slow, cardiac=rec.cardiac, pulse=rec.pulse,
                            invalid=rec.invalid, motion=rec.motion,
                            column_original_indices=sorted({p[k] for p in rec.pairs for k in ["original_a", "original_b"]}),
                            original_invalid_packed=np.packbits(rec.sample_flags, axis=0),
                            original_invalid_shape=rec.sample_flags.shape)
        for r in (e for e in events if e["file"] == path.name):
            features = candidate_features(rec, r["onset_s"])
            trials.append(Trial(rec.subject, r["trial"], r["onset_s"], r["end_s"], rec.file, features, {}))
            for p, pair in enumerate(rec.pairs):
                sci = features["sci"][:, p]
                quality_rows.append(dict(subject_id=rec.subject, trial=r["trial"], **pair,
                                         median_sci=float(np.nanmedian(sci)) if np.isfinite(sci).any() else np.nan,
                                         fraction_candidates_sci075=float(np.mean(sci >= .75)),
                                         fraction_valid_candidates=float(features["valid"][:, p].mean()),
                                         recording_invalid_pct=100*float(rec.sample_flags[:, [pair["a"], pair["b"]]].mean())))
        if rec.subject in development:
            example_records[rec.subject] = rec
    write_csv(out/"channel_quality.csv", quality_rows)
    write_csv(out/"montage_selection.csv", montage_rows)
    manifest = dict(data_dir=str(data.resolve()), settings=SETTINGS, features=FEATURES,
                    sources=sources, development_subjects=development, labelled_trials=len(known),
                    unlabelled_trials=len(events)-len(known), python=platform.python_version(),
                    versions={m: __import__(m).__version__ for m in ["numpy", "scipy", "sklearn", "h5py"]})
    manifest["code_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in Path(__file__).parent.glob("*.py")}
    write_json(out/"manifest.json", manifest)
    joblib.dump(trials, out/"features.joblib", compress=3)
    from .report import plot_development
    plot_development(example_records, examples, out)
    return trials, manifest


def prediction_row(t, result, method, variant, fold, selected):
    error = result["predicted_end_s"]-t.end if np.isfinite(t.end) else np.nan
    return dict(subject_id=t.subject, trial=t.number, method=method, variant=variant, fold=fold,
                onset_s=t.onset, reference_end_s=t.end, **result, error_s=error,
                abs_error_s=abs(error), selected_parameter=selected.get("parameter", ""),
                sci_threshold=selected.get("threshold"), offset_s=selected.get("offset_s", 0.),
                inner_loss_s=selected.get("inner_loss_s", np.nan))


def evaluate(trials, manifest, out):
    labelled = [t for t in trials if np.isfinite(t.end)]
    ids = sorted({t.subject for t in labelled})
    rows, demos, audits, curves = [], [], [], {}
    for fold, held in enumerate(ids, 1):
        train = [t for t in labelled if t.subject != held]
        test = [t for t in labelled if t.subject == held]
        demo = [t for t in trials if t.subject == held and not np.isfinite(t.end)]
        assert not {t.subject for t in train} & {t.subject for t in test+demo}
        typical = float(np.median([t.end-t.onset for t in train]))
        for t in test+demo:
            endpoint = t.onset+typical
            inside = len(t.features["times"]) and t.features["times"][0] <= endpoint <= t.features["times"][-1]
            result = dict(predicted_end_s=endpoint if inside else np.nan, score=np.nan,
                          available_candidates=len(t.features["times"]), pairs_at_prediction=0,
                          peak_time_s=np.nan,
                          status="estimated" if inside else "outside_recording_or_search")
            (rows if np.isfinite(t.end) else demos).append(prediction_row(t, result, "duration_baseline", "none", fold, {}))
        for kind, quality in VARIANTS:
            variant = variant_name(kind, quality)
            for method in ["rule", "logistic"]:
                fitted, selected, grid = tune(train, kind, quality, method)
                for t in test+demo:
                    result, scores = predict(fitted, t)
                    (rows if np.isfinite(t.end) else demos).append(prediction_row(t, result, method, variant, fold, selected))
                    if kind == "both" and method == "logistic":
                        curves[f"{held}_{t.number}_{variant}"] = dict(times=t.features["times"], scores=scores)
                audits.append(dict(held_subject=held, training_subjects=sorted({t.subject for t in train}),
                                   method=method, variant=variant, selected=selected, inner_grid=grid))
        print(f"LOPO {fold}/{len(ids)}: subject {held} complete", flush=True)
        # Checkpoint each completed outer fold.
        write_csv(out/"predictions_lopo.csv", rows)
        write_csv(out/"predictions_unmarked.csv", demos)
        write_json(out/"fold_audit.json", audits)
    joblib.dump(curves, out/"score_curves.joblib", compress=3)
    summaries = []
    for scope in ["all_lopo", "excluding_development"]:
        available = [r for r in rows if scope == "all_lopo" or r["subject_id"] not in manifest["development_subjects"]]
        for method, variant in sorted({(r["method"], r["variant"]) for r in available}):
            subset = [r for r in available if r["method"] == method and r["variant"] == variant]
            summaries.append(dict(scope=scope, method=method, variant=variant, **metrics(subset)))
    write_csv(out/"metrics.csv", summaries)
    write_csv(out/"paired_comparisons.csv", paired_comparisons(rows, manifest))
    write_csv(out/"quality_comparisons.csv", quality_comparisons(rows, manifest))
    (out/"models").mkdir(exist_ok=True)
    for kind, quality in VARIANTS:
        for method in ["rule", "logistic"]:
            fitted, selected, _ = tune(labelled, kind, quality, method)
            joblib.dump(fitted, out/"models"/f"{method}_{variant_name(kind, quality)}.joblib")
    return rows, demos, summaries


def paired_comparisons(rows, manifest):
    result = []
    for method, variant in sorted({(r["method"], r["variant"]) for r in rows if r["method"] != "duration_baseline"}):
        differences, subjects = [], []
        for sid in sorted({r["subject_id"] for r in rows} - set(manifest["development_subjects"])):
            baseline = {r["trial"]: r["abs_error_s"] for r in rows if r["subject_id"] == sid and r["method"] == "duration_baseline"}
            current = {r["trial"]: r["abs_error_s"] for r in rows if r["subject_id"] == sid and r["method"] == method and r["variant"] == variant}
            # Use identical trials; penalize unavailable outputs as 120 s (also during tuning).
            delta = [(v if np.isfinite(v) else 120.) - (baseline[k] if np.isfinite(baseline[k]) else 120.) for k, v in current.items()]
            differences.append(np.mean(delta)); subjects.append(sid)
        rng = np.random.default_rng(42)
        samples = rng.choice(differences, size=(10000, len(differences)), replace=True).mean(axis=1)
        result.append(dict(method=method, variant=variant, n_subjects=len(subjects),
                           mean_error_difference_vs_baseline_s=float(np.mean(differences)),
                           bootstrap_ci_low_s=float(np.quantile(samples, .025)),
                           bootstrap_ci_high_s=float(np.quantile(samples, .975))))
    return result


def quality_comparisons(rows, manifest):
    result = []
    subset = [r for r in rows if r["subject_id"] not in manifest["development_subjects"]]
    for method in ["rule", "logistic"]:
        for kind in ["short", "long", "both"]:
            a = [r for r in subset if r["method"] == method and r["variant"] == kind+"_all"]
            b = [r for r in subset if r["method"] == method and r["variant"] == kind+"_qc"]
            ma, mb = metrics(a), metrics(b)
            differences = []
            for sid in sorted({r["subject_id"] for r in a}):
                base = {r["trial"]: r["abs_error_s"] for r in a if r["subject_id"] == sid}
                current = {r["trial"]: r["abs_error_s"] for r in b if r["subject_id"] == sid}
                differences.append(np.mean([(v if np.isfinite(v) else 120.) -
                                             (base[k] if np.isfinite(base[k]) else 120.) for k, v in current.items()]))
            rng = np.random.default_rng(42)
            bootstrap = rng.choice(differences, size=(10000, len(differences)), replace=True).mean(axis=1)
            result.append(dict(method=method, kind=kind, n_subjects=len(differences), n_trials=len(a),
                               coverage_all_pct=ma["coverage_pct"], coverage_qc_pct=mb["coverage_pct"],
                               median_abs_error_all_s=ma["median_abs_error_s"],
                               median_abs_error_qc_s=mb["median_abs_error_s"],
                               mean_penalized_error_difference_qc_minus_all_s=float(np.mean(differences)),
                               bootstrap_ci_low_s=float(np.quantile(bootstrap, .025)),
                               bootstrap_ci_high_s=float(np.quantile(bootstrap, .975))))
    return result


def extensions(rows, out, metadata):
    """Exploratory associations of OUT-OF-PERSON errors with participant metadata."""
    from scipy import stats
    if not metadata.exists():
        return
    with metadata.open(encoding="utf-8-sig") as f:
        meta = list(csv.DictReader(f))
    subjects, tests = [], []
    for person in meta:
        sid = int(person["Nr osoby badanej"])
        selected = [r for r in rows if r["subject_id"] == sid and r["method"] == "logistic" and r["variant"] == "both_qc"]
        if not selected:
            continue
        get = lambda prefix: next((v for k, v in person.items() if k.startswith(prefix)), "")
        num = lambda s: float(s.replace(",", ".")) if s.strip() and s != "-" else np.nan
        errors = [r["abs_error_s"] for r in selected if np.isfinite(r["abs_error_s"])]
        subjects.append(dict(subject_id=sid, age=num(get("Wiek")), activity=num(get("Ocena aktywno")),
                             hair_colour=get("Kolor"), hair_type=get("Typ"), hair_texture=get("Tekstura"),
                             median_abs_error_s=float(np.median(errors)) if errors else np.nan,
                             n_predictions=len(errors), n_trials=len(selected)))
    for field in ["age", "activity", "hair_colour", "hair_type", "hair_texture"]:
        valid = [r for r in subjects if np.isfinite(r["median_abs_error_s"]) and (isinstance(r[field], str) or np.isfinite(r[field]))]
        y = np.array([r["median_abs_error_s"] for r in valid])
        rng = np.random.default_rng(42)
        levels = sorted({r[field] for r in valid})
        if len(levels) < 2:
            continue
        if field in ["age", "activity"]:
            x = np.array([r[field] for r in valid])
            ranks_x, ranks_y = stats.rankdata(x), stats.rankdata(y)
            ranks_x -= ranks_x.mean(); ranks_y -= ranks_y.mean()
            denominator = np.linalg.norm(ranks_x)*np.linalg.norm(ranks_y)
            observed = float(np.dot(ranks_x, ranks_y)/denominator)
            permuted = [np.dot(ranks_x, rng.permutation(ranks_y))/denominator for _ in range(9999)]
            p = (1+np.sum(abs(np.array(permuted)) >= abs(observed)-1e-12))/10000
            kind = "Spearman rho"
        else:
            labels = np.array([levels.index(r[field]) for r in valid]); ranks = stats.rankdata(y)
            counts = np.bincount(labels)
            def stat(r):
                sums = np.bincount(labels, weights=r)
                return float(12/(len(r)*(len(r)+1))*np.sum(sums*sums/counts)-3*(len(r)+1))
            observed = stat(ranks)
            p = (1+sum(stat(rng.permutation(ranks)) >= observed-1e-12 for _ in range(9999)))/10000
            kind = "Kruskal H (permutation)"
        tests.append(dict(covariate=field, outcome="participant median LOPO absolute error: logistic both_qc",
                          n_subjects=len(valid), statistic=kind, value=observed, p_permutation=float(p), q_bh=1.))
    q = 1.
    ordered = sorted(tests, key=lambda r: r["p_permutation"])
    for i in range(len(ordered)-1, -1, -1):
        q = min(q, ordered[i]["p_permutation"]*len(ordered)/(i+1)); ordered[i]["q_bh"] = q
    write_csv(out/"participant_extensions.csv", subjects)
    write_csv(out/"exploratory_associations.csv", tests)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT/"fnirs_high_freq_supine/data/BH")
    parser.add_argument("--output", type=Path, default=ROOT/"breathback_results")
    parser.add_argument("--stage", choices=["all", "prepare", "evaluate", "report"], default="all")
    args = parser.parse_args()
    out = args.output; out.mkdir(exist_ok=True, parents=True)
    if args.stage in ["all", "prepare"]:
        trials, manifest = preparation(args.data, out)
    else:
        trials = joblib.load(out/"features.joblib")
        manifest = json.loads((out/"manifest.json").read_text(encoding="utf-8"))
        if manifest.get("settings") != SETTINGS:
            raise ValueError("Cached features use different channel/settings selection. Run --stage all or --stage prepare first.")
    if args.stage in ["all", "evaluate"]:
        with threadpool_limits(limits=1):
            rows, demos, summaries = evaluate(trials, manifest, out)
        extensions(rows, out, ROOT/"fnirs_high_freq_supine/subjects_description.csv")
    if args.stage != "prepare":
        from .report import make_report
        make_report(out, trials, manifest)


if __name__ == "__main__":
    main()
