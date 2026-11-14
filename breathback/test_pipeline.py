"""Behavioral checks for endpoint estimation and leakage boundaries.

Run from the project root:
    python -m unittest fNIRS.breathback.test_pipeline
"""
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np
from sklearn.preprocessing import RobustScaler

from . import detector
from .signals import (Recording, FEATURES, aggregate, candidate_features, event_table,
                      search_times, forehead_optodes, select_forehead_pairs, prepare)


def synthetic_recording(transition_amplitude=.02):
    time = np.arange(0, 200, .05)
    # Known endpoint at t=90: a reproducible change of slope.
    base = .01*np.sin(.3*time) + transition_amplitude*np.maximum(time-90, 0)
    slow = base[:, None]*np.array([[1., 1.1, .9, 1.2]])
    cardiac = np.sin(2*np.pi*1.2*time)[:, None]*np.ones((1, 4))
    pairs = [dict(a=0, b=1, kind="short"), dict(a=2, b=3, kind="long")]
    return Recording(1, "synthetic.snirf", time, slow, slow, cardiac,
                     np.ones_like(slow), cardiac, np.zeros_like(slow, dtype=bool),
                     np.zeros((len(time), 6)), pairs, np.array([60.]),
                     np.array([90.]), np.zeros_like(slow, dtype=bool), 20.)


def feature_fixture():
    pair_features = np.empty((2, 2, 12))
    pair_features[:, 0] = 2.
    pair_features[:, 1] = 8.
    return dict(times=np.array([10., 11.]), pair_features=pair_features,
                kinds=np.array(["short", "long"]), valid=np.ones((2, 2), dtype=bool),
                sci=np.array([[.9, .4], [.2, .8]]))


class EventTests(unittest.TestCase):
    def test_markers_use_onsets_and_leave_final_hold_unmarked(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"Breath holding_V2_17_X_BH.snirf"
            with h5py.File(path, "w") as f:
                # Deliberately unsorted event arrays, and nonsensical durations.
                for group, name, times in [("stim9", "2", [250., 70., 160.]),
                                            ("stim4", "1", [190., 0., 90.])]:
                    g = f.create_group("nirs/"+group)
                    g["name"] = name
                    g["data"] = np.column_stack([times, np.full(3, 999.), np.ones(3)])
            rows = event_table(path)
            self.assertEqual([r["onset_s"] for r in rows], [70., 160., 250.])
            self.assertEqual([r["duration_s"] for r in rows[:2]], [20., 30.])
            self.assertTrue(np.isnan(rows[2]["end_s"]))
            self.assertEqual(rows[2]["reference"], "unmarked")
            self.assertEqual(rows[0]["subject_id"], 17)

    def test_end_is_not_borrowed_across_following_hold(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"V2_4_test.snirf"
            with h5py.File(path, "w") as f:
                for group, name, times in [("stim1", "2", [10., 30.]), ("stim2", "1", [40.])]:
                    g = f.create_group("nirs/"+group)
                    g["name"] = name
                    g["data"] = np.column_stack([times, np.ones(len(times)), np.ones(len(times))])
            rows = event_table(path)
            self.assertTrue(np.isnan(rows[0]["end_s"]))
            self.assertEqual(rows[1]["end_s"], 40.)


class FeatureTests(unittest.TestCase):
    def test_search_is_bounded_by_onset_and_recording_only(self):
        times = search_times(60., 200.)
        self.assertEqual(times[0], 64.)
        self.assertEqual(times[-1], 180.)
        self.assertEqual(search_times(60., 80.)[-1], 76.)

    def test_future_markers_do_not_change_features_or_search(self):
        rec = synthetic_recording()
        first = candidate_features(rec, 60.)
        # No marker except the supplied onset may affect data preparation/features.
        altered = replace(rec, onsets=np.array([60., 61., 62.]), breathing=np.array([64., 190.]))
        second = candidate_features(altered, 60.)
        for field in ["times", "pair_features", "sci", "valid"]:
            np.testing.assert_allclose(first[field], second[field], equal_nan=True)

    def test_channel_variants_and_quality_abstention(self):
        features = feature_fixture()
        short, count = aggregate(features, "short")
        self.assertEqual(short.shape, (2, len(FEATURES)))
        np.testing.assert_array_equal(count, [1, 1])
        np.testing.assert_allclose(short[:, 0], 2.)
        long, _ = aggregate(features, "long")
        np.testing.assert_allclose(long[:, 0], 8.)
        both, _ = aggregate(features, "both")
        np.testing.assert_allclose(both[:, 0], 5.)
        quality_short, quality_count = aggregate(features, "short", .75)
        np.testing.assert_array_equal(quality_count, [1, 0])
        self.assertTrue(np.isnan(quality_short[1]).all())
        quality_both, _ = aggregate(features, "both", .75)
        np.testing.assert_allclose(quality_both[:, 0], [2., 8.])

    def test_slope_detector_localizes_synthetic_transition(self):
        rec = synthetic_recording()
        features = candidate_features(rec, 60.)
        trial = detector.Trial(1, 1, 60., 90., rec.file, features, {})
        x, _ = trial.data("both", None)
        # Fixed identity scaling isolates feature localization from extreme-IQR
        # clipping. The separate saturation test covers that numerical boundary.
        scaler = RobustScaler(with_centering=False, with_scaling=False).fit(x)
        fitted = detector.Fitted("both", None, "rule", "slope", 0., scaler, None)
        result, _ = detector.predict(fitted, trial)
        self.assertLessEqual(abs(result["predicted_end_s"]-90.), 1.)
        # Prediction must be invariant if a reference endpoint is unavailable/wrong.
        trial.end = 172.
        changed, _ = detector.predict(fitted, trial)
        self.assertEqual(result, changed)

    def test_extreme_transition_clipping_stays_within_feature_window(self):
        rec = synthetic_recording(transition_amplitude=1.)
        trial = detector.Trial(1, 1, 60., 90., rec.file, candidate_features(rec, 60.), {})
        x, _ = trial.data("both", None)
        fitted = detector.Fitted("both", None, "rule", "slope", 0., RobustScaler().fit(x), None)
        result, scores = detector.predict(fitted, trial)
        self.assertTrue(np.isfinite(scores).all())
        self.assertLessEqual(np.max(scores), 10.)
        self.assertLessEqual(abs(result["predicted_end_s"]-90.), 4.)


class ForeheadSelectionTests(unittest.TestCase):
    def test_forehead_labels_reject_other_frontal_and_frontocentral_rows(self):
        labels = np.array(["Fp1", "Fpz", "AFp1", "AF7", "AFz", "AFF1h", "F3", "FC1"])
        landmarks = np.column_stack([np.arange(len(labels))*100., np.zeros((len(labels), 2))])
        tags = forehead_optodes(landmarks.copy(), landmarks, labels)
        self.assertEqual([t["forehead"] for t in tags], [True]*5+[False]*3)
        self.assertEqual([t["nearest_landmark"] for t in tags], list(labels))

    def test_nearest_landmark_is_selected_from_full_set_not_allowed_subset(self):
        # F3 is nearest; nearby AF7 must not rescue a non-forehead placement.
        landmarks = np.array([[0., 0., 0.], [10., 0., 0.]])
        tag = forehead_optodes(np.array([[1., 0., 0.]]), landmarks, ["F3", "AF7"])[0]
        self.assertEqual(tag["nearest_landmark"], "F3")
        self.assertFalse(tag["forehead"])

    def test_nominal_forehead_match_has_maximum_geometry_distance(self):
        tags = forehead_optodes(np.array([[0., 25., 0.], [0., 25.01, 0.]]),
                                np.array([[0., 0., 0.]]), ["Fp1"])
        self.assertTrue(tags[0]["forehead"])
        self.assertFalse(tags[1]["forehead"])

    def test_both_source_and_detector_must_be_forehead(self):
        yes = dict(nearest_landmark="Fp1", distance_to_landmark_mm=0., forehead=True)
        no = dict(nearest_landmark="F3", distance_to_landmark_mm=0., forehead=False)
        pairs = [dict(source=s, detector=d, a=2*i, b=2*i+1, kind="long", distance_mm=30.)
                 for i, (s, d) in enumerate([(1, 1), (1, 2), (2, 1), (2, 2)])]
        kept, audit = select_forehead_pairs(pairs, [yes, no], [yes, no])
        self.assertEqual([(p["source"], p["detector"]) for p in kept], [(1, 1)])
        self.assertEqual([a["selected"] for a in audit], [True, False, False, False])
        with self.assertRaises(ValueError):
            select_forehead_pairs(pairs, [no, no], [yes, yes])

    def test_excluded_raw_columns_cannot_influence_preprocessing_or_features(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"V2_1_synthetic.snirf"
            t = np.arange(0, 200, 1/70.)
            base = 1000.*np.exp(-.02*np.sin(2*np.pi*1.2*t)-.01*np.sin(.2*t))
            raw = np.column_stack([base, base*2, base*.9, base*3])
            with h5py.File(path, "w") as f:
                n = f.create_group("nirs")
                n["data1/dataTimeSeries"] = raw
                n["data1/time"] = t
                n["metaDataTags/missing_sample"] = np.zeros(len(t))
                n["metaDataTags/LengthUnit"] = "mm"
                n["probe/sourcePos3D"] = [[0., 0., 0.], [100., 0., 0.]]
                n["probe/detectorPos3D"] = [[10., 0., 0.], [110., 0., 0.]]
                n["probe/landmarkPos3D"] = [[0., 0., 0.], [10., 0., 0.], [100., 0., 0.], [110., 0., 0.]]
                n.create_dataset("probe/landmarkLabels", data=["Fp1", "AF7", "F3", "F4"], dtype=h5py.string_dtype())
                for i, (s, d, w) in enumerate([(1, 1, 1), (2, 2, 1), (1, 1, 2), (2, 2, 2)], 1):
                    g = n.create_group(f"data1/measurementList{i}")
                    g["sourceIndex"], g["detectorIndex"], g["wavelengthIndex"] = s, d, w
                for i in range(1, 7):
                    n[f"aux{i}/dataTimeSeries"] = np.zeros((len(t), 1))
            first = prepare(path)
            with h5py.File(path, "a") as f:
                d = f["nirs/data1/dataTimeSeries"]
                d[:, 1] = np.nan
                d[:, 3] = -1e12
            second = prepare(path)
            self.assertEqual(first.od.shape[1], 2)
            self.assertEqual((first.pairs[0]["original_a"], first.pairs[0]["original_b"]), (0, 2))
            self.assertEqual((first.pairs[0]["a"], first.pairs[0]["b"]), (0, 1))
            for field in ["od", "slow", "cardiac", "pulse", "sci_signal", "invalid", "sample_flags"]:
                np.testing.assert_array_equal(getattr(first, field), getattr(second, field))
            a, b = candidate_features(first, 60.), candidate_features(second, 60.)
            for field in ["times", "pair_features", "sci", "valid"]:
                np.testing.assert_allclose(a[field], b[field], equal_nan=True)

    def test_available_supine_montages_keep_only_six_forehead_pairs(self):
        data = Path(__file__).resolve().parents[1]/"fnirs_high_freq_supine/data/BH"
        files = list(data.glob("*.snirf"))
        if not files:
            self.skipTest("Optional local dataset unavailable")
        expected_pairs = {(5, 3), (5, 4), (6, 4), (6, 5), (7, 4), (8, 5)}
        expected_columns = {0, 1, 4, 5, 7, 8, 10, 11, 14, 15, 17, 18}
        for path in files:
            with self.subTest(file=path.name), h5py.File(path) as f:
                n = f["nirs"]
                probe, data_group = n["probe"], n["data1"]
                landmarks = probe["landmarkPos3D"][()]
                labels = probe["landmarkLabels"].asstr()[()].ravel()
                source = probe["sourcePos3D"][()]
                det = probe["detectorPos3D"][()]
                grouped = {}
                for name in data_group:
                    if name.startswith("measurementList") and name[15:].isdigit():
                        m = data_group[name]
                        s, d, w = (int(m[k][()]) for k in ["sourceIndex", "detectorIndex", "wavelengthIndex"])
                        grouped.setdefault((s, d), {})[w] = int(name[15:])-1
                pairs = []
                for (s, d), cols in grouped.items():
                    distance = float(np.linalg.norm(source[s-1]-det[d-1]))
                    pairs.append(dict(source=s, detector=d, a=cols[1], b=cols[2], distance_mm=distance,
                                      kind="short" if distance<15 else "long"))
                kept, _ = select_forehead_pairs(pairs, forehead_optodes(source, landmarks, labels),
                                                forehead_optodes(det, landmarks, labels))
                self.assertEqual({(p["source"], p["detector"]) for p in kept}, expected_pairs)
                self.assertEqual({p[key] for p in kept for key in ["a", "b"]}, expected_columns)
                self.assertEqual(sum(p["kind"]=="short" for p in kept), 2)
                self.assertEqual(sum(p["kind"]=="long" for p in kept), 4)


class EvaluationTests(unittest.TestCase):
    def test_metrics_count_abstentions_in_accuracy_denominator(self):
        result = detector.metrics([{"error_s": -2.}, {"error_s": 9.}, {"error_s": 30.}, {"error_s": np.nan}])
        self.assertEqual(result["n_predictions"], 3)
        self.assertEqual(result["coverage_pct"], 75.)
        self.assertEqual(result["within_5s_pct"], 25.)
        self.assertEqual(result["within_10s_pct"], 50.)
        self.assertEqual(result["n_no_prediction"], 1)
        self.assertEqual(result["n_not_correct_10s"], 2)
        self.assertEqual(result["median_abs_error_s"], 9.)

    def test_inner_tuning_never_predicts_a_training_participant(self):
        features = dict(times=np.arange(41.), pair_features=np.ones((41, 2, 12)),
                        kinds=np.array(["short", "long"]), valid=np.ones((41, 2), dtype=bool),
                        sci=np.ones((41, 2)))
        trials = [detector.Trial(sid, trial, 0., 20., "synthetic", features, {})
                  for sid in range(1, 7) for trial in [1, 2]]
        observed = []

        class FakeFit:
            def __init__(self, people):
                self.people = people
                self.offset_s = 0.

        def fake_fit(training, *args):
            return FakeFit({t.subject for t in training})

        def fake_predict(fitted, trial):
            self.assertNotIn(trial.subject, fitted.people)
            observed.append(trial.subject)
            return {"predicted_end_s": 21.}, np.array([1.])

        with patch.object(detector, "fit", side_effect=fake_fit), patch.object(detector, "predict", side_effect=fake_predict):
            fitted, selected, audit = detector.tune(trials, "both", False, "rule")
        self.assertEqual(set(observed), set(range(1, 7)))
        self.assertEqual(fitted.people, set(range(1, 7)))
        self.assertEqual(selected["offset_s"], -1.)
        self.assertEqual(len(audit), 3)

    def test_no_usable_candidates_returns_explicit_abstention(self):
        features = feature_fixture()
        trial = detector.Trial(1, 1, 0., np.nan, "synthetic", features, {})
        x, _ = trial.data("both", None)
        fitted = detector.Fitted("both", 1., "rule", "slope", 0., RobustScaler().fit(x), None)
        result, scores = detector.predict(fitted, trial)
        self.assertTrue(np.isnan(result["predicted_end_s"]))
        self.assertEqual(result["status"], "no_usable_candidates")
        self.assertTrue(np.isnan(scores).all())

    def test_calibrated_time_remains_in_quality_valid_candidates(self):
        features = feature_fixture()
        trial = detector.Trial(1, 1, 0., np.nan, "synthetic", features, {})
        x, _ = trial.data("short", None)
        fitted = detector.Fitted("short", .75, "rule", "slope", 1., RobustScaler().fit(x), None)
        result, _ = detector.predict(fitted, trial)
        # Raw t=10 is usable; lag correction targets t=11 but that candidate fails SCI.
        self.assertEqual(result["peak_time_s"], 10.)
        self.assertEqual(result["predicted_end_s"], 10.)
        self.assertEqual(result["pairs_at_prediction"], 1)

    def test_untrainable_quality_grid_abstains_without_crashing(self):
        features = dict(times=np.arange(41.), pair_features=np.ones((41, 2, 12)),
                        kinds=np.array(["short", "long"]), valid=np.ones((41, 2), dtype=bool),
                        sci=np.full((41, 2), .1))
        trials = [detector.Trial(sid, 1, 0., 20., "synthetic", features, {}) for sid in range(1, 7)]
        fitted, selected, audit = detector.tune(trials, "both", True, "rule")
        self.assertEqual(selected["status"], "no_trainable_configuration")
        self.assertTrue(all(r["status"] == "not_trainable_in_inner_fold" for r in audit))
        result, _ = detector.predict(fitted, trials[0])
        self.assertTrue(np.isnan(result["predicted_end_s"]))


if __name__ == "__main__":
    unittest.main()
