from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Koniec wstrzymania oddechu · fNIRS",
    page_icon="〰",
    layout="wide",
    initial_sidebar_state="expanded",
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DERIVED_DIR = PROJECT_ROOT / "fnirs_high_freq_supine" / "derived"
BACKEND_DIR = PROJECT_ROOT / "breathback_results"
WEIGHTS_PATH = PROJECT_ROOT / "breathback" / "weights" / "wagi-dziala.json"


@st.cache_data(show_spinner=False)
def read_csv_file(path_text: str, modified: float) -> pd.DataFrame:
    del modified  # Included in the cache key so changed files are reloaded.
    path = Path(path_text)
    if not path.is_file():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError):
        return pd.DataFrame()


def load_csv(path: Path) -> pd.DataFrame:
    modified = path.stat().st_mtime if path.exists() else 0.0
    return read_csv_file(str(path), modified)


@st.cache_data(show_spinner=False)
def load_weights(path_text: str, modified: float) -> dict:
    del modified
    try:
        return json.loads(Path(path_text).read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return {}


def first_available(*paths: Path) -> Path | None:
    return next((path for path in paths if path.is_file()), None)


def load_events() -> tuple[pd.DataFrame, str]:
    path = first_available(BACKEND_DIR / "events.csv", DERIVED_DIR / "BH_event_table.csv")
    if path is None:
        return pd.DataFrame(columns=["ID osoby", "Numer próby", "Początek [s]", "Koniec [s]", "Czas trwania [s]"]), ""
    frame = load_csv(path)
    frame = frame.rename(columns={
        "subject_id": "ID osoby",
        "trial": "Numer próby",
        "onset_s": "Początek [s]",
        "end_s": "Koniec [s]",
        "duration_s": "Czas trwania [s]",
    })
    required = ["ID osoby", "Numer próby", "Początek [s]", "Koniec [s]", "Czas trwania [s]"]
    for column in required:
        if column not in frame:
            frame[column] = np.nan
    return frame, str(path.relative_to(PROJECT_ROOT))


def as_float(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def format_seconds(value: object) -> str:
    number = as_float(value)
    return f"{number:.1f} s" if number is not None else "—"


def select_prediction(frame: pd.DataFrame, subject: int, trial: int) -> pd.Series | None:
    if frame.empty or "subject_id" not in frame or "trial" not in frame:
        return None
    rows = frame[(frame["subject_id"] == subject) & (frame["trial"] == trial)]
    if rows.empty:
        return None
    if "method" in rows:
        preferences = [
            ("logistic", "both_qc"),
            ("logistic", "both_all"),
            ("rule", "both_qc"),
            ("duration_baseline", "none"),
        ]
        for method, variant in preferences:
            if "variant" not in rows:
                continue
            candidate = rows[(rows["method"] == method) & (rows["variant"] == variant)]
            if not candidate.empty:
                return candidate.iloc[0]
    return rows.iloc[0]


def prediction_values(subject: int, trial: int) -> tuple[float | None, float | None, str]:
    # Prefer output produced by the live BreathBack pipeline.
    model_file = BACKEND_DIR / ("predictions_unmarked.csv" if trial == 3 else "predictions_lopo.csv")
    model_row = select_prediction(load_csv(model_file), subject, trial)
    if model_row is not None:
        predicted = as_float(model_row.get("predicted_end_s"))
        reference = as_float(model_row.get("reference_end_s"))
        label = f"Model: {model_row.get('method', 'BreathBack')} · {model_row.get('variant', '')}"
        return predicted, reference, label if predicted is not None else f"{label} · brak wskazania"

    baseline_name = "BH_baseline_demo.csv" if trial == 3 else "BH_baseline_evaluation.csv"
    baseline = load_csv(DERIVED_DIR / baseline_name)
    if baseline.empty:
        return None, None, "Brak predykcji"
    id_col = "ID osoby"
    trial_col = "Numer próby"
    if id_col not in baseline or trial_col not in baseline:
        return None, None, "Brak predykcji"
    row = baseline[(baseline[id_col] == subject) & (baseline[trial_col] == trial)]
    if row.empty:
        return None, None, "Brak predykcji"
    row = row.iloc[0]
    return (
        as_float(row.get("Koniec przewidziany [s]")),
        as_float(row.get("Koniec rzeczywisty [s]")),
        "Punkt odniesienia · typowy czas wstrzymania",
    )


def plot_feature_windows(subject: int, trial: int, variant: str, onset: float, reference: float | None, predicted: float | None, feature: str) -> bool:
    filename = "BH_detector_presentation.csv" if trial == 3 else "BH_detector_test_LOSO.csv"
    frame = load_csv(DERIVED_DIR / filename)
    needed = {"subject_id", "trial", "channel_variant", "elapsed_since_hold_start_s"}
    if frame.empty or not needed.issubset(frame.columns):
        return False
    frame = frame[(frame["subject_id"] == subject) & (frame["trial"] == trial) & (frame["channel_variant"] == variant)]
    features = {
        "Nachylenie wolnego sygnału": "slow_slope_median_od_per_s",
        "Zmiana nachylenia": "slope_change_median_od_per_s",
        "Zmienność lokalna": "slow_local_sd_median_od",
        "Pulsacja": "pulse_rms_median_od",
        "Zgodność kanałów": "channel_change_agreement_median_corr",
    }
    if frame.empty or feature not in features.values() or feature not in frame.columns:
        return False
    plot_frame = frame[["elapsed_since_hold_start_s", feature]].dropna().sort_values("elapsed_since_hold_start_s")
    if plot_frame.empty:
        return False
    st.caption("Pokazano cechę policzoną w 5-sekundowych oknach, nie surowy przebieg fNIRS.")
    st.line_chart(plot_frame.set_index("elapsed_since_hold_start_s"), y=feature, height=300)
    markers = []
    if predicted is not None:
        markers.append(f"przewidywany koniec: {predicted - onset:.1f} s od początku")
    if reference is not None:
        markers.append(f"koniec referencyjny: {reference - onset:.1f} s od początku")
    if markers:
        st.caption("Znaczniki: " + " · ".join(markers))
    return True


st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
    :root { --ink:#14232d; --muted:#71808a; --teal:#078f80; --line:#e5e9e8; --paper:#f6f8f6; }
    .stApp { background:var(--paper); color:var(--ink); font-family:'DM Sans',sans-serif; }
    [data-testid="stHeader"] { background:transparent; }
    section[data-testid="stSidebar"] { background:#17252e; border-right:1px solid #263640; }
    section[data-testid="stSidebar"] > div { padding:1.35rem 1rem; }
    section[data-testid="stSidebar"] * { color:#eef5f5; }
    .brand { display:flex; align-items:center; gap:12px; padding:6px 8px 28px; }
    .brand-mark { color:#22b5a3; font-size:31px; line-height:1; font-weight:700; }
    .brand-name { font-family:'Manrope',sans-serif; font-size:21px; font-weight:800; letter-spacing:-.7px; }
    .brand-sub { color:#a9b6bb; font-size:12px; margin-top:1px; }
    .side-kicker { color:#9babb1 !important; font-size:10px; text-transform:uppercase; letter-spacing:1.5px; margin:0 9px 8px; }
    .side-focus { border:1px solid #405058; background:#202f38; border-radius:10px; padding:12px; margin:0 0 18px; }
    .side-focus-label { color:#9eb0b5; font-size:10px; text-transform:uppercase; letter-spacing:1px; }
    .side-focus-value { color:#fff; font-size:13px; font-weight:700; margin-top:5px; }
    .side-step { display:flex; gap:10px; align-items:flex-start; color:#ced9dc; font-size:12px; padding:9px 7px; line-height:1.45; }
    .side-step b { color:#40c2ae; font-size:11px; }
    .side-footer { position:fixed; bottom:20px; color:#98a9af; font-size:11px; border-top:1px solid #405058; padding:13px 7px 0; width:calc(15.5rem - 2rem); }
    .main .block-container { max-width:1500px; padding-top:2rem; padding-bottom:2.5rem; }
    h1 { font-family:'Manrope',sans-serif !important; font-size:32px !important; font-weight:800 !important; letter-spacing:-1.1px !important; color:var(--ink) !important; margin:0 0 5px !important; }
    h2,h3 { font-family:'Manrope',sans-serif !important; color:var(--ink) !important; letter-spacing:-.35px !important; }
    .eyebrow { color:var(--teal); font-weight:700; font-size:10px; text-transform:uppercase; letter-spacing:1.5px; margin-bottom:7px; }
    .subtitle { color:var(--muted); font-size:14px; margin-bottom:20px; }
    .topline { display:flex; justify-content:space-between; align-items:flex-start; gap:20px; }
    .status-pill { display:inline-flex; align-items:center; gap:8px; border:1px solid #dce6e3; background:#edf7f4; color:#247c70; padding:8px 12px; border-radius:999px; font-size:11px; font-weight:700; margin-top:8px; }
    .status-dot { width:7px; height:7px; border-radius:50%; background:#18a58f; }
    .summary-row { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:12px; margin:5px 0 18px; }
    .summary-card { background:white; border:1px solid var(--line); border-radius:12px; padding:14px 16px; min-height:82px; }
    .summary-label { color:#77858c; font-size:11px; margin-bottom:8px; }
    .summary-value { font-family:'Manrope',sans-serif; color:var(--ink); font-weight:800; font-size:20px; }
    .summary-note { color:#96a0a4; font-size:10px; margin-left:5px; font-weight:500; }
    .panel-title { font-family:'Manrope',sans-serif; font-size:16px; font-weight:800; margin-bottom:3px; }
    .panel-caption { color:#879399; font-size:11px; }
    .empty-state { text-align:center; padding:38px 18px; border:1px dashed #d8e1df; border-radius:11px; background:#fcfdfc; }
    .empty-icon { width:46px; height:46px; margin:0 auto 12px; border-radius:14px; display:flex; align-items:center; justify-content:center; background:#eaf6f3; color:#078f80; font-size:22px; }
    .empty-title { font-family:'Manrope',sans-serif; color:#263740; font-size:15px; font-weight:800; margin-bottom:5px; }
    .empty-copy { color:#829097; font-size:12px; max-width:510px; line-height:1.6; margin:0 auto; }
    .bottom-note { color:#929da1; font-size:10px; text-align:center; margin-top:17px; }
    div[data-testid="stButton"] button { border:1px solid #d7e2df; border-radius:9px; color:#52636b; background:white; font-weight:600; min-height:40px; }
    .stTabs [data-baseweb="tab-list"] { gap:7px; }
    @media (max-width:850px) { .summary-row { grid-template-columns:repeat(2,minmax(0,1fr)); } .topline { display:block; } }
    </style>
    """,
    unsafe_allow_html=True,
)


events, events_source = load_events()
training_preview_path = DERIVED_DIR / "BH_detector_training_LOSO.csv"
training_preview = load_csv(training_preview_path)
weights = load_weights(str(WEIGHTS_PATH), WEIGHTS_PATH.stat().st_mtime if WEIGHTS_PATH.exists() else 0.0)
backend_metrics = load_csv(BACKEND_DIR / "metrics.csv")
baseline_evaluation = load_csv(DERIVED_DIR / "BH_baseline_evaluation.csv")
baseline_demo = load_csv(DERIVED_DIR / "BH_baseline_demo.csv")
expected_features = weights.get("feature_names", [])
available_features = set(training_preview.columns)
missing_weight_features = [name for name in expected_features if name not in available_features]
weight_region = str(weights.get("channel_region", ""))


@st.dialog("Podgląd okien treningowych", width="large")
def show_training_preview() -> None:
    if training_preview.empty:
        st.info("Nie znaleziono przygotowanych okien treningowych.")
        return
    st.caption("To przygotowane cechy w przesuwających się oknach. Wagi pobrane z brancha data wymagają 14 cech, a ten plik zawiera ich 5 — ten podgląd nie może być użyty do predykcji tym modelem.")
    if "cv_test_subject_id" in training_preview:
        fold_ids = sorted(training_preview["cv_test_subject_id"].dropna().unique().tolist())
        held_out = st.selectbox("Osoba pominięta w tym foldzie", fold_ids, key="training_fold_subject")
        rows = training_preview[training_preview["cv_test_subject_id"] == held_out]
        st.caption(f"Fold osoby {held_out}: {len(rows):,} wierszy treningowych; jej rekordy nie są w tym foldzie testowym.")
    else:
        rows = training_preview
    st.dataframe(rows.head(500), use_container_width=True, hide_index=True)
    if len(rows) > 500:
        st.caption("Pokazano pierwsze 500 wierszy tego foldu.")


with st.sidebar:
    st.markdown('<div class="brand"><div class="brand-mark">〰</div><div><div class="brand-name">fNIRS</div><div class="brand-sub">Detektor wstrzymania oddechu</div></div></div>', unsafe_allow_html=True)
    st.markdown('<div class="side-kicker">Cel</div><div class="side-focus"><div class="side-focus-label">Szacujemy</div><div class="side-focus-value">Koniec wstrzymania z sygnału fNIRS</div></div>', unsafe_allow_html=True)
    st.markdown('<div class="side-kicker">Pipeline</div><div class="side-step"><b>01</b><span>Zdarzenia z markerów</span></div><div class="side-step"><b>02</b><span>Cechy i detektor</span></div><div class="side-step"><b>03</b><span>Walidacja po osobach</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="side-footer">Analiza offline · tylko kanały czołowe</div>', unsafe_allow_html=True)


if not weights:
    connection_label = "Brak pliku wag"
elif not missing_weight_features:
    connection_label = "Wagi i cechy dostępne"
else:
    connection_label = "Wagi wczytane · brak zgodnych cech"
st.markdown(
    f'<div class="topline"><div><div class="eyebrow">Panel analizy · BreathBack</div><h1>Koniec wstrzymania oddechu</h1><div class="subtitle">Widok czyta tabele z repozytorium i wyniki wytworzone przez backend.</div></div><div class="status-pill"><span class="status-dot"></span>{connection_label}</div></div>',
    unsafe_allow_html=True,
)

baseline_mae = None
baseline_median = None
if not baseline_evaluation.empty and "Błąd bezwzględny [s]" in baseline_evaluation:
    abs_errors = pd.to_numeric(baseline_evaluation["Błąd bezwzględny [s]"], errors="coerce").dropna()
    if not abs_errors.empty:
        baseline_mae = float(abs_errors.mean())
        baseline_median = float(abs_errors.median())

labelled_events = events[events["Koniec [s]"].notna()] if not events.empty else events
subject_count = int(events["ID osoby"].nunique()) if not events.empty else 0
st.markdown(
    '<div class="summary-row">'
    f'<div class="summary-card"><div class="summary-label">Zbiór aktywny</div><div class="summary-value">BH <span class="summary-note">high freq · supine</span></div></div>'
    f'<div class="summary-card"><div class="summary-label">Osoby w tabeli</div><div class="summary-value">{subject_count or "—"}</div></div>'
    f'<div class="summary-card"><div class="summary-label">Oznaczone próby</div><div class="summary-value">{len(labelled_events) if not events.empty else "—"} <span class="summary-note">marker końca dostępny</span></div></div>'
    f'<div class="summary-card"><div class="summary-label">Punkt odniesienia · MAE</div><div class="summary-value">{f"{baseline_mae:.2f} s" if baseline_mae is not None else "—"}<span class="summary-note">mediana {f"{baseline_median:.2f} s" if baseline_median is not None else "—"}</span></div></div>'
    '</div>',
    unsafe_allow_html=True,
)

if weights:
    st.info(f"Wczytano `breathback/weights/wagi-dziala.json` ({len(expected_features)} cech). Tabela cech dostępna w repozytorium nie zawiera {len(missing_weight_features)} wymaganych kolumn. Model opisano jako: {weight_region}.")

preview_tab, evaluation_tab = st.tabs(["Podgląd próby", "Ocena metody"])

with preview_tab:
    if events.empty:
        st.markdown('<div class="empty-state"><div class="empty-icon">〰</div><div class="empty-title">Brak tabeli zdarzeń</div><div class="empty-copy">Nie znaleziono wyników przygotowania danych. Sprawdź pliki w katalogu derived.</div></div>', unsafe_allow_html=True)
    else:
        subject_ids = sorted(events["ID osoby"].dropna().astype(int).unique().tolist())
        controls = st.columns([1, 1, 1.5, 1.5])
        with controls[0]:
            subject = st.selectbox("Osoba", subject_ids, index=0, format_func=lambda value: f"ID {value}")
        subject_events = events[events["ID osoby"] == subject]
        trials = sorted(subject_events["Numer próby"].dropna().astype(int).unique().tolist())
        with controls[1]:
            trial = st.selectbox("Próba", trials, index=trials.index(3) if 3 in trials else 0)
        selected_event = subject_events[subject_events["Numer próby"] == trial].iloc[0]
        onset = as_float(selected_event.get("Początek [s]")) or 0.0
        reference_end = as_float(selected_event.get("Koniec [s]"))
        predicted_end, prediction_reference, prediction_method = prediction_values(subject, trial)
        if reference_end is None:
            reference_end = prediction_reference
        with controls[2]:
            variants = ["short", "long", "both"]
            variant = st.selectbox("Kanały cech", variants, index=2, format_func={"short": "Krótkie", "long": "Długie", "both": "Wszystkie"}.get)
        with controls[3]:
            if st.button("Podgląd okien treningowych", use_container_width=True, disabled=training_preview.empty):
                show_training_preview()

        plot_col, result_col = st.columns([1.55, 1], gap="large")
        with plot_col:
            st.markdown('<div class="panel-title">Sygnał i przewidywanie</div>', unsafe_allow_html=True)
            feature_names = {
                "slow_slope_median_od_per_s": "Nachylenie wolnego sygnału",
                "slope_change_median_od_per_s": "Zmiana nachylenia",
                "slow_local_sd_median_od": "Zmienność lokalna",
                "pulse_rms_median_od": "Amplituda pulsacji",
                "channel_change_agreement_median_corr": "Zgodność kanałów",
            }
            available_frame = load_csv(DERIVED_DIR / ("BH_detector_presentation.csv" if trial == 3 else "BH_detector_test_LOSO.csv"))
            feature_options = [key for key in feature_names if key in available_frame.columns]
            has_signal = False
            if feature_options:
                feature = st.selectbox("Cecha w oknie", feature_options, format_func=feature_names.get)
                has_signal = plot_feature_windows(subject, trial, variant, onset, reference_end, predicted_end, feature)
            if not has_signal:
                st.markdown('<div class="empty-state"><div class="empty-icon">〰</div><div class="empty-title">Brak cech dla tej próby</div><div class="empty-copy">Potrzebny jest plik z cechami wejściowymi zgodnymi z modelem.</div></div>', unsafe_allow_html=True)
        with result_col:
            st.markdown('<div class="panel-title">Wynik dla wybranej próby</div>', unsafe_allow_html=True)
            st.caption(f"{prediction_method} · początek z markera 2")
            if weights and missing_weight_features and "Punkt odniesienia" in prediction_method:
                st.warning("To przewidywanie pochodzi z metody odniesienia. Nie uruchomiłem wag MLP, bo dostępne okna nie mają wymaganych cech wejściowych.")
            st.metric("Koniec przewidziany", format_seconds(predicted_end))
            if reference_end is not None:
                st.metric("Koniec referencyjny", format_seconds(reference_end))
                signed_error = predicted_end - reference_end if predicted_end is not None else None
                st.metric("Błąd ze znakiem", format_seconds(signed_error))
            else:
                st.metric("Błąd względem znacznika", "Nie dotyczy · brak etykiety")
            if trial == 3:
                st.caption("Próba 3 nie ma znacznika końca; jej przewidywanie nie ma błędu referencyjnego.")
            else:
                st.caption("Błąd ze znakiem = przewidywany koniec minus koniec ze znacznika.")

        st.divider()
        st.markdown('<div class="panel-title">Tabela zdarzeń</div><div class="panel-caption">Dane wczytane z przygotowanego zbioru; koniec pochodzi z kolejnego znacznika 1.</div>', unsafe_allow_html=True)
        st.dataframe(events[["ID osoby", "Numer próby", "Początek [s]", "Koniec [s]", "Czas trwania [s]"]], use_container_width=True, hide_index=True)
        st.caption(f"Źródło: `{events_source}` · marker 2 oznacza początek; zapisane duration markera nie jest używane.")

with evaluation_tab:
    st.markdown('<div class="panel-title">Walidacja po osobach</div><div class="panel-caption">Każda osoba jest testowana na modelu uczonym na pozostałych. Jej próby nie trafiają do treningu tego foldu.</div>', unsafe_allow_html=True)
    if weights:
        weight_results = weights.get("evaluation_lopo", {})
        st.markdown("**Wyniki zapisane razem z wagami MLP**")
        if weight_results:
            result_cols = st.columns(5)
            result_cols[0].metric("MAE", format_seconds(weight_results.get("mean_absolute_error_s")))
            result_cols[1].metric("Mediana |błędu|", format_seconds(weight_results.get("median_absolute_error_s")))
            result_cols[2].metric("Wyniki ±5 s", f"{weight_results.get('within_5s_pct', float('nan')):.1f}%")
            result_cols[3].metric("Wyniki ±10 s", f"{weight_results.get('within_10s_pct', float('nan')):.1f}%")
            result_cols[4].metric("Pokrycie", f"{weight_results.get('coverage_pct', float('nan')):.1f}%")
            st.caption(f"Walidacja: {weight_results.get('labelled_trials', '—')} prób u {weight_results.get('participants', '—')} osób. Punkt odniesienia — mediana błędu: {format_seconds(weight_results.get('duration_baseline_median_absolute_error_s'))}.")
        if "outside the forehead-only ROI" in weight_region:
            st.warning("Te wagi opisano jako trenowane na wszystkich kanałach montażu, także poza czołem. Nie odpowiadają założeniu analizy wyłącznie kanałów czołowych.")
        if missing_weight_features:
            st.warning(f"Nie da się jeszcze wykonać predykcji tymi wagami: model oczekuje {len(expected_features)} cech, a dostępny plik okien ma {len(available_features.intersection(expected_features))} z nich. Brakuje m.in. {', '.join(missing_weight_features[:5])}.")
        with st.expander("Informacje o zapisanych wagach"):
            st.write(f"Typ modelu: `{weights.get('model_type', 'nieznany')}`")
            st.write(f"Obszar kanałów: `{weight_region}`")
            st.write(f"Liczba cech: {len(expected_features)}")
            st.write(f"Ustawienia: `{weights.get('best_params', {})}`")
            st.write("Cechy oczekiwane przez model:")
            st.code(", ".join(expected_features))

    if not backend_metrics.empty:
        scopes = backend_metrics["scope"].dropna().unique().tolist() if "scope" in backend_metrics else []
        selected_scope = st.selectbox("Zakres podsumowania", scopes, index=scopes.index("excluding_development") if "excluding_development" in scopes else 0) if scopes else None
        shown_metrics = backend_metrics[backend_metrics["scope"] == selected_scope] if selected_scope and "scope" in backend_metrics else backend_metrics
        st.dataframe(shown_metrics, use_container_width=True, hide_index=True)
        st.caption("Metryki modelu pochodzą z wyników backendu BreathBack.")
    elif baseline_mae is not None:
        st.info(f"Punkt odniesienia z przygotowanej oceny: MAE {baseline_mae:.2f} s, mediana błędu bezwzględnego {baseline_median:.2f} s.")
        comparison_rows = [{"Metoda": "Typowy czas wstrzymania", "MAE [s]": baseline_mae, "Mediana |błędu| [s]": baseline_median}]
        if weights:
            results = weights.get("evaluation_lopo", {})
            if results:
                comparison_rows.append({
                    "Metoda": "MLP · raport z pliku wag",
                    "MAE [s]": results.get("mean_absolute_error_s"),
                    "Mediana |błędu| [s]": results.get("median_absolute_error_s"),
                })
        st.dataframe(pd.DataFrame(comparison_rows), use_container_width=True, hide_index=True)
    else:
        st.info("Nie znaleziono jeszcze wyników oceny. Najpierw przygotuj dane i uruchom pipeline.")

    with st.expander("Metoda odniesienia"):
        st.write("Przewidywany koniec = początek wstrzymania + mediana czasu trwania z osób treningowych. Ta mediana jest przeliczana wewnątrz każdego foldu.")
    with st.expander("Co porównujemy dalej?"):
        st.write("Kanały krótkie, długie i oba razem; warianty z kontrolą jakości; potem wpływ ruchu. Włosy i aktywność zostają analizą dodatkową.")

st.markdown('<div class="bottom-note">Wyniki markerów są referencją eksperymentalną, a nie niezależnym pomiarem pierwszego wdechu.</div>', unsafe_allow_html=True)
