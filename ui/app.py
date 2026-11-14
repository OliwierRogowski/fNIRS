from __future__ import annotations

import sys
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ui.data_access import (  # noqa: E402
    DATA_DIR,
    FEATURE_NAMES,
    RESULTS_DIR,
    WEIGHTS_PATH,
    PrototypeDataError,
    infer_from_weight_file,
    load_prototype_data,
    mean_slow_od_trace,
)


st.set_page_config(
    page_title="BreathBack · koniec wstrzymania",
    page_icon="〰",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root { --ink:#152832; --muted:#596970; --teal:#006b60; --line:#e4e9e7; --paper:#f5f8f6; }
    .stApp { background:var(--paper); color:var(--ink); }
    [data-testid="stHeader"] { background:transparent; }
    section[data-testid="stSidebar"] { background:#17262e; border-right:1px solid #263740; }
    section[data-testid="stSidebar"] * { color:#edf4f3; }
    section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] * { color:#a9b7ba; }
    .main .block-container { max-width:1480px; padding-top:1.7rem; padding-bottom:2rem; }
    h1,h2,h3 { color:var(--ink) !important; letter-spacing:-.4px; }
    .eyebrow { color:var(--teal); font-size:10px; font-weight:800; letter-spacing:1.5px; text-transform:uppercase; margin-bottom:7px; }
    .subtitle { color:var(--muted); font-size:14px; margin:0 0 18px; }
    .hero { display:flex; align-items:flex-start; justify-content:space-between; gap:20px; }
    .badge { color:#207c70; background:#eaf6f2; border:1px solid #d3e8e1; padding:8px 12px; border-radius:99px; font-size:11px; font-weight:700; white-space:nowrap; margin-top:9px; }
    .panel { background:#fff; border:1px solid var(--line); border-radius:13px; padding:16px 18px; }
    .panel-title { color:var(--ink); font-size:16px; font-weight:800; margin-bottom:2px; }
    .panel-caption { color:#596970; font-size:11px; margin-bottom:10px; }
    .small-note { color:#596970; font-size:11px; line-height:1.5; }
    .footer-note { color:#596970; text-align:center; font-size:10px; margin:22px 0 4px; }
    div[data-testid="stMetric"] { background:#fff; border:1px solid var(--line); padding:13px 15px; border-radius:11px; }
    div[data-testid="stMetricLabel"] { color:#596970; }
    div[data-testid="stButton"] button { border-radius:9px; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False, max_entries=28)
def cached_trace(filename: str) -> pd.DataFrame:
    return mean_slow_od_trace(filename)


def fmt_seconds(value: object, digits: int = 2) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    return "—" if not np.isfinite(number) else f"{number:.{digits}f} s"


def event_plot(trace: pd.DataFrame, onset_s: float, predicted_s: float,
               reference_s: float | None) -> alt.TopLevelMixin:
    marker_times = [predicted_s]
    if reference_s is not None and np.isfinite(reference_s):
        marker_times.append(reference_s)
    plot_right = min(float(trace["time_s"].max()), max(marker_times)+15.0)
    plot_left = max(float(trace["time_s"].min()), onset_s-10.0)
    plot = trace[trace["time_s"].between(plot_left, plot_right)].copy()
    plot["elapsed_s"] = plot["time_s"]-onset_s
    plot = plot.dropna(subset=["slow_od_mean"])
    if len(plot) > 2200:
        plot = plot.iloc[::max(1, len(plot)//2200)]

    markers = [dict(elapsed_s=0.0, marker="Początek · marker 2")]
    markers.append(dict(elapsed_s=predicted_s-onset_s, marker="Przewidywany koniec"))
    if reference_s is not None and np.isfinite(reference_s):
        markers.append(dict(elapsed_s=reference_s-onset_s, marker="Koniec · marker 1"))

    color_scale = alt.Scale(
        domain=["Początek · marker 2", "Przewidywany koniec", "Koniec · marker 1"],
        range=["#946200", "#245a9b", "#b54740"],
    )
    signal = alt.Chart(plot).mark_line(color="#006b60", strokeWidth=1.7).encode(
        x=alt.X("elapsed_s:Q", title="Czas od początku wstrzymania [s]"),
        y=alt.Y("slow_od_mean:Q", title="Średnia wolna OD [j.u.]"),
        tooltip=[
            alt.Tooltip("elapsed_s:Q", title="Czas [s]", format=".1f"),
            alt.Tooltip("slow_od_mean:Q", title="Średnia OD", format=".4f"),
            alt.Tooltip("pairs_available:Q", title="Dostępne pary"),
        ],
    )
    rules = alt.Chart(pd.DataFrame(markers)).mark_rule(strokeWidth=2, strokeDash=[5, 4]).encode(
        x=alt.X("elapsed_s:Q"),
        color=alt.Color("marker:N", scale=color_scale, title="Znaczniki"),
        tooltip=[alt.Tooltip("marker:N", title=""), alt.Tooltip("elapsed_s:Q", title="Czas [s]", format=".2f")],
    )
    return (signal+rules).properties(height=350).interactive()


def score_plot(scores: pd.DataFrame, predicted_s: float, onset_s: float) -> alt.TopLevelMixin:
    frame = scores.copy()
    line = alt.Chart(frame).mark_line(color="#4875aa", strokeWidth=2).encode(
        x=alt.X("elapsed_s:Q", title="Czas od początku wstrzymania [s]"),
        y=alt.Y("smoothed_score:Q", title="Wynik modelu (nieskalibrowany)"),
        tooltip=[
            alt.Tooltip("elapsed_s:Q", title="Czas [s]", format=".1f"),
            alt.Tooltip("smoothed_score:Q", title="Wynik", format=".3f"),
        ],
    )
    marker = alt.Chart(pd.DataFrame({"elapsed_s": [predicted_s-onset_s]})).mark_rule(
        color="#078f80", strokeDash=[5, 4], strokeWidth=2
    ).encode(x="elapsed_s:Q")
    return (line+marker).properties(height=200).interactive()


try:
    data = load_prototype_data()
except PrototypeDataError as exc:
    st.error(str(exc))
    st.stop()

events: pd.DataFrame = data["events"]
features: pd.DataFrame = data["features"]
predictions: pd.DataFrame = data["predictions"]
metrics: pd.DataFrame = data["metrics"]
weights: dict = data["weights"]

people = sorted(events["subject_id"].unique().tolist())
labelled_count = int(events["reference_end_s"].notna().sum())
demo_count = int(events["reference_end_s"].isna().sum())
recording_count = int(events.groupby("subject_id")["file"].nunique().sum())

with st.sidebar:
    st.markdown("### 〰 BreathBack")
    st.caption("Prototyp offline · dane BH")
    st.divider()
    st.markdown("**Baza uczestników**")
    selected_subject = st.selectbox("ID uczestnika", people, format_func=lambda x: f"Uczestnik {x}")
    subject_events = events[events["subject_id"] == selected_subject].sort_values("trial")
    available_trials = subject_events["trial"].astype(int).tolist()
    default_trial = 3 if 3 in available_trials else available_trials[0]
    selected_trial = st.selectbox(
        "Próba",
        available_trials,
        index=available_trials.index(default_trial),
        format_func=lambda x: f"Próba {x} · {'LOPO/test' if x in (1, 2) else 'demo bez etykiety'}",
    )
    st.divider()
    st.markdown("**Podłączone artefakty**")
    st.caption(f"Wagi: `{WEIGHTS_PATH.name}`")
    st.caption(f"Dane przygotowane: `{RESULTS_DIR.relative_to(ROOT)}`")
    st.caption(f"Nagrania lokalne: {len(list(DATA_DIR.glob('*.snirf')))}")
    if st.button("Odśwież dane", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

st.markdown(
    '<div class="hero"><div><div class="eyebrow">Panel prototypu · zbiór supine BH</div>'
    '<h1>Koniec wstrzymania oddechu</h1>'
    '<div class="subtitle">Próby 1–2 pokazują ocenę LOPO na osobach pominiętych w treningu. Próba 3 jest demonstracją bez oznaczonego końca; przewidują ją końcowe wagi <code>wagi-dziala.json</code>, wytrenowane na 56 oznaczonych próbach.</div></div>'
    '<div class="badge">LOPO · bez mieszania osób</div></div>',
    unsafe_allow_html=True,
)

summary = st.columns(4)
summary[0].metric("Uczestnicy", len(people))
summary[1].metric("Nagrania", recording_count)
summary[2].metric("Oznaczone · ocena", labelled_count)
summary[3].metric("Demo · bez końca", demo_count)

preview_tab, evaluation_tab, database_tab = st.tabs(
    ["Podgląd próby", "Ocena modelu", "Baza uczestników"]
)

selected_event = subject_events[subject_events["trial"] == selected_trial].iloc[0]
onset_s = float(selected_event["onset_s"])
reference_value = selected_event["reference_end_s"]
reference_s = float(reference_value) if pd.notna(reference_value) else None
selected_features = features[
    (features["subject_id"] == selected_subject) & (features["trial"] == selected_trial)
].copy()
is_demo = selected_trial == 3 or reference_s is None

if is_demo:
    result = infer_from_weight_file(selected_features, weights)
    prediction_s = float(result["prediction_s"]) if result["status"] == "estimated" else np.nan
    prediction_row = None
else:
    matches = predictions[
        (predictions["subject_id"] == selected_subject)
        & (predictions["trial"] == selected_trial)
        & (predictions["method"] == "mlp_optuna")
    ]
    if matches.empty:
        prediction_row = None
        result = dict(status="missing_lopo_prediction", score=np.nan, pairs_used=0,
                      available_candidates=0, scores=pd.DataFrame())
        prediction_s = np.nan
    else:
        prediction_row = matches.iloc[0]
        prediction_s = float(prediction_row["predicted_end_s"])
        result = dict(
            status=str(prediction_row["status"]),
            score=float(prediction_row["score"]) if pd.notna(prediction_row["score"]) else np.nan,
            pairs_used=int(prediction_row["pairs_at_prediction"]),
            available_candidates=int(prediction_row["available_candidates"]),
            scores=pd.DataFrame(),
        )

with preview_tab:
    if not np.isfinite(prediction_s):
        st.warning("Model nie znalazł poprawnego okna dla tej próby.")
    chart_column, result_column = st.columns([1.65, 1], gap="large")
    with chart_column:
        st.markdown('<div class="panel-title">Sygnał fNIRS</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="panel-caption">Wykres pokazuje średni wolny przebieg OD jako kontekst. Detektor wyznacza koniec z 14 cech liczonych w przesuwających się oknach na podstawie wielu par kanałów. Linie oznaczają początek, przewidywany koniec i — dla prób 1–2 — marker referencyjny.</div>',
            unsafe_allow_html=True,
        )
        try:
            with st.spinner("Wczytuję zapis SNIRF uczestnika…"):
                trace = cached_trace(str(selected_event["file"]))
            if np.isfinite(prediction_s):
                st.altair_chart(
                    event_plot(trace, onset_s, prediction_s, reference_s),
                    use_container_width=True,
                )
            else:
                st.info("Brak predykcji do naniesienia na wykres.")
        except (PrototypeDataError, ImportError, OSError, ValueError) as exc:
            st.error(f"Nie udało się wczytać sygnału: {exc}")

        st.markdown('<div class="panel-title">Wybrane zdarzenie</div>', unsafe_allow_html=True)
        selected_event_table = subject_events[subject_events["trial"] == selected_trial][[
            "trial", "onset_s", "reference_end_s", "duration_s", "reference"
        ]].rename(columns={
            "trial": "Próba", "onset_s": "Początek [s]",
            "reference_end_s": "Koniec referencyjny [s]",
            "duration_s": "Czas trwania [s]", "reference": "Źródło końca",
        }).copy()
        st.dataframe(
            selected_event_table.style.format({
                "Początek [s]": "{:.3f}",
                "Koniec referencyjny [s]": lambda x: "—" if pd.isna(x) else f"{x:.3f}",
                "Czas trwania [s]": lambda x: "—" if pd.isna(x) else f"{x:.3f}",
            }),
            use_container_width=True, hide_index=True,
        )

    with result_column:
        st.markdown('<div class="panel-title">Wynik detektora</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="panel-caption">' +
            ("Próba demo · prognoza z wag JSON" if is_demo else "Próba oznaczona · predykcja LOPO na pominiętej osobie") +
            '</div>',
            unsafe_allow_html=True,
        )
        st.metric("Przewidywany koniec", fmt_seconds(prediction_s))
        if reference_s is None:
            st.metric("Błąd względem markera", "Brak referencji")
        elif np.isfinite(prediction_s):
            signed_error = prediction_s-reference_s
            st.metric("Błąd względem markera", f"{signed_error:+.2f} s", delta=f"|błąd| {abs(signed_error):.2f} s", delta_color="off")
        else:
            st.metric("Błąd względem markera", "Brak predykcji")
        st.metric(
            "Pary z poprawnymi cechami",
            f"{result['pairs_used']} / {int(selected_features['pairs_total'].max()) if len(selected_features) else 0}",
            help="To wskaźnik dostępności kanałów w wybranym oknie, nie bezpośredni pomiar jakości kontaktu optod ze skórą.",
        )
        st.metric("Kandydaci w obszarze poszukiwań", int(result["available_candidates"]))
        if is_demo and result["status"] == "estimated":
            st.caption(f"Maksymalny wynik modelu: {float(result['score']):.3f}. Nie jest skalibrowanym prawdopodobieństwem poprawności.")
            st.altair_chart(score_plot(result["scores"], prediction_s, onset_s), use_container_width=True)
        elif not is_demo:
            st.caption("Wynik dla próby 1–2 pochodzi z folda, w którym uczestnik nie był w treningu.")
        st.info("Referencją jest marker eksperymentalny, nie niezależny zapis pierwszego wdechu. Prototyp działa offline i nie służy do zastosowań klinicznych.")

with evaluation_tab:
    metric_map = {str(row["method"]): row for _, row in metrics.iterrows()}
    mlp_metric = metric_map.get("mlp_optuna")
    baseline_metric = metric_map.get("duration_baseline")
    if mlp_metric is not None:
        cols = st.columns(4)
        cols[0].metric("Mediana |błędu|", fmt_seconds(mlp_metric["median_abs_error_s"]))
        cols[1].metric("MAE", fmt_seconds(mlp_metric["mean_abs_error_s"]))
        cols[2].metric("W ±5 s", f"{float(mlp_metric['within_5s_pct']):.1f}%")
        cols[3].metric("W ±10 s", f"{float(mlp_metric['within_10s_pct']):.1f}%")

    display_rows = []
    display_names = {
        "duration_baseline": "Typowy czas trwania",
        "logistic_optuna": "Regresja logistyczna · Optuna",
        "mlp_optuna": "MLP · wagi-dziala",
    }
    for method, row in metric_map.items():
        display_rows.append({
            "Metoda": display_names.get(method, method),
            "Próby": int(row["n_trials"]),
            "Mediana |błędu| [s]": float(row["median_abs_error_s"]),
            "MAE [s]": float(row["mean_abs_error_s"]),
            "W ±5 s": f"{float(row['within_5s_pct']):.1f}%",
            "W ±10 s": f"{float(row['within_10s_pct']):.1f}%",
            "Pokrycie": f"{float(row['coverage_pct']):.0f}%",
            "Bez predykcji": int(row["n_no_prediction"]),
        })
    st.dataframe(pd.DataFrame(display_rows), use_container_width=True, hide_index=True)
    st.caption("Ocena obejmuje 56 oznaczonych wstrzymań u 28 osób. Predykcje dla każdej osoby pochodzą z modelu trenowanego na pozostałych osobach.")
    st.caption("To regresja czasu zakończenia, dlatego raportujemy błędy w sekundach i odsetki w tolerancji ±5/±10 s, a nie klasyczne accuracy.")
    if mlp_metric is not None and baseline_metric is not None:
        delta = float(mlp_metric["within_10s_pct"])-float(baseline_metric["within_10s_pct"])
        st.info(f"W ±10 s MLP uzyskał {delta:+.1f} p.p. względem metody typowego czasu trwania.")
    st.download_button(
        "Pobierz predykcje LOPO (CSV)",
        data=predictions.to_csv(index=False).encode("utf-8-sig"),
        file_name="breathback_lopo_predictions.csv",
        mime="text/csv",
    )

with database_tab:
    st.markdown("### Lokalna baza uczestników")
    st.caption("Indeks powstał z tabeli zdarzeń i cech przygotowanych z 28 plików BH. Interfejs pokazuje identyfikatory, a nie dane osobowe.")
    valid_feature_counts = (
        features.groupby(["subject_id", "trial"])[FEATURE_NAMES]
        .apply(lambda frame: int(np.isfinite(frame.to_numpy(dtype=float)).all(axis=1).sum()))
        .rename("Okna z cechami")
    )
    catalogue = events.groupby("subject_id").agg(
        Nagrania=("file", "nunique"),
        Próby=("trial", "count"),
        Oznaczone=("reference_end_s", "count"),
    )
    demo_counts = events.assign(demo=events["reference_end_s"].isna()).groupby("subject_id")["demo"].sum()
    catalogue["Demo bez etykiety"] = demo_counts
    catalogue.index.name = "ID uczestnika"
    st.dataframe(catalogue.reset_index(), use_container_width=True, hide_index=True)

    subject_catalogue = events[events["subject_id"] == selected_subject].merge(
        valid_feature_counts.reset_index(), on=["subject_id", "trial"], how="left"
    )
    subject_catalogue = subject_catalogue[[
        "subject_id", "trial", "onset_s", "reference_end_s", "duration_s", "Okna z cechami"
    ]].rename(columns={
        "subject_id": "ID uczestnika", "trial": "Próba", "onset_s": "Początek [s]",
        "reference_end_s": "Koniec marker 1 [s]", "duration_s": "Czas [s]",
    })
    st.markdown(f"**Szczegóły uczestnika {selected_subject}**")
    st.dataframe(subject_catalogue, use_container_width=True, hide_index=True)
    st.download_button(
        "Pobierz tabelę zdarzeń (CSV)",
        data=events.to_csv(index=False).encode("utf-8-sig"),
        file_name="breathback_event_database.csv",
        mime="text/csv",
    )

st.markdown(
    '<div class="footer-note">Prototyp demonstracyjny na gotowych nagraniach BH. Wagi końcowe służą do prób demo; metryki prób oznaczonych pochodzą wyłącznie z LOPO.</div>',
    unsafe_allow_html=True,
)
