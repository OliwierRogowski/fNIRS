"""Figures and a Polish report; presentation never changes detector parameters."""
import csv
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_csv(path):
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def fmt(value, digits=2):
    v = number(value)
    return f"{v:.{digits}f}" if np.isfinite(v) else "—"


def normalized(x, baseline):
    center = np.nanmedian(x[baseline], axis=0)
    scale = np.maximum(np.nanstd(x[baseline], axis=0), 1e-6)
    return (x-center)/scale


def markers(ax, onset, end=None):
    ax.axvline(0, color="black", lw=1, ls=":", label="Początek: marker 2")
    if end is not None and np.isfinite(end):
        ax.axvline(end-onset, color="#2c8c58", lw=1.6, label="Koniec referencyjny: marker 1")
    ax.grid(alpha=.15)


def plot_development(example_records, examples, out):
    """Three inspected examples; these subjects are excluded from the main summary."""
    fig, axes = plt.subplots(len(examples), 3, figsize=(15, 3.4*len(examples)), squeeze=False,
                             layout="constrained")
    labels = ["Krótkie", "Średnie", "Długie"]
    for row, event in enumerate(examples):
        rec = example_records[event["subject_id"]]
        onset, end = event["onset_s"], event["end_s"]
        window = (rec.time >= onset-15) & (rec.time <= min(onset+120, rec.time[-1]))
        baseline = (rec.time >= onset-30) & (rec.time < onset-5)
        time = rec.time[window]-onset
        for kind, color, label in [("short", "#2b75a9", "Krótkie kanały"),
                                    ("long", "#d17b25", "Długie kanały")]:
            cols = [p[k] for p in rec.pairs if p["kind"] == kind for k in ["a", "b"]]
            if cols:
                slow = normalized(rec.slow[:, cols], baseline)
                pulse = normalized(rec.pulse[:, cols], baseline)
                axes[row, 0].plot(time, np.median(slow[window], axis=1), color=color, label=label)
                axes[row, 1].plot(time, np.median(pulse[window], axis=1), color=color, label=label)
        motion = normalized(rec.motion, baseline)
        axes[row, 2].plot(time, np.sqrt(np.mean(motion[window]**2, axis=1)), color="#70539e",
                          label="Ruch: RMS po standaryzacji 6 osi")
        for column, title in enumerate(["Wolny przebieg OD", "Obwiednia pulsacji", "Ruch (nie jest wejściem modelu)"]):
            ax = axes[row, column]
            markers(ax, onset, end)
            ax.set_title(f"{labels[row]}: ID {rec.subject}, próba {event['trial']} — {title}", fontsize=10)
            ax.set_xlabel("Czas od początku bezdechu [s]")
            ax.set_ylabel("Jednostki względem okresu przed próbą")
            ax.legend(fontsize=7, loc="upper left")
    fig.suptitle("Tylko czoło — przegląd rozwojowy: przykłady według czasu trwania\n"
                 "Linie są opisem sygnału; nie stanowią niezależnego pomiaru oddechu", fontsize=13)
    fig.savefig(out/"development_examples.png", dpi=140)
    plt.close(fig)


def plot_metrics(rows, out):
    selected = [r for r in rows if r["scope"] == "excluding_development"]
    if not selected:
        selected = rows
    baseline = next((r for r in selected if r["method"] == "duration_baseline"), None)
    variants = ["short_all", "short_qc", "long_all", "long_qc", "both_all", "both_qc"]
    labels = ["Krótkie", "Krótkie + SCI", "Długie", "Długie + SCI", "Oba", "Oba + SCI"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.8), layout="constrained")
    for method, offset, color, label in [("rule", -.18, "#d17b25", "Reguła"),
                                        ("logistic", .18, "#2b75a9", "Regresja logistyczna")]:
        group = {r["variant"]: r for r in selected if r["method"] == method}
        for ax, metric in zip(axes, ["median_abs_error_s", "within_10s_pct", "coverage_pct"]):
            values = [number(group[v][metric]) if v in group else np.nan for v in variants]
            ax.bar(np.arange(6)+offset, values, .34, color=color, label=label)
            if baseline and method == "rule":
                ax.axhline(number(baseline[metric]), color="#333333", ls="--", label="Typowy czas bezdechu")
            ax.set_xticks(np.arange(6), labels, rotation=45, ha="right")
            ax.grid(axis="y", alpha=.15)
    for ax, title in zip(axes, ["Mediana błędu bezwzględnego [s] ↓",
                                "Wyniki w granicach ±10 s [%] ↑", "Próby ze wskazaniem [%]"]):
        ax.set_title(title, fontsize=11)
    axes[1].set_ylim(0, 105); axes[2].set_ylim(0, 105)
    axes[0].legend(fontsize=8)
    fig.suptitle("Tylko czoło — LOPO: uczestnicy poza przeglądem rozwojowym\n"
                 "Odsetek ±10 s obejmuje wszystkie próby, również bez wskazania")
    fig.savefig(out/"evaluation_summary.png", dpi=150)
    plt.close(fig)


def plot_subjects(out, trials, predictions, demos):
    curves_path = out/"score_curves.joblib"
    curves = joblib.load(curves_path) if curves_path.exists() else {}
    destination = out/"subject_plots"
    destination.mkdir(exist_ok=True)
    all_predictions = predictions+demos
    for sid in sorted({t.subject for t in trials}):
        path = out/"signals"/f"subject_{sid:02d}.npz"
        if not path.exists():
            continue
        with np.load(path) as rec:
            time, slow = rec["time"], rec["slow"]
            subject_trials = sorted([t for t in trials if t.subject == sid], key=lambda t: t.number)
            fig, axes = plt.subplots(len(subject_trials), 2, figsize=(13, 3.1*len(subject_trials)),
                                     squeeze=False, layout="constrained")
            for row, trial in enumerate(subject_trials):
                ax, score_ax = axes[row]
                baseline = (time >= trial.onset-30) & (time < trial.onset-5)
                window = (time >= trial.onset-12) & (time <= trial.onset+120)
                trace = np.median(normalized(slow, baseline), axis=1)
                ax.plot(time[window]-trial.onset, trace[window], color="#777777", label="Mediana wolnego OD (wybrane kanały czoła)")
                for method, variant, color, label in [
                    ("duration_baseline", "none", "#a56a24", "Typowy czas"),
                    ("logistic", "both_all", "#2b75a9", "Logistyczny: oba"),
                    ("logistic", "both_qc", "#9a4b9f", "Logistyczny: oba + SCI")]:
                    matches = [p for p in all_predictions if int(p["subject_id"]) == sid and int(p["trial"]) == trial.number
                               and p["method"] == method and p["variant"] == variant]
                    if matches:
                        estimate = number(matches[0]["predicted_end_s"])
                        if np.isfinite(estimate):
                            for panel in [ax, score_ax]:
                                panel.axvline(estimate-trial.onset, color=color, ls="--", lw=1.2, label=label)
                for variant, color in [("both_all", "#2b75a9"), ("both_qc", "#9a4b9f")]:
                    curve = curves.get(f"{sid}_{trial.number}_{variant}")
                    if curve is not None:
                        score_ax.plot(curve["times"]-trial.onset, curve["scores"], color=color, alpha=.8)
                for panel in [ax, score_ax]:
                    markers(panel, trial.onset, trial.end)
                    panel.set_xlabel("Czas od początku bezdechu [s]")
                    panel.legend(fontsize=6, loc="upper right")
                ax.set_ylabel("OD względem okresu przed próbą")
                score_ax.set_ylabel("Wygładzony wynik klasyfikatora\n(niepewność nie jest skalibrowana)")
                ax.set_title(f"Próba {trial.number}: " + ("referencja dostępna" if np.isfinite(trial.end) else "brak referencji — demonstracja"), fontsize=10)
                score_ax.set_title("Krzywa detektora; pionowe linie po korekcie opóźnienia", fontsize=10)
            fig.suptitle(f"Tylko czoło — uczestnik {sid}: modele uczone bez tej osoby\n"
                         "Analiza offline; marker końca jest referencją eksperymentalną", fontsize=12)
            fig.savefig(destination/f"subject_{sid:02d}.png", dpi=130)
            plt.close(fig)


def metric_table(rows):
    text = ["| Metoda | Kanały | Próby | Mediana | Błąd ze znakiem | Średni błąd abs. | ±5 s | ±10 s | Pokrycie | Bez wskazania | Błąd >10 s |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    methods = {"duration_baseline": "Typowy czas", "rule": "Reguła", "logistic": "Logistyczna"}
    for r in rows:
        text.append(f"| {methods.get(r['method'], r['method'])} | {r['variant']} | {r['n_trials']} | "
                    f"{fmt(r['median_abs_error_s'])} s | {fmt(r['median_signed_error_s'])} s | "
                    f"{fmt(r['mean_abs_error_s'])} s | {fmt(r['within_5s_pct'], 1)}% | "
                    f"{fmt(r['within_10s_pct'], 1)}% | {fmt(r['coverage_pct'], 1)}% | "
                    f"{r['n_no_prediction']} | {r['n_outside_10s']} |")
    return text


def make_report(out, trials, manifest):
    out = Path(out)
    metrics = read_csv(out/"metrics.csv")
    predictions = read_csv(out/"predictions_lopo.csv")
    demos = read_csv(out/"predictions_unmarked.csv")
    comparisons = read_csv(out/"paired_comparisons.csv")
    quality_comparisons = read_csv(out/"quality_comparisons.csv")
    associations = read_csv(out/"exploratory_associations.csv")
    if not metrics:
        raise ValueError("No metrics.csv found: run the evaluate stage before report")
    plot_metrics(metrics, out)
    plot_subjects(out, trials, predictions, demos)
    primary = [r for r in metrics if r["scope"] == "excluding_development"]
    baseline = next(r for r in primary if r["method"] == "duration_baseline")
    signal_rows = [r for r in primary if r["method"] != "duration_baseline"]
    descriptive_best = min(signal_rows, key=lambda r: number(r["mean_abs_error_s"]))
    delta = number(descriptive_best["mean_abs_error_s"])-number(baseline["mean_abs_error_s"])
    all_rows = [r for r in metrics if r["scope"] == "all_lopo"]
    all_baseline = next(r for r in all_rows if r["method"] == "duration_baseline")
    all_best = min([r for r in all_rows if r["method"] != "duration_baseline"],
                   key=lambda r: number(r["median_abs_error_s"]))
    does_not_beat_baseline = all(number(r["median_abs_error_s"]) >= number(baseline["median_abs_error_s"])
                                 and number(r["mean_abs_error_s"]) >= number(baseline["mean_abs_error_s"])
                                 for r in signal_rows)
    text = ["# BreathBack — estymacja końca wstrzymania oddechu wyłącznie z czoła", "",
            "## Cel i wynik", "",
            "Na podstawie sygnału fNIRS szacujemy moment zakończenia wstrzymania oddechu. "
            "Początek pochodzi z markera 2. Jest to analiza gotowych nagrań, a nie detektor czasu rzeczywistego.", "",
            "Analizujemy wyłącznie pary, których **obie optody** spełniają ścisłą regułę nominalnego obszaru czoła. "
            "Nie wykorzystujemy całego montażu przedczołowego. Szczegóły i ograniczenia lokalizacji podano niżej.", "",
            f"Przetworzono {len(manifest['sources'])} nagrań; {manifest['labelled_trials']} prób ma referencyjny koniec, "
            f"a {manifest['unlabelled_trials']} nie ma oznaczonego końca. "
            f"Uczestnicy przeglądu rozwojowego: {', '.join(map(str, manifest['development_subjects']))}. "
            "Główne zestawienie wyłącza te osoby z oceny; dodatkowo podano LOPO dla całego zbioru.", "",
            ("**Wynik negatywny: żaden badany wariant sygnałowy nie pokonał przewidywania na podstawie typowego czasu bezdechu "
             "pod względem mediany i średniego błędu na osobach poza przeglądem rozwojowym.** "
             if does_not_beat_baseline else "Wyniki wariantów należy porównać z metodą odniesienia i ich pokryciem. ") +
            f"W całym zbiorze ({all_baseline['n_trials']} prób) mediana błędu odniesienia wynosi {fmt(all_baseline['median_abs_error_s'])} s, "
            f"a najniższa mediana wśród detektorów sygnałowych wynosi {fmt(all_best['median_abs_error_s'])} s "
            f"({all_best['method']} / {all_best['variant']}). "
            f"Po wyłączeniu osób rozwojowych ({baseline['n_trials']} prób) mediana odniesienia to {fmt(baseline['median_abs_error_s'])} s. "
            "Działający prototyp nie oznacza zatem potwierdzonej przewagi informacji z fNIRS w tym zadaniu.", "",
            f"Metoda odniesienia: mediana błędu {fmt(baseline['median_abs_error_s'])} s; "
            f"średni błąd bezwzględny {fmt(baseline['mean_abs_error_s'])} s; "
            f"trafienia ±10 s: {fmt(baseline['within_10s_pct'], 1)}%. "
            f"Opisowo najniższy średni błąd wśród wariantów sygnałowych uzyskał {descriptive_best['method']} / "
            f"{descriptive_best['variant']}: {fmt(descriptive_best['mean_abs_error_s'])} s "
            f"(różnica względem odniesienia {delta:+.2f} s). "
            "To porównanie po obejrzeniu wyników, nie niezależnie potwierdzony wybór najlepszego modelu.", "",
            "![Porównanie metod](evaluation_summary.png)", "",
            "## Ścisły wybór optod czoła", "",
            "Dla źródła i detektora wyznaczamy najbliższy punkt orientacyjny spośród landmarków zapisanych w SNIRF. "
            "Obie optody muszą być przypisane do grupy `Fp*`, `AFp*` lub `AF*` i oddalone od swojego najbliższego punktu o najwyżej 25 mm. "
            "Wykluczamy `AFF*`, pozostałe `F*` oraz `FC*`; grupa `Fp*` jest jawnie dozwolona. "
            "Nie wystarcza położenie tylko jednej optody lub środka kanału na czole. To z góry ustalony filtr geometryczny, nie parametr dobierany do błędu detektora.", "",
            "W tym montażu zachowane pary to **S5–D3, S5–D4, S6–D4, S6–D5, S7–D4 i S8–D5**: "
            "4 długie i 2 krótkie pary, łącznie 12 kolumn dla 760/850 nm. "
            "Odrzucone pary: S1–D1, S2–D1, S3–D2 i S4–D2. "
            "Audyt każdej pary i przyczyn odrzucenia: [montage_selection.csv](montage_selection.csv).", "",
            "Wybór opiera się na **nominalnych pozycjach montażu**, nie na indywidualnym pomiarze faktycznego ułożenia czepka. "
            "Nie potwierdza położenia optod na skórze konkretnej osoby ani braku włosów pod optodami. "
            "Wszystkie wyniki tej wersji wymagają ponownego przygotowania cech i ponownego LOPO po filtrze czoła; "
            "wyniki poprzedniego pełnego montażu nie są dowodem skuteczności tego podzbioru.", "",
            "## Ocena poza trzema osobami przeglądu rozwojowego", "",
            "Mediana i średni błąd bezwzględny są liczone wśród prób z predykcją; "
            "odsetki ±5/±10 s mają w mianowniku wszystkie próby. Pokrycie to odsetek prób, dla których metoda wskazała koniec. "
            "Błąd ze znakiem = predykcja − marker: dodatni oznacza za późno. Brak wskazania i wskazanie z błędem >10 s raportowane są osobno.", ""]
    text += metric_table(primary)
    text += ["", "## Cały zbiór — zewnętrzne LOPO", ""]
    text += metric_table([r for r in metrics if r["scope"] == "all_lopo"])
    text += ["", "## Porównania sparowane z metodą odniesienia", "",
             "Dla każdej osoby poza przeglądem rozwojowym uśredniono różnice bezwzględnych błędów na tych samych próbach. "
             "Brak predykcji otrzymuje karę 120 s także w metodzie odniesienia. Ujemna różnica oznacza niższy błąd detektora. "
             "Przedziały bootstrapowe (10 000 losowań osób) są opisem niepewności; nie uwzględniają wyboru spośród wielu wariantów i zależności między modelami LOPO.", "",
             "| Metoda | Wariant | Osoby | Różnica średniego błędu [s] | Przedział bootstrapowy 95% [s] |",
             "|---|---|---:|---:|---:|"]
    for r in comparisons:
        text.append(f"| {r['method']} | {r['variant']} | {r['n_subjects']} | {fmt(r['mean_error_difference_vs_baseline_s'])} | "
                    f"[{fmt(r['bootstrap_ci_low_s'])}, {fmt(r['bootstrap_ci_high_s'])}] |")
    text += ["", "## Wpływ kontroli jakości", "",
             "Wariant `_all` korzysta z kanałów spełniających warunek dostępności próbek. Wariant `_qc` dodatkowo "
             "odrzuca pary o niskim lokalnym SCI. Próg SCI (0,5 / 0,75 / 0,9) jest wybierany wyłącznie wewnątrz treningu. "
             "Porównując warianty należy sprawdzić jednocześnie błąd, pokrycie i liczbę niepoprawnych wskazań; "
             "mniejszy błąd wśród pozostałych prób nie wystarcza do uznania poprawy. "
             "Szczegóły kanałów: [channel_quality.csv](channel_quality.csv).", ""]
    if quality_comparisons:
        text += ["Poniższe porównanie QC z `_all` dotyczy tych samych osób i prób poza przeglądem rozwojowym. "
                 "Różnica to średnia po osobach różnic błędu QC − all; każda brakująca predykcja otrzymuje karę 120 s "
                 "w obu wariantach. Ujemny wynik sprzyja QC. Przedział 95% pochodzi z bootstrapu osób. "
                 "To porównania eksploracyjne, bez korekty na wiele wariantów; nie są wnioskami klinicznymi.", "",
                 "| Metoda | Kanały | Próby | Mediana all / QC [s] | Pokrycie all / QC | Różnica QC − all [s] | Przedział 95% [s] |",
                 "|---|---|---:|---:|---:|---:|---:|"]
        for r in quality_comparisons:
            text.append(f"| {r['method']} | {r['kind']} | {r['n_trials']} | "
                        f"{fmt(r['median_abs_error_all_s'])} / {fmt(r['median_abs_error_qc_s'])} | "
                        f"{fmt(r['coverage_all_pct'], 1)}% / {fmt(r['coverage_qc_pct'], 1)}% | "
                        f"{fmt(r['mean_penalized_error_difference_qc_minus_all_s'])} | "
                        f"[{fmt(r['bootstrap_ci_low_s'])}, {fmt(r['bootstrap_ci_high_s'])}] |")
        text += ["", "Dane: [quality_comparisons.csv](quality_comparisons.csv).", ""]
    text += [
             "## Przegląd sygnałów i demonstracje", "",
             "![Krótkie, średnie i długie wstrzymanie](development_examples.png)", "",
             "Wykresy wszystkich uczestników zestawiają sygnał, referencję dla dwóch pierwszych prób i predykcje trzeciej próby. "
             "Trzecie próby nie mają referencyjnego końca: nie przypisujemy im błędu ani trafności. "
             "Ich predykcje pochodzą z modeli zewnętrznych LOPO, bez uczenia na danej osobie.", ""]
    for sid in sorted({t.subject for t in trials}):
        text.append(f"- [Uczestnik {sid}](subject_plots/subject_{sid:02d}.png)")
    text += ["", "## Rozszerzenia: włosy, aktywność i wiek", "",
             "To eksploracyjna analiza związku metadanych z **błędem detektora** (logistyczny, oba rodzaje kanałów + SCI), "
             "nie pomiar wydolności ani test wpływu włosów na oddychanie. Jednostką jest uczestnik, wynik to mediana "
             "jego błędów LOPO. Poniższe testy nie korygują zależności od leków, wieku/płci czy jakości kontaktu i nie ustalają przyczynowości.", ""]
    if associations:
        text += ["| Zmienna | Osoby | Test | Statystyka | p permutacyjne | q BH |", "|---|---:|---|---:|---:|---:|"]
        for r in associations:
            text.append(f"| {r['covariate']} | {r['n_subjects']} | {r['statistic']} | {fmt(r['value'], 3)} | "
                        f"{fmt(r['p_permutation'], 4)} | {fmt(r['q_bh'], 4)} |")
        text += ["", "Korekta Benjamini–Hochberga dotyczy wyłącznie pięciu testów tej tabeli. "
                 "Liczebności poszczególnych kategorii i wartości: [participant_extensions.csv](participant_extensions.csv)."]
    else:
        text += ["Nie wygenerowano testów metadanych (brak tabeli lub wystarczających danych)."]
    text += ["", "## Metoda i ochrona przed wyciekiem etykiet", "",
             "1. Koniec referencyjny: pierwszy marker 1 po markerze 2 i przed następnym markerem 2. "
             "Czas trwania markera (1 s) jest ignorowany. Następny marker początku służy wyłącznie do budowy referencji, nigdy do ograniczania zakresu poszukiwania.",
             "2. Zakres kandydatów: od początku +4 s do min(początek +120 s, koniec zapisu −4 s), co 1 s. "
             "Nie zależy od prawdziwego końca ani od następnych zdarzeń. Krótkie zapisy trzeciej próby mogą nie obejmować rzeczywistego końca.",
             "3. Niepoprawne i brakujące próbki są oznaczane i interpolowane do filtracji; maski są zachowane. "
             "Kandydat wykorzystuje kanały z odsetkiem niepoprawnych próbek nie większym niż 5% w oknie. "
             "Po redukcji częstości próbkowania każda próbka otrzymuje flagę, jeśli którakolwiek z siedmiu oryginalnych próbek była niepoprawna. "
             "To konserwatywna maska: odsetek flag po redukcji może być wyższy niż w oryginale, a kryterium 5% dotyczy właśnie tej maski. "
             "Interpolacja nie odtwarza rzeczywistego pomiaru; duże braki pozostają ograniczeniem.",
             "4. OD = −log(I / mediana I), następnie resampling z filtrem antyaliasingowym (czynnik 7). "
             "Wolny przebieg: dolnoprzepustowo 0,4 Hz; pulsacja: 0,7–2,5 Hz i obwiednia Hilberta; SCI: korelacja 760/850 nm w paśmie 0,7–1,5 Hz. "
             "Krótkie pary: odległość źródło–detektor <15 mm; pozostałe są długie.",
             "5. Cechy na 4 s przed i 4 s po kandydacie: nachylenia, ich zmiana, poziom, lokalna zmienność, zmiana obwiedni pulsacji i zgodność zmian między parami. "
             "Skalowanie sygnału danej próby odnosi się do okresu od −30 do −5 s przed początkiem. "
             "Ruch jest pokazany na wykresach, ale nie jest wejściem detektora fNIRS.",
             "6. Reguła wybiera maksimum zmiany nachylenia, pulsacji lub ich zestawu; mały model to regresja logistyczna. "
             "Po RobustScaler cechy są ograniczane do przedziału [−10, 10] dla odporności numerycznej. "
             "Regresja logistyczna używa solvera liblinear (maks. 1000 iteracji, ziarno 42); te ustawienia stabilizują zbieżność i nie są wybierane na podstawie błędów testowych. "
             "W treningu dodatnie okna leżą ≤3 s od referencji, ujemne ≥6 s; pasmo pośrednie jest pomijane wyłącznie przy tworzeniu etykiet treningowych.",
             "7. Zewnętrzne LOPO: wszystkie próby pominiętej osoby są poza treningiem. Wewnętrzny GroupKFold (3 części) dobiera wariant reguły / C, próg SCI oraz korektę opóźnienia (ograniczoną do ±8 s). "
             "Korekta jest wyliczana z wewnętrznych predykcji; nie jest stałą fizjologiczną. RobustScaler uczy się wyłącznie na danych treningowych. "
             "Wybór ustawień minimalizuje średni błąd z karą 120 s za brak wskazania.",
             "8. W obu metodach wybierane jest maksimum wygładzonego wyniku (3 kandydaci). Wynik modelu nie jest skalibrowanym prawdopodobieństwem poprawnego zakończenia. "
             "Pole `peak_time_s` wskazuje surowe maksimum; `predicted_end_s` uwzględnia korektę opóźnienia z treningu i jest przesuwane do najbliższego dostępnego kandydata spełniającego warunki jakości. "
             "Pole `score` dotyczy surowego maksimum, nie skorygowanego czasu. "
             "Brak wskazania wynika z braku użytecznych kandydatów albo braku trenowalnej konfiguracji, nie z klinicznie zweryfikowanego progu pewności.", "",
             "## Ograniczenia interpretacji", "",
             "- Referencją jest marker eksperymentatora, nie niezależny zapis pierwszego wdechu z pasa oddechowego lub przepływomierza. "
             "Oceniamy zgodność z markerem, nie dokładność wykrycia fizjologicznego pierwszego wdechu.",
             "- Filtracja dwukierunkowa, transformacja Hilberta i okna po kandydacie korzystają z późniejszych próbek. "
             "To metoda offline; sam dostęp do +4 s nie opisuje całego zasięgu przyszłych danych filtrów. Wyniku nie należy przedstawiać jako detekcji online.",
             "- Trzy osoby obejrzane w przeglądzie rozwojowym wyłączono z głównego podsumowania, ale dobór krótkiej/średniej/długiej próby powoduje, że pozostały podzbiór nie jest losową próbą populacji. "
             "Podano też wszystkie wyniki LOPO, aby ujawnić tę różnicę.",
             "- 28 osób i dwa oznaczone końce na osobę to mały zbiór. Najlepszy z wielu wariantów wybrany po wynikach nie ma osobnego, niezależnego testu. "
             "Nie rozpoznajemy chorób ani nie szacujemy VO₂max, SpO₂ lub klinicznej reaktywności naczyniowej.", "",
             "## Odtworzenie i pliki", "",
             "Uruchom z katalogu projektu: `python -m fNIRS.breathback.run --stage all`. "
             "Można uruchomić kolejno `--stage prepare`, `--stage evaluate`, `--stage report`.", "",
             "- [events.csv](events.csv): tabela wszystkich zdarzeń, także bez referencji.",
             "- [montage_selection.csv](montage_selection.csv): nominalne pozycje i audyt wyboru wyłącznie par czoła.",
             "- [predictions_lopo.csv](predictions_lopo.csv): wszystkie zewnętrzne predykcje i błędy.",
             "- [predictions_unmarked.csv](predictions_unmarked.csv): demonstracje bez oceny trafności.",
             "- [metrics.csv](metrics.csv): wszystkie metryki i pokrycie.",
             "- [paired_comparisons.csv](paired_comparisons.csv): sparowane porównania z odniesieniem.",
             "- [quality_comparisons.csv](quality_comparisons.csv): sparowane porównania kontroli jakości z wariantem all.",
             "- [fold_audit.json](fold_audit.json): osoby treningowe, pominięte i ustawienia każdego folda.",
             "- [manifest.json](manifest.json): źródła, ich SHA-256, ustawienia i wersje bibliotek.",
             "- `signals/`: OD, przebieg wolny, pulsacja, ruch i maski niepoprawnych próbek.",
             "- `models/`: końcowe modele uczone na wszystkich oznaczonych próbach do kolejnych nagrań. "
             "Nie są używane do raportowanej oceny LOPO ani demonstracji pominiętych osób.", "",
             "Źródła metody: lokalny opis zbioru; [SCI — MNE](https://mne.tools/stable/generated/mne.preprocessing.nirs.scalp_coupling_index.html); "
             "[walidacja grupowa — scikit-learn](https://scikit-learn.org/stable/modules/cross_validation.html).", ""]
    (out/"REPORT.md").write_text("\n".join(text), encoding="utf-8")
