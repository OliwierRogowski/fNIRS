# fNIRS BreathBack

BreathBack to prototyp offline szacujący **koniec wstrzymania oddechu przy znanym początku** na podstawie sygnałów fNIRS. Marker `2` wyznacza początek, a następny marker `1` jest referencją końca w dwóch oznaczonych próbach. Program nie wykrywa początku samodzielnie; marker końca nie jest niezależnym pomiarem przepływu powietrza ani pierwszego wdechu. Projekt nie jest narzędziem klinicznym ani systemem czasu rzeczywistego.

## Struktura projektu

- `breathback/` — przetwarzanie sygnałów, detektory, raportowanie i predykcja.
  - `run.py` — analiza bazowa z parami optod wybranymi z obszaru czoła.
  - `mean_channels.py` — osobny wariant wykorzystujący wszystkie kompletne pary montażu; stroi regresję logistyczną i MLP oraz ocenia je metodą LOPO.
  - `predict.py` — predykcja dla nowego pliku z użyciem zapisanego modelu bazowego.
- `fnirs_high_freq_supine/` — opis lokalnego zbioru i skrypty przygotowania sygnałów. Surowe pliki `.snirf` są lokalne i ignorowane przez Git.
- `ui/` — aplikacja Streamlit do przeglądania prób, cech i predykcji.
- `docs/` — wyniki agregatowe, metryki i wykresy.
- `breathback_results/` — lokalne artefakty przetwarzania i uczenia: tabele CSV, modele, audyty oraz wykresy. Katalog jest ignorowany przez Git.

## Przygotowanie środowiska

Uruchamiaj polecenia z katalogu głównego tego repozytorium:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r breathback/requirements.txt
```

Umieść lokalne nagrania zadania BH w `fnirs_high_freq_supine/data/BH/`. Dla analizy opisanej w `breathback/README.md` opcjonalna tabela uczestników znajduje się w `fnirs_high_freq_supine/subjects_description.csv`. Zbiór nie jest częścią repozytorium; stosuj warunki jego dostawcy.

## Uruchomienie analiz

Analiza czołowa, porównująca metody regułowe, logistyczne i czas trwania:

```bash
python -m breathback.run --stage all
```

Uczenie wariantu ze wszystkimi kompletnymi parami kanałów, w tym MLP:

```bash
python -m breathback.mean_channels --stage all --optuna-trials 12
```

Druga komenda przygotowuje dane i wykonuje zewnętrzne podziały leave-one-participant-out. Wyniki zapisuje domyślnie do `breathback_results/all_channels_mean/`. Predykcje dla prób z oznaczonym końcem, demonstracje prób bez markera końca, metryki, parametry strojenia i finalny model MLP są w osobnych plikach. Pełne objaśnienie wejść sieci oraz wyniki ponownego treningu są w [raporcie audytu MLP](docs/MLP_AUDYT.odt), a skrócony zapis kontrolny w [pliku z dowodami treningu](docs/mlp_retrain_evidence.json).

## Interfejs Streamlit

Zainstaluj zależności interfejsu i uruchom go z katalogu repozytorium:

```bash
python -m pip install -r ui/requirements.txt
python -m streamlit run ui/app.py
```

Aplikacja korzysta z lokalnych plików w `breathback_results/all_channels_mean/` oraz z wag JSON `wagi-dziala.json`. Dla prób 1–2 pokazuje predykcje LOPO; trzecia próba nie ma oznaczonego końca i służy wyłącznie jako demonstracja działania finalnego modelu.

## Wyniki i ograniczenia

`docs/RESULTS.md` opisuje dotychczasową ocenę wariantu czołowego. Raport MLP dotyczy innego wariantu: wszystkich 10 kompletnych par, 56 oznaczonych prób u 28 osób i zagnieżdżonej walidacji grupowej. Wyników tych wariantów nie należy ze sobą utożsamiać. Szczegóły filtracji, etykiet, cech, zapisanych plików, metryk i ograniczeń są w [raporcie MLP](docs/MLP_AUDYT.odt).

Modele końcowe są dopasowywane na całym zbiorze oznaczonym i służą do demonstracji lub predykcji nowych nagrań. Do oceny jakości używaj wyłącznie predykcji z pominiętymi uczestnikami w pliku `lopo_predictions.csv`. Wartość `score` nie jest skalibrowanym prawdopodobieństwem poprawności.
