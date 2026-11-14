# fNIRS — BreathBack

Prototyp offline szacujący koniec wstrzymania oddechu ze znanego początku.
Wykorzystuje wyłącznie pary optod czoła: obie optody muszą spełniać kryterium
geometryczne Fp/AFp/AF. Porównuje regułę zmiany i regresję logistyczną z metodą
odniesienia opartą na typowym czasie bezdechu.

- [Instrukcja i opis metody](breathback/README.md)
- [Zagregowane wyniki i ograniczenia](docs/RESULTS.md)

## Uruchomienie

Z katalogu nadrzędnego zawierającego `fNIRS`:

```powershell
python -m pip install -r fNIRS/breathback/requirements.txt
python -m fNIRS.breathback.run --stage all
python -m unittest fNIRS.breathback.test_pipeline -v
```

Lokalne dane należy umieścić w `fnirs_high_freq_supine/data/BH/`, a tabelę
uczestników w `fnirs_high_freq_supine/subjects_description.csv`. Zbiór nie jest
dołączony do repozytorium. Jego wykorzystanie podlega warunkom dostawcy.

Szczegółowy raport powstaje lokalnie w `breathback_results/REPORT.md`.
Źródłowe nagrania, metadane uczestników, indywidualne predykcje i modele
pozostają poza repozytorium. W `docs/` znajdują się tylko wyniki zagregowane.
