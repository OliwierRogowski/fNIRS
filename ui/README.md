# fNIRS · prototyp BreathBack

Streamlitowy prototyp podłączony do uczestników i prób ze zbioru
`fnirs_high_freq_supine/data/BH`. Pokazuje sygnał uśredniony z dostępnych par,
znaczniki i prognozę końca wstrzymania. Próby 1–2 używają zapisanych predykcji
LOPO, a próbę 3 oblicza bezpośrednio z wag `wagi-dziala.json`.

Interfejs czyta lokalne pliki przygotowane przez `breathback.mean_channels`:

- `breathback_results/all_channels_mean/event_targets.csv` — katalog 28 osób i 84 prób;
- `candidate_features.csv` — cechy wejściowe do inferencji wag JSON;
- `lopo_predictions.csv` i `lopo_metrics.csv` — uczciwa ocena prób 1–2 po osobach;
- źródłowe pliki `.snirf` — przebieg wolnej OD do wykresu.

## Pierwsze uruchomienie

W katalogu głównym repozytorium:

```bash
python -m pip install -r breathback/requirements.txt
python -m pip install -r ui/requirements.txt
python -m breathback.mean_channels --stage all
python -m streamlit run ui/app.py
```

Przygotowanie i trenowanie generuje lokalne pliki w `breathback_results/`.
Jeśli te wyniki już istnieją, pomiń polecenie `--stage all`. Nagrania BH są
lokalne i ignorowane przez Git; do wykresów muszą być dostępne w
`fnirs_high_freq_supine/data/BH/`.

## Widoki

- **Podgląd próby:** wybór uczestnika i próby, wykres OD z markerem początku,
  końcem referencyjnym i predykcją. Próby 1–2 pokazują wynik LOPO, a próba 3
  prognozę obliczoną przez MLP z `wagi-dziala.json`.
- **Ocena modelu:** mediana błędu, MAE, wyniki w ±5/±10 s i porównanie z metodą
  typowego czasu trwania. Nie używa klasycznego `accuracy`, bo model szacuje czas.
- **Baza uczestników:** katalog ID, liczba prób, oznaczone końce i dostępne okna
  cech; można pobrać tabelę zdarzeń oraz predykcje LOPO w CSV.

Lokalna baza pokazuje identyfikatory uczestników zamiast inicjałów z nazw plików.
Wskaźnik dostępnych par opisuje kompletność cech w wybranym oknie; nie jest
bezpośrednim pomiarem kontaktu optod ze skórą. Aplikacja działa offline i ma
charakter demonstracyjny, nie kliniczny.
