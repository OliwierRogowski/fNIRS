# Przygotowanie sygnału BH — pierwsza wersja

## Wynik

- Przetworzono 28 nagrań BH z `data/BH/`.
- Oryginalne pliki SNIRF pozostają bez zmian.
- W `BH_signals/` zapisano po jednym skompresowanym pliku HDF5 na nagranie. Każdy zawiera grupy `signals/short` i `signals/long`, a w nich równoległe przebiegi `od`, `od_slow` i `od_pulse` oraz maskę nieprawidłowych próbek.
- `BH_channel_quality.csv` zawiera 280 ocen par źródło–detektor. 267 ma status `good`, a 13 `review`: 12 par ma SCI poniżej 0,75, jedna dodatkowo nie osiąga SNR 5. Żaden kanał nie został usunięty.
- Metadane oznaczają 1 530 brakujących próbek z 1 160 737 wierszy (0,132%). Wśród wartości intensywności nie znaleziono próbek niefinitych, zerowych ani ujemnych.
- W 4 932 miejscach `sample_index` nie zmienia się o dokładnie 1; zapisano je osobną flagą ciągłości i nie maskowano automatycznie, bo wektor czasu pozostaje równomierny.

## Reguły

1. **Braki i błędne próbki.** Flaga `missing_sample` z SNIRF oraz wartości intensywności niefinitych lub `<= 0` są zapisywane osobno i łącznie w `invalid_sample_mask`. Takie wiersze mają `NaN` w OD. Zmiany w `sample_index` są niezależnie zapisane jako `sample_index_discontinuity_after`; nie przesuwają osi czasu ani same nie unieważniają próbki.
2. **Kontakt optod ze skórą.** Dla każdej pary źródło–detektor obliczono SCI: korelację przy zerowym przesunięciu między pulsacyjnymi składowymi obu długości fal po filtracji 0,5–2,5 Hz. `SCI < 0,75`, SNR surowej intensywności `min/SD < 5` lub udział prawidłowych próbek poniżej 95% daje status `review`. To sygnał do sprawdzenia, nie automatyczne odrzucenie. Próg SCI 0,75 jest często stosowany w ocenie jakości kontaktu, ale nie jest uniwersalnym prawem.
3. **Intensywność do OD.** Dla każdego kanału `I0` to średnia dodatnia intensywność z prawidłowych próbek; `OD = -log10(I / I0)`. `I0` i SNR intensywności są zapisane w metadanych kanałów. Intensywność surowa pozostaje w pliku źródłowym.
4. **Odległości kanałów.** Odległość źródło–detektor pochodzi z geometrii sondy SNIRF. Kanały do 15 mm włącznie są krótkie, pozostałe długie. W tym zestawie podział daje 4 kanały krótkie i 16 długich, licząc obie długości fal.
5. **Zachowanie wolnej i szybkiej dynamiki.** Oprócz nieprzefiltrowanego OD zapisano równolegle `od_slow` (Butterworth, 4. rząd, filtr dolnoprzepustowy 0,5 Hz) i `od_pulse` (Butterworth, 4. rząd, pasmo 0,5–2,5 Hz). Filtracja jest bez przesunięcia fazowego. Braki są interpolowane liniowo wyłącznie na czas filtracji, a potem ponownie maskowane jako `NaN`. Sygnał pulsacyjny nie jest usuwany z danych.

## Odtworzenie

Z katalogu repozytorium uruchom:

```bash
python fnirs_high_freq_supine/prepare_bh_signals.py
```

Wymagane biblioteki: `h5py`, `numpy`, `scipy`.

## Odniesienia metodyczne

- [Specyfikacja SNIRF](https://github.com/fNIRS/snirf/blob/master/snirf_specification.md)
- [fNIRS Society: konwersja intensywności do OD](https://fnirs.org/courses/introduction-to-fnirs/lessons/from-raw-intensity-to-hemoglobin/)
- [Signal quality index i definicja SCI](https://pmc.ncbi.nlm.nih.gov/articles/PMC7687963/)
- [Najlepsze praktyki raportowania filtracji fNIRS](https://pmc.ncbi.nlm.nih.gov/articles/PMC7793571/)
