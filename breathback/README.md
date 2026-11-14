# BreathBack

Estymacja końca wstrzymania oddechu z fNIRS przy znanym początku. Prototyp
analizuje nagrania offline; referencją jest marker eksperymentalny, nie pomiar
pierwszego wdechu. Nie służy do diagnostyki, oceny bezpieczeństwa bezdechu ani
monitorowania na żywo.

## Wyłącznie optody na czole

Analiza stosuje ścisły filtr montażu zamiast korzystać z całego układu
przedczołowego. **Zarówno źródło, jak i detektor** danej pary muszą mieć najbliższy
punkt orientacyjny głowy w grupie `Fp*`, `AFp*` lub `AF*` i leżeć nie dalej niż
25 mm od tego punktu. Punkty `AFF*`, `F*` i `FC*` są wyłączone; `Fp*` jest jawnie
dopuszczonym wyjątkiem od ogólnego wyłączenia punktów `F*`. Nie wystarcza, że
na czole jest tylko jedna optoda albo środek kanału.

Jest to operacyjna definicja obszaru czoła na podstawie **nominalnych współrzędnych
montażu zapisanych w SNIRF**. Nie oznacza indywidualnego pomiaru rzeczywistego
położenia optod ani potwierdzenia, że skóra pod nimi była pozbawiona włosów.

W dostarczonym montażu supine pozostaje 6 par (12 kolumn: 2 długości fali
na parę), obejmujących 4 pary długie i 2 krótkie:

| Decyzja | Pary źródło–detektor |
|---|---|
| Zachowane | S5–D3, S5–D4, S6–D4, S6–D5, S7–D4, S8–D5 |
| Odrzucone | S1–D1, S2–D1, S3–D2, S4–D2 |

Pełny audyt wyboru jest zapisywany w `breathback_results/montage_selection.csv`.
Odrzucone pary nie są wejściem cech, detektora ani porównania krótkie/długie.
Po zmianie obszaru trzeba przeliczyć przygotowanie, modele i ocenę — wyników
poprzedniego pełnego montażu nie należy przedstawiać jako wyników tylko z czoła.

## Uruchomienie

Polecenia wykonuj z katalogu `hackhaton`:

```powershell
python -m pip install -r fNIRS/breathback/requirements.txt
python -m fNIRS.breathback.run --stage all
python -m unittest fNIRS.breathback.test_pipeline -v
```

Można osobno uruchomić `--stage prepare`, `--stage evaluate` lub `--stage report`.
Etap evaluate wykorzystuje lokalny `features.joblib`, etap report gotowe wyniki.
Po zmianach sposobu przygotowania cech trzeba ponownie wykonać prepare i evaluate.
Parametry `--data` i `--output` pozwalają zmienić katalogi.

Gotowy raport: `fNIRS/breathback_results/REPORT.md`.

## Co wykonuje program

1. Tworzy tabelę zdarzeń: marker 2 → następny marker 1, przed kolejnym początkiem.
   Pozostawia trzeci bezdech bez referencji. Ignoruje zastępcze długości markerów.
2. Wybiera krótki, środkowy i długi przykład; ich trzech uczestników oznacza jako
   rozwojowych. Raport pokazuje zarówno wszystkie 28 osób, jak i pozostałe 25.
   Wyłączenie przykładów zawęża rozkład czasów, więc pokazujemy oba wyniki.
3. Oznacza braki, wartości niedodatnie i niefinitywne; interpoluje je przed filtracją.
   Zapisuje maskę oryginalnych niepoprawnych próbek oraz sygnały po przetworzeniu.
4. Ogranicza pary do ścisłego obszaru czoła zgodnie z regułą powyżej. Przelicza
   OD, oddziela wolny sygnał, pulsacje i obwiednię; rozdziela zachowane krótkie
   i długie pary według geometrii. Domyślny graniczny dystans: 15 mm.
5. Tworzy kandydatów co 1 s od początku +4 s do początku +120 s, ograniczonych
   końcem nagrania minus 4 s. Ani koniec referencyjny, ani następne markery nie
   określają tego obszaru. Okna obejmują 4 s przed kandydatem i 4 s po nim.
   Filtracja przód–tył całego nagrania korzysta również z dalszego kontekstu.
6. Porównuje medianę czasu bezdechu osób treningowych, regułę zmiany oraz małą
   regresję logistyczną. Regresja odróżnia kandydatów w odległości ≤3 s od
   referencji od odległych ≥6 s; nie używa czasu od początku jako cechy.
7. Zewnętrzne LOPO: oba oznaczone bezdechy i demo danej osoby są przewidywane
   modelem nauczonym na pozostałych osobach. Wewnętrzny GroupKFold (3 części)
   wybiera regułę/C, próg SCI (0,5/0,75/0,9) w wariancie QC i korektę opóźnienia.
   Skalowanie cech jest dopasowane wyłącznie na treningu. Po skalowaniu wartości
   są ograniczane do ±10 dla stabilności numerycznej.
8. Przewidywany koniec to maksimum wygładzonego wyniku + korekta czasowa
   nauczona wewnątrz treningu (maks. ±8 s), przesunięte na najbliższego
   dopuszczalnego kandydata. Wynik `score` dotyczy nieprzesuniętego maksimum
   `peak_time_s`; nie jest skalibrowaną pewnością ani prawdopodobieństwem poprawności.
9. Zapisuje metryki, audyt podziałów i parametrów, predykcje trzecich prób,
   12 modeli finalnych, wykresy i analizę związku metadanych z błędem detektora.

Bez QC nadal wyłączane są okna z nadmiarem braków (>5% bloków po redukcji
próbkowania). Maskę redukujemy przez „jakakolwiek niepoprawna próbka w bloku”,
więc próg jest bardziej konserwatywny niż 5% próbek oryginalnych. Warianty QC
dodatkowo wymagają SCI w lokalnym 8-sekundowym oknie. Gdy nie ma dopuszczalnego
kandydata, detektor odmawia wskazania. Odsetki ±5/±10 s mają w mianowniku
wszystkie oceniane próby, również bez wskazania.

## Nowe nagranie

```powershell
python -m fNIRS.breathback.predict "sciezka/do/Breath holding_V2_61_XX_BH_pomiar.snirf"
```

Domyślnie bierze początki z markerów 2. Można podać `--onset 70.0` (wielokrotnie),
`--model` i `--output`. Obsługiwany jest układ i nazewnictwo zbioru supine.
Modele finalne są nauczone na wszystkich oznaczonych próbach, więc nie należy
ich używać do deklarowania skuteczności na tych samych plikach. Do tego służą
wyłącznie `predictions_lopo.csv`. Dema trzecich prób również powstają w LOPO,
bez treningu na pierwszych dwóch próbach pokazywanej osoby.

Włosy, wiek i aktywność nie są wejściami detektora. Rozszerzenie bada ich
związek z błędem na nowych osobach; nie ustanawia biomarkerów sprawności.
Rodzina pięciu eksploracyjnych testów ma korektę FDR. Nie łączymy identyfikatorów
uczestników DOT i supine. Program nie modyfikuje źródłowych SNIRF i CSV.

## Źródła i odtwarzalność

- Lokalny `fnirs_high_freq_supine/DATASET_DESCRIPTION.md`.
- SCI: https://mne.tools/stable/generated/mne.preprocessing.nirs.scalp_coupling_index.html
- Podziały grupowe: https://scikit-learn.org/stable/modules/cross_validation.html
- `manifest.json`: ustawienia, wersje bibliotek i SHA256 źródłowych SNIRF.
- `montage_selection.csv`: nominalne położenie optod, decyzje wyboru par i przyczyny odrzucenia.
- `fold_audit.json`: osoby treningowe oraz wszystkie wyniki wewnętrznego doboru.
- Użytkowanie zbioru podlega warunkom jego dostawcy zapisanym w opisie danych.
