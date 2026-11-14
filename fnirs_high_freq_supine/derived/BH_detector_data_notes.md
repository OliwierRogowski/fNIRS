# Dane do pierwszego detektora

Dane zostały przygotowane do modelowania, ale **żaden model nie został utworzony ani wytrenowany**.

## Podział

- Próby 1–2 (56 zdarzeń) są etykietowanym materiałem do uczenia i oceny.
- Zastosowano 28 foldów leave-one-participant-out. W każdym foldzie wszystkie wiersze danej osoby są testowe, a jej próby nie występują w treningu. W `BH_detector_training_LOSO.csv` wiersze treningowe powtarzają się między foldami; wybór folda robi się przez `cv_test_subject_id`.
- Próba 3 każdej osoby jest materiałem prezentacyjnym. Zapisano okna od początku próby do końca nagrania bez etykiety końca. Dla spójności fold prezentacyjny wyklucza tę osobę z listy treningowej.
- `BH_detector_LOSO_folds.csv` zawiera listę osób treningowych i testową dla każdego foldu.

## Okna i cechy

- Okno: 5 s; krok: 1 s. Każde okno kończy się przed znacznikiem końca w próbach etykietowanych. Cechy korzystają tylko z próbek dostępnych do końca okna.
- Etykieta w próbach 1–2: `time_to_end_s`, czyli czas od końca okna do oznaczonego końca wstrzymania. Nie jest cechą wejściową.
- Cechy: mediana nachylenia OD wolnego, mediana zmiany nachylenia między połówkami okna, mediana lokalnego SD OD wolnego, mediana RMS OD pulsacyjnego oraz mediana korelacji zmian między kanałami. Cechy kanałowe agreguje się medianą.
- Wszystkie kanały w tym zbiorze pochodzą z montażu przedczołowego (czoło); nie ma tu kanałów z innych obszarów głowy. Warianty `short`, `long`, `both` porównują wyłącznie kanały czołowe. Ich spis jest w `BH_forehead_channels.csv`.
- Każdy wariant daje te same 5 zagregowanych cech fNIRS na okno, zamiast osobnej cechy dla każdego kanału. To utrzymuje mały wymiar wejścia; model nie został jeszcze zbudowany ani trenowany.
- Kanały nie są globalnie odrzucane według SCI, aby nie wybierać ich na podstawie osób testowych. `channels_used` liczy kanały z co najmniej 80% prawidłowych próbek w oknie. Podsumowanie SCI jest opisowe; późniejsze porównanie odrzucania słabych kanałów trzeba wykonać osobno wewnątrz foldów treningowych.
- Opisowo, mediany SCI par kanałów mieszczą się w zakresie 0,9847–0,9974; próg SCI ≥ 0,75 spełnia 25–28 z 28 nagrań dla każdej pary. Nie oznacza to, że jakość jest dobra w każdej próbie i u każdej osoby.
- `elapsed_since_hold_start_s` i indeksy kanałów są metadanymi. Model może użyć wyłącznie fNIRS cech, jeśli celem jest sprawdzenie informacji sygnałowej; kolumna czasu od początku pozwala jawnie ocenić wariant korzystający także z czasu.
- W plikach prezentacyjnych kolumna `time_to_end_s` jest pusta. Żaden znacznik końca nie został dopisany ani wywnioskowany.

## Pliki

- `BH_detector_training_LOSO.csv` — wiersze do uczenia we wszystkich foldach.
- `BH_detector_test_LOSO.csv` — wiersze testowe, każda osoba występuje tylko w swoim foldzie.
- `BH_detector_presentation.csv` — okna trzecich prób bez etykiet.
- `BH_detector_LOSO_folds.csv` — manifest foldów.
- `BH_forehead_channels.csv` — kanały czołowe używane w porównaniu oraz ich opisowa jakość SCI.

Liczba wierszy: trening 84,321, test 3,123, prezentacja 2,208.
Liczby wierszy nie są liczbą niezależnych przykładów: 5-sekundowe okna nakładają się co 1 s, a wiersze treningowe powtarzają się między foldami. Dane obejmują 56 oznaczonych wstrzymań u 28 osób.
Punkt odniesienia do porównania to LOSO MAE 8,210071 s. Detektor należy oceniać na tych samych foldach; jego MAE powinno być niższe.
