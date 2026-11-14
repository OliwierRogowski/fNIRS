# Prosta metoda odniesienia: typowy czas wstrzymania

Predykcja końca: `początek wstrzymania + mediana czasu trwania w danych treningowych`. 
Mediana jest liczona z oznaczonych prób 1–2; próby 3 nie uczestniczą w treningu.

## Ocena na oznaczonych końcach

- Podział: leave-one-participant-out — obie oznaczone próby danej osoby są wyłączone z jej treningu.
- Uczestników: 28; ocenianych zdarzeń: 56.
- MAE: **8.210071 s**.
- Mediana błędu bezwzględnego: **4.333250 s**.

MAE jest główną miarą porównania. Detektor powinien uzyskać niższe MAE na tych samych foldach uczestników.

## Predykcje prób 3

- Zapisano 28 predykcji bez błędu, ponieważ koniec próby 3 nie jest oznaczony.
- Dla każdej osoby użyto mediany treningowej z pozostałych uczestników, zgodnie z oceną leave-one-participant-out.

## Pliki

- `BH_baseline_evaluation.csv` — predykcje i błędy dla prób 1–2.
- `BH_baseline_demo.csv` — przewidywane końce prób 3.

- MAE dla próby 1: 8.502732 s (28 zdarzeń).
- MAE dla próby 2: 7.917411 s (28 zdarzeń).
