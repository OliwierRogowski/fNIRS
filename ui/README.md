# fNIRS · UI

Panel do analizy końca wstrzymania oddechu na podstawie gotowych nagrań fNIRS.

Pierwszy zbiór docelowy to `fnirs_high_freq_supine/data/BH`: 28 nagrań i 56 oznaczonych wstrzymań (próby 1–2). Próba 3 służy do demonstracji predykcji bez znanego końca. Interfejs pokazuje plan analizy i miejsca na dane; nie zawiera jeszcze podłączonych sygnałów ani wyliczonych wyników.

Walidacja pierwszej wersji ma odbywać się schematem leave-one-subject-out. Metadane włosów i aktywności są odłożone do analizy dodatkowej.

## Uruchomienie w PowerShell

Z katalogu głównego repozytorium `fNIRS`:

```powershell
$python312 = "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe"
& $python312 -m pip install -r .\ui\requirements.txt
& $python312 -m streamlit run .\ui\app.py
```
