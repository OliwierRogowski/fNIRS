# fNIRS · UI

Panel Streamlit czyta tabelę zdarzeń i przygotowane okna z `fnirs_high_freq_supine/derived/`, a artefakt wag z `breathback/weights/wagi-dziala.json`. Wyświetla też metryki zapisane w JSON oraz punkt odniesienia z przygotowanej oceny.

Sam plik wag nie zawiera danych konkretnej próby. Wymaga 14 cech wejściowych; dostępny plik okien ma 5 innych cech. Ponadto opis wag wskazuje wszystkie kanały montażu, także spoza czoła. UI pokazuje te metryki informacyjnie, ale nie przedstawia ich jako wyniku modelu czołowego i nie generuje na ich podstawie predykcji.

Próby 1–2 są oceniane, a próba 3 służy do demonstracji bez etykiety końca. Metadane włosów i aktywności pozostają analizą dodatkową.

## Instalacja i uruchomienie w PowerShell

Z katalogu głównego repozytorium `fNIRS`:

```powershell
$python312 = "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe"
& $python312 -m pip install -r .\ui\requirements.txt
& $python312 -m streamlit run .\ui\app.py
```

Interfejs działa bez plików SNIRF: pokazuje przygotowaną tabelę zdarzeń, podgląd cech, wagi i zapisane metryki. Do predykcji z wagi potrzebny jest zgodny zestaw 14 cech oraz zgodność zakresu kanałów.
