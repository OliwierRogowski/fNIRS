# fNIRS · UI

Panel badania fNIRS z widokiem uczestników i polem wyszukiwania. Interfejs nie zawiera jeszcze danych — można je podłączyć, gdy będą dostępne.

## Uruchomienie w PowerShell

Z katalogu głównego repozytorium `fNIRS`:

```powershell
$python312 = "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe"
& $python312 -m pip install -r .\ui\requirements.txt
& $python312 -m streamlit run .\ui\app.py
```
