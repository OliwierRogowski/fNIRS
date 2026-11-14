from __future__ import annotations

import streamlit as st


st.set_page_config(
    page_title="Koniec wstrzymania oddechu · fNIRS",
    page_icon="〰",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
    :root { --ink:#14232d; --muted:#71808a; --teal:#078f80; --line:#e5e9e8; --paper:#f6f8f6; }
    .stApp { background:var(--paper); color:var(--ink); font-family:'DM Sans',sans-serif; }
    [data-testid="stHeader"] { background:transparent; }
    section[data-testid="stSidebar"] { background:#17252e; border-right:1px solid #263640; }
    section[data-testid="stSidebar"] > div { padding:1.35rem 1rem; }
    section[data-testid="stSidebar"] * { color:#eef5f5; }
    .brand { display:flex; align-items:center; gap:12px; padding:6px 8px 28px; }
    .brand-mark { color:#22b5a3; font-size:31px; line-height:1; font-weight:700; }
    .brand-name { font-family:'Manrope',sans-serif; font-size:21px; font-weight:800; letter-spacing:-.7px; }
    .brand-sub { color:#a9b6bb; font-size:12px; margin-top:1px; }
    .side-kicker { color:#9babb1 !important; font-size:10px; text-transform:uppercase; letter-spacing:1.5px; margin:0 9px 8px; }
    .side-focus { border:1px solid #405058; background:#202f38; border-radius:10px; padding:12px; margin:0 0 18px; }
    .side-focus-label { color:#9eb0b5; font-size:10px; text-transform:uppercase; letter-spacing:1px; }
    .side-focus-value { color:#fff; font-size:13px; font-weight:700; margin-top:5px; }
    .side-step { display:flex; gap:10px; align-items:flex-start; color:#ced9dc; font-size:12px; padding:9px 7px; line-height:1.45; }
    .side-step b { color:#40c2ae; font-size:11px; }
    .side-footer { position:fixed; bottom:20px; color:#98a9af; font-size:11px; border-top:1px solid #405058; padding:13px 7px 0; width:calc(15.5rem - 2rem); }
    .main .block-container { max-width:1500px; padding-top:2rem; padding-bottom:2.5rem; }
    h1 { font-family:'Manrope',sans-serif !important; font-size:32px !important; font-weight:800 !important; letter-spacing:-1.1px !important; color:var(--ink) !important; margin:0 0 5px !important; }
    h2,h3 { font-family:'Manrope',sans-serif !important; color:var(--ink) !important; letter-spacing:-.35px !important; }
    .eyebrow { color:var(--teal); font-weight:700; font-size:10px; text-transform:uppercase; letter-spacing:1.5px; margin-bottom:7px; }
    .subtitle { color:var(--muted); font-size:14px; margin-bottom:20px; }
    .topline { display:flex; justify-content:space-between; align-items:flex-start; gap:20px; }
    .status-pill { display:inline-flex; align-items:center; gap:8px; border:1px solid #dce6e3; background:#edf7f4; color:#247c70; padding:8px 12px; border-radius:999px; font-size:11px; font-weight:700; margin-top:8px; }
    .status-dot { width:7px; height:7px; border-radius:50%; background:#18a58f; }
    .summary-row { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:12px; margin:5px 0 18px; }
    .summary-card { background:white; border:1px solid var(--line); border-radius:12px; padding:14px 16px; min-height:82px; }
    .summary-label { color:#77858c; font-size:11px; margin-bottom:8px; }
    .summary-value { font-family:'Manrope',sans-serif; color:var(--ink); font-weight:800; font-size:20px; }
    .summary-note { color:#96a0a4; font-size:10px; margin-left:5px; font-weight:500; }
    .panel { background:white; border:1px solid var(--line); border-radius:13px; padding:17px 19px; margin-bottom:13px; }
    .panel-title { font-family:'Manrope',sans-serif; font-size:16px; font-weight:800; margin-bottom:3px; }
    .panel-caption { color:#879399; font-size:11px; }
    .step-card { border:1px solid var(--line); border-radius:11px; background:#fff; padding:14px; min-height:112px; }
    .step-num { color:var(--teal); font-weight:800; font-size:10px; letter-spacing:1px; margin-bottom:8px; }
    .step-title { color:#263740; font-size:13px; font-weight:700; margin-bottom:5px; }
    .step-copy { color:#829097; font-size:11px; line-height:1.5; }
    .empty-state { text-align:center; padding:44px 18px; border:1px dashed #d8e1df; border-radius:11px; background:#fcfdfc; }
    .empty-icon { width:46px; height:46px; margin:0 auto 12px; border-radius:14px; display:flex; align-items:center; justify-content:center; background:#eaf6f3; color:#078f80; font-size:22px; }
    .empty-title { font-family:'Manrope',sans-serif; color:#263740; font-size:15px; font-weight:800; margin-bottom:5px; }
    .empty-copy { color:#829097; font-size:12px; max-width:510px; line-height:1.6; margin:0 auto; }
    .legend { display:flex; gap:16px; color:#71808a; font-size:11px; margin:8px 0 13px; }
    .legend span { display:flex; align-items:center; gap:6px; }
    .legend i { width:9px; height:9px; border-radius:50%; display:inline-block; }
    div[data-testid="stButton"] button { border:1px solid #d7e2df; border-radius:9px; color:#52636b; background:white; font-weight:600; min-height:40px; }
    div[data-testid="stButton"] button:hover { border-color:#0b9b89; color:#087c70; }
    div[data-testid="stTextInput"] input,div[data-testid="stSelectbox"] div[data-baseweb="select"] > div { border-color:#dce2e1; border-radius:9px; }
    .stTabs [data-baseweb="tab-list"] { gap:7px; }
    .stTabs [data-baseweb="tab"] { border-radius:8px 8px 0 0; }
    .bottom-note { color:#929da1; font-size:10px; text-align:center; margin-top:17px; }
    @media (max-width:850px) { .summary-row { grid-template-columns:repeat(2,minmax(0,1fr)); } .topline { display:block; } }
    </style>
    """,
    unsafe_allow_html=True,
)


EVENT_COLUMNS = ["ID osoby", "Numer próby", "Początek [s]", "Koniec [s]", "Czas trwania [s]"]
TRAINING_COLUMNS = ["ID osoby", "Numer próby", "Czas [s]", "HbO", "HbR", "Etykieta"]


@st.dialog("Podgląd danych treningowych", width="large")
def show_training_preview() -> None:
    st.caption("Osoby w treningu są rozdzielane od osoby ocenianej. Fragmenty jednej osoby nie trafiają do obu zbiorów.")
    training_rows: list[dict[str, object]] = []
    if training_rows:
        st.dataframe(training_rows, use_container_width=True, hide_index=True)
    else:
        st.info("Dane treningowe pojawią się tutaj po podłączeniu nagrań i tabeli zdarzeń.")
        st.dataframe({column: [] for column in TRAINING_COLUMNS}, use_container_width=True, hide_index=True)


with st.sidebar:
    st.markdown(
        '<div class="brand"><div class="brand-mark">〰</div><div><div class="brand-name">fNIRS</div><div class="brand-sub">Detektor wstrzymania oddechu</div></div></div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="side-kicker">Cel pierwszej wersji</div>', unsafe_allow_html=True)
    st.markdown('<div class="side-focus"><div class="side-focus-label">Szukamy</div><div class="side-focus-value">Momentu zakończenia wstrzymania</div></div>', unsafe_allow_html=True)
    st.markdown('<div class="side-kicker">Kolejność prac</div>', unsafe_allow_html=True)
    st.markdown('<div class="side-step"><b>01</b><span>Zdarzenia i sygnały</span></div><div class="side-step"><b>02</b><span>Detektor i walidacja po osobach</span></div><div class="side-step"><b>03</b><span>Jakość, ruch i kanały</span></div><div class="side-step"><b>04</b><span>Włosy i aktywność — później</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="side-footer">Badanie pilotażowe · gotowe nagrania</div>', unsafe_allow_html=True)


st.markdown(
    '<div class="topline"><div><div class="eyebrow">Panel analizy · etap 1</div><h1>Koniec wstrzymania oddechu</h1><div class="subtitle">Na podstawie sygnału fNIRS szacujemy moment zakończenia wstrzymania. Początek wyznacza znacznik.</div></div><div class="status-pill"><span class="status-dot"></span>Plan analizy · BH</div></div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="summary-row">'
    '<div class="summary-card"><div class="summary-label">Zbiór pierwszej wersji</div><div class="summary-value">BH <span class="summary-note">supine · high freq</span></div></div>'
    '<div class="summary-card"><div class="summary-label">Nagrania w planie</div><div class="summary-value">28 <span class="summary-note">nagrań</span></div></div>'
    '<div class="summary-card"><div class="summary-label">Oznaczone wstrzymania</div><div class="summary-value">56 <span class="summary-note">próby 1–2</span></div></div>'
    '<div class="summary-card"><div class="summary-label">Demonstracja</div><div class="summary-value">Próba 3 <span class="summary-note">bez końca referencyjnego</span></div></div>'
    '</div>',
    unsafe_allow_html=True,
)

preview_tab, evaluation_tab = st.tabs(["Podgląd próby", "Ocena metody"])

with preview_tab:
    left, right = st.columns([1.4, 1], gap="medium")
    with left:
        st.markdown('<div class="panel-title">Podgląd sygnału</div><div class="panel-caption">Wybierz nagranie i próbę po podłączeniu plików.</div>', unsafe_allow_html=True)
        controls = st.columns([1.2, 1, 1.1])
        with controls[0]:
            st.selectbox("Osoba", ["Wybierz osobę"], disabled=True)
        with controls[1]:
            st.selectbox("Nagranie", ["Wybierz nagranie"], disabled=True)
        with controls[2]:
            st.selectbox("Próba", ["Próba 3 · demo"], disabled=True)
        st.markdown('<div class="legend"><span><i style="background:#078f80"></i>Sygnał fNIRS</span><span><i style="background:#e6a33a"></i>Początek · znacznik 2</span><span><i style="background:#e76e67"></i>Koniec · znacznik 1</span></div>', unsafe_allow_html=True)
        st.markdown('<div class="empty-state"><div class="empty-icon">〰</div><div class="empty-title">Sygnał pojawi się po podłączeniu nagrania</div><div class="empty-copy">W demonstracji pokażemy przebieg, początek wstrzymania, przewidywany koniec oraz — dla prób oznaczonych — rzeczywisty znacznik końca.</div></div>', unsafe_allow_html=True)
    with right:
        st.markdown('<div class="panel-title">Wynik detektora</div><div class="panel-caption">Próba demonstracyjna · bez etykiety końca</div>', unsafe_allow_html=True)
        st.metric("Przewidywany koniec", "—", help="Wyznaczony wyłącznie z sygnału po początku wstrzymania.")
        st.metric("Błąd względem znacznika", "Nie dotyczy", help="Próba 3 nie ma oznaczonego końca i nie służy do oceny błędu.")
        st.metric("Jakość zapisu", "Do oceny")
        st.markdown('<div class="panel-caption">Próba 3 służy do demonstracji przewidywania. Prawdziwy koniec nie może zawężać obszaru poszukiwań detektora.</div>', unsafe_allow_html=True)
        if st.button("▤  Podgląd danych treningowych", use_container_width=True):
            show_training_preview()

    st.markdown('<div class="bottom-note">Znacznik końca jest referencją eksperymentalną; nie oznacza niezależnego pomiaru pierwszego wdechu.</div>', unsafe_allow_html=True)
    st.divider()
    st.markdown('<div class="panel-title">Tabela zdarzeń</div><div class="panel-caption">Jeden wiersz na wstrzymanie oddechu. Czas bierzemy z wystąpienia znacznika, nie z jego pola duration.</div>', unsafe_allow_html=True)
    st.info("Tabela zostanie wypełniona po podłączeniu plików zdarzeń. Początek: znacznik 2. Koniec: następny znacznik 1.")
    st.dataframe({column: [] for column in EVENT_COLUMNS}, use_container_width=True, hide_index=True)
    with st.expander("Jak podzielimy próby?"):
        st.markdown("**Próby 1–2 — ocena metody:** 56 oznaczonych wstrzymań służy do wyznaczania zdarzeń i oceny predykcji. Walidację prowadzimy całymi osobami.")
        st.markdown("**Próba 3 — demonstracja:** nie ma oznaczonego końca, więc pokazuje przewidywanie bez wyniku błędu.")

with evaluation_tab:
    st.markdown('<div class="panel-title">Walidacja: leave-one-subject-out</div><div class="panel-caption">Uczymy na wszystkich osobach poza jedną, testujemy na pominiętej i powtarzamy dla każdej osoby.</div>', unsafe_allow_html=True)
    st.caption("Kolejność prac: zdarzenia → prosty detektor → walidacja po osobach → jakość, kanały i ruch → włosy oraz aktywność jako analiza dodatkowa.")
    st.info("Wyniki pojawią się po uruchomieniu detektora. Progi i ustawienia będą dobierane wyłącznie na danych treningowych.")
    metric_cols = st.columns(4)
    for col, label in zip(metric_cols, ["Mediana |błędu|", "Błąd ze znakiem", "Wyniki w ±5 / ±10 s", "Brak poprawnego wskazania"]):
        col.metric(label, "—")
    st.markdown("**Porównanie kanałów na tych samych podziałach osób**")
    st.dataframe(
        {
            "Wariant": ["Kanały krótkie", "Kanały długie", "Krótkie + długie"],
            "Mediana błędu [s]": [None, None, None],
            "Wyniki ±5 s": [None, None, None],
            "Pokrycie prób": [None, None, None],
        },
        use_container_width=True,
        hide_index=True,
    )
    with st.expander("Metoda odniesienia"):
        st.write("Przewidywany koniec = początek wstrzymania + typowy czas trwania wyznaczony wyłącznie na osobach treningowych.")
        st.caption("Detektor musi osiągnąć mniejszy błąd niż ta metoda, aby wnosił wartość.")
    with st.expander("Rozszerzenia po pierwszej wersji"):
        st.markdown("**Jakość i kanały:** porównamy kanały krótkie, długie i połączone; sprawdzimy, czy odrzucenie słabych kanałów zmniejsza błąd oraz dla ilu prób detektor nadal zwraca wynik.")
        st.markdown("**Ruch:** zestawimy wskazania z akcelerometrem, by ocenić, czy detektor nie opiera się głównie na ruchu głowy przy wznowieniu oddychania.")
        st.markdown("**Później:** wpływ włosów i aktywności fizycznej dołączymy jako analizę dodatkową.")
    with st.expander("Przygotowanie sygnału"):
        st.write("Oznacz braki i niepoprawne próbki, przelicz intensywność na OD, rozdziel kanały krótkie i długie oraz zachowaj osobno wolny przebieg i składową pulsacyjną. Nie usuwaj szybkich zmian z góry — mogą zawierać informację o zakończeniu wstrzymania.")

st.markdown('<div class="bottom-note">Cel pierwszej wersji: jak dokładnie można odnaleźć koniec wstrzymania i które kanały pomagają najbardziej?</div>', unsafe_allow_html=True)
