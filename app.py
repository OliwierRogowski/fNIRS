from __future__ import annotations

import streamlit as st


st.set_page_config(
    page_title="Uczestnicy · fNIRS",
    page_icon="〰",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');

    :root {
        --ink: #14232d;
        --muted: #71808a;
        --teal: #078f80;
        --teal-dark: #08786e;
        --line: #e5e9e8;
        --paper: #f7f8f6;
        --white: #ffffff;
    }
    .stApp { background: var(--paper); color: var(--ink); font-family: 'DM Sans', sans-serif; }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stToolbar"] { right: 1rem; }
    section[data-testid="stSidebar"] {
        background: #17252e; border-right: 1px solid #263640;
    }
    section[data-testid="stSidebar"] > div { padding: 1.35rem 1rem 1rem; }
    section[data-testid="stSidebar"] * { color: #eef5f5; }
    .brand { display:flex; align-items:center; gap:12px; padding: 6px 8px 34px; }
    .brand-mark { color:#22b5a3; font-size:31px; line-height:1; font-weight:700; }
    .brand-name { font-family:'Manrope',sans-serif; font-size:21px; font-weight:800; letter-spacing:-.7px; }
    .brand-sub { color:#a9b6bb; font-size:12px; margin-top:1px; }
    .nav-label { color:#93a3aa !important; font-size:10px; text-transform:uppercase; letter-spacing:1.5px; margin: 0 10px 10px; }
    .nav-item { display:flex; gap:13px; align-items:center; padding:12px 13px; margin:4px 0; border-radius:10px; font-weight:600; font-size:14px; color:#d8e1e4; }
    .nav-item.active { color:white; background:linear-gradient(115deg,#078f80,#087a70); box-shadow:0 6px 18px #00685a42; }
    .nav-icon { width:21px; text-align:center; font-size:17px; }
    .sidebar-footer { position:fixed; bottom:22px; color:#9aaab0; font-size:12px; border-top:1px solid #405058; padding:14px 7px 0; width:calc(15.5rem - 2rem); }
    .main .block-container { padding-top: 2.1rem; padding-bottom: 2.5rem; max-width: 1500px; }
    .eyebrow { color:var(--teal); font-weight:700; font-size:11px; text-transform:uppercase; letter-spacing:1.45px; margin-bottom:7px; }
    h1 { font-family:'Manrope',sans-serif !important; font-size:34px !important; font-weight:800 !important; letter-spacing:-1.2px !important; color:var(--ink) !important; margin:0 0 5px !important; }
    .subtitle { color:var(--muted); font-size:15px; margin-bottom:27px; }
    .topline { display:flex; justify-content:space-between; align-items:flex-start; gap:20px; }
    .quiet-pill { display:inline-flex; align-items:center; gap:8px; border:1px solid #dce6e3; background:#eef7f4; color:#247c70; padding:8px 12px; border-radius:999px; font-size:12px; font-weight:600; margin-top:7px; }
    .quiet-dot { width:7px; height:7px; border-radius:50%; background:#18a58f; }
    .summary-row { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:14px; margin: 2px 0 22px; }
    .summary-card { background:white; border:1px solid var(--line); border-radius:13px; padding:16px 18px; min-height:98px; }
    .summary-label { color:#77858c; font-size:12px; margin-bottom:10px; }
    .summary-value { font-family:'Manrope',sans-serif; color:var(--ink); font-weight:800; font-size:21px; }
    .summary-note { color:#96a0a4; font-size:11px; margin-left:7px; font-weight:500; }
    .panel { background:white; border:1px solid var(--line); border-radius:14px; padding:20px 21px; }
    .panel-head { display:flex; justify-content:space-between; align-items:center; margin-bottom:15px; }
    .panel-title { font-family:'Manrope',sans-serif; font-size:17px; font-weight:800; letter-spacing:-.3px; }
    .panel-caption { color:#879399; font-size:12px; }
    .search-hint { color:#879399; font-size:12px; margin-top:10px; }
    div[data-testid="stTextInput"] input { border:1px solid #dce2e1; border-radius:9px; background:#fbfcfb; height:45px; }
    div[data-testid="stTextInput"] input:focus { border-color:#12a18e; box-shadow:0 0 0 2px #12a18e1c; }
    div[data-testid="stSelectbox"] div[data-baseweb="select"] > div { border-color:#dce2e1; border-radius:9px; min-height:45px; }
    div[data-testid="stButton"] button { border:1px solid #d7e2df; border-radius:9px; color:#52636b; background:white; font-weight:600; min-height:42px; }
    div[data-testid="stButton"] button:hover { border-color:#0b9b89; color:#087c70; }
    .empty-state { text-align:center; padding:56px 18px 58px; border:1px dashed #d8e1df; border-radius:12px; background:#fcfdfc; margin-top:8px; }
    .empty-icon { width:52px; height:52px; margin:0 auto 15px; border-radius:15px; display:flex; align-items:center; justify-content:center; background:#eaf6f3; color:#078f80; font-size:24px; }
    .empty-title { font-family:'Manrope',sans-serif; color:#263740; font-size:16px; font-weight:800; margin-bottom:6px; }
    .empty-copy { color:#829097; font-size:13px; max-width:390px; line-height:1.6; margin:0 auto; }
    .bottom-note { color:#929da1; font-size:11px; text-align:center; margin-top:23px; }
    @media (max-width: 800px) {
       .summary-row { grid-template-columns:1fr; }
       .topline { display:block; }
       .main .block-container { padding-top:1.3rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown(
        '<div class="brand"><div class="brand-mark">〰</div><div><div class="brand-name">fNIRS</div><div class="brand-sub">Panel badania</div></div></div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="nav-label">Przestrzeń badawcza</div>', unsafe_allow_html=True)
    st.markdown('<div class="nav-item active"><span class="nav-icon">♧</span>Uczestnicy</div>', unsafe_allow_html=True)
    st.markdown('<div class="nav-item"><span class="nav-icon">▣</span>Nagrania</div>', unsafe_allow_html=True)
    st.markdown('<div class="nav-item"><span class="nav-icon">▥</span>Analiza</div>', unsafe_allow_html=True)
    st.markdown('<div class="nav-item"><span class="nav-icon">▤</span>Protokół</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-footer">Badanie pilotażowe</div>', unsafe_allow_html=True)

st.markdown(
    '<div class="topline"><div><div class="eyebrow">Panel badania · fNIRS</div><h1>Uczestnicy</h1><div class="subtitle">Wyszukuj i organizuj uczestników badania.</div></div><div class="quiet-pill"><span class="quiet-dot"></span>Panel gotowy do konfiguracji</div></div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="summary-row">'
    '<div class="summary-card"><div class="summary-label">Wszyscy uczestnicy</div><div class="summary-value">— <span class="summary-note">brak danych</span></div></div>'
    '<div class="summary-card"><div class="summary-label">Sesje badawcze</div><div class="summary-value">— <span class="summary-note">brak danych</span></div></div>'
    '<div class="summary-card"><div class="summary-label">Ostatnia aktywność</div><div class="summary-value">— <span class="summary-note">brak danych</span></div></div>'
    '</div>',
    unsafe_allow_html=True,
)

st.markdown('<div class="panel">', unsafe_allow_html=True)
st.markdown(
    '<div class="panel-head"><div><div class="panel-title">Lista uczestników</div><div class="panel-caption">Znajdź osobę po identyfikatorze lub nazwie grupy</div></div><div class="panel-caption">Katalog badania</div></div>',
    unsafe_allow_html=True,
)
search_col, filter_col, action_col = st.columns([5, 2, 1.25], vertical_alignment="bottom")
with search_col:
    query = st.text_input(
        "Szukaj uczestników",
        placeholder="⌕   Wpisz ID uczestnika lub nazwę grupy…",
        label_visibility="collapsed",
        key="participant_search",
    )
with filter_col:
    st.selectbox("Status", ["Wszystkie statusy", "Aktywni", "Zakończone"], label_visibility="collapsed")
with action_col:
    st.button("＋  Dodaj osobę", use_container_width=True, disabled=True)
st.markdown('<div class="search-hint">Wyszukiwanie będzie działać na danych uczestników po ich podłączeniu.</div>', unsafe_allow_html=True)
participants: list[dict[str, str]] = []
filtered_participants = [
    person
    for person in participants
    if query.casefold() in f"{person.get('id', '')} {person.get('group', '')}".casefold()
]
if filtered_participants:
    st.dataframe(filtered_participants, use_container_width=True, hide_index=True)
else:
    if query.strip():
        empty_title = "Brak wyników wyszukiwania"
        empty_copy = "Nie znaleziono uczestnika pasującego do podanej frazy. Po podłączeniu danych wyniki pojawią się tutaj."
    else:
        empty_title = "Brak uczestników do wyświetlenia"
        empty_copy = "Gdy dodasz dane badania, lista pojawi się tutaj. Możesz wtedy wyszukiwać uczestników po identyfikatorze lub grupie."
    st.markdown(
        f'<div class="empty-state"><div class="empty-icon">⌕</div><div class="empty-title">{empty_title}</div><div class="empty-copy">{empty_copy}</div></div>',
        unsafe_allow_html=True,
    )
st.markdown('</div>', unsafe_allow_html=True)
st.markdown('<div class="bottom-note">Wyniki testów i dane sygnałowe będą dostępne po podłączeniu źródła danych.</div>', unsafe_allow_html=True)
