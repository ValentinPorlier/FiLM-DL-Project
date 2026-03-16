"""Thème UI partagé pour FiLM Explorer — injected via st.markdown."""

# ─── Palette ──────────────────────────────────────────────────────────────────
# primary  : #6366f1  (indigo)
# secondary: #06b6d4  (cyan)
# accent   : #f59e0b  (amber)
# success  : #10b981  (emerald)
# bg       : #0d0d1a  (dark navy)
# card     : rgba(255,255,255,0.03)
# ──────────────────────────────────────────────────────────────────────────────

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Base ─────────────────────────────────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Inter', 'Segoe UI', system-ui, sans-serif !important;
}

.stApp {
    background: linear-gradient(160deg, #0d0d1a 0%, #111827 100%) !important;
    color: #e2e8f0 !important;
}

/* ── Masquer éléments Streamlit par défaut ─────────────────────────────────── */
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }

/* ── Sidebar ──────────────────────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: #0f0f1e !important;
    border-right: 1px solid rgba(99,102,241,0.15) !important;
}
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] .stMarkdown p {
    color: #cbd5e1 !important;
}

/* ── Block container ──────────────────────────────────────────────────────── */
.main .block-container {
    padding: 2rem 2.5rem 4rem 2.5rem;
    max-width: 1280px;
}

/* ── Titres ───────────────────────────────────────────────────────────────── */
h1, h2, h3, h4 {
    color: #e2e8f0 !important;
    font-weight: 700 !important;
}

/* ── Boutons ──────────────────────────────────────────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, #6366f1, #4f46e5) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    padding: 0.5rem 1.5rem !important;
    transition: all 0.25s ease !important;
    box-shadow: 0 2px 12px rgba(99,102,241,0.3) !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #818cf8, #6366f1) !important;
    box-shadow: 0 4px 24px rgba(99,102,241,0.45) !important;
    transform: translateY(-1px) !important;
}
.stButton > button:disabled {
    background: rgba(99,102,241,0.2) !important;
    box-shadow: none !important;
    transform: none !important;
    color: #64748b !important;
}

/* ── Barre de progression ─────────────────────────────────────────────────── */
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #6366f1, #06b6d4) !important;
    border-radius: 999px !important;
}

/* ── Divider ──────────────────────────────────────────────────────────────── */
hr {
    border-color: rgba(99,102,241,0.2) !important;
    margin: 2rem 0 !important;
}

/* ── Alertes ──────────────────────────────────────────────────────────────── */
div[data-testid="stAlert"] {
    border-radius: 12px !important;
    border-left-width: 4px !important;
}

/* ── Dataframe ────────────────────────────────────────────────────────────── */
.stDataFrame {
    border-radius: 12px !important;
    overflow: hidden !important;
    border: 1px solid rgba(99,102,241,0.2) !important;
}

/* ── Onglets ──────────────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.03) !important;
    border-radius: 12px !important;
    padding: 4px !important;
    gap: 4px !important;
    border: 1px solid rgba(99,102,241,0.15) !important;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px !important;
    font-weight: 500 !important;
    color: #94a3b8 !important;
}
.stTabs [aria-selected="true"] {
    background: rgba(99,102,241,0.2) !important;
    color: #818cf8 !important;
}

/* ── Métriques ────────────────────────────────────────────────────────────── */
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(99,102,241,0.2) !important;
    border-radius: 12px !important;
    padding: 1rem !important;
}
[data-testid="stMetricValue"] {
    color: #818cf8 !important;
    font-weight: 700 !important;
}

/* ── Code ─────────────────────────────────────────────────────────────────── */
code {
    background: rgba(99,102,241,0.12) !important;
    color: #a5b4fc !important;
    border-radius: 4px !important;
    padding: 0.1em 0.35em !important;
}
pre code {
    background: transparent !important;
    padding: 0 !important;
}

/* ── Composants custom ────────────────────────────────────────────────────── */

/* Carte navigation */
.nav-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(99,102,241,0.2);
    border-radius: 18px;
    padding: 1.75rem 1.5rem;
    text-align: center;
    height: 100%;
    transition: all 0.3s ease;
    cursor: pointer;
}
.nav-card:hover {
    background: rgba(99,102,241,0.08);
    border-color: rgba(99,102,241,0.5);
    transform: translateY(-4px);
    box-shadow: 0 12px 40px rgba(99,102,241,0.15);
}
.nav-icon   { font-size: 2.8rem; margin-bottom: 0.75rem; }
.nav-title  { font-size: 1.15rem; font-weight: 700; color: #e2e8f0; margin-bottom: 0.5rem; }
.nav-desc   { font-size: 0.85rem; color: #94a3b8; line-height: 1.6; }

/* Carte contenu */
.film-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(99,102,241,0.18);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    transition: all 0.3s ease;
}
.film-card:hover {
    border-color: rgba(99,102,241,0.4);
    box-shadow: 0 6px 30px rgba(99,102,241,0.1);
}

/* Badge */
.badge {
    display: inline-block;
    padding: 0.2rem 0.7rem;
    border-radius: 999px;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin: 0 0.15rem;
}
.badge-purple { background: rgba(99,102,241,0.15); color: #818cf8; border: 1px solid rgba(99,102,241,0.3); }
.badge-cyan   { background: rgba(6,182,212,0.15);  color: #22d3ee; border: 1px solid rgba(6,182,212,0.3);  }
.badge-amber  { background: rgba(245,158,11,0.15); color: #fbbf24; border: 1px solid rgba(245,158,11,0.3); }
.badge-green  { background: rgba(16,185,129,0.15); color: #34d399; border: 1px solid rgba(16,185,129,0.3); }
.badge-red    { background: rgba(239,68,68,0.15);  color: #f87171; border: 1px solid rgba(239,68,68,0.3);  }

/* Titre hero */
.hero-title {
    font-size: clamp(2.2rem, 5vw, 3.8rem);
    font-weight: 800;
    background: linear-gradient(135deg, #6366f1 0%, #06b6d4 60%, #10b981 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.1;
    margin-bottom: 0.5rem;
    letter-spacing: -0.02em;
}
.hero-sub {
    font-size: 1.1rem;
    color: #94a3b8;
    font-weight: 400;
    margin-bottom: 1.5rem;
    line-height: 1.6;
}

/* En-tête de section */
.section-header {
    font-size: 1.4rem;
    font-weight: 700;
    color: #e2e8f0;
    margin: 2.5rem 0 1rem 0;
    padding-bottom: 0.6rem;
    border-bottom: 2px solid rgba(99,102,241,0.3);
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* Puce d'étape */
.step-pill {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 28px; height: 28px;
    border-radius: 50%;
    background: linear-gradient(135deg, #6366f1, #4f46e5);
    color: white;
    font-size: 0.8rem;
    font-weight: 700;
    margin-right: 0.5rem;
    flex-shrink: 0;
}

/* Colonne formule */
.formula-box {
    background: rgba(99,102,241,0.07);
    border: 1px solid rgba(99,102,241,0.25);
    border-radius: 14px;
    padding: 1.75rem;
    text-align: center;
}
.formula-text {
    font-size: 1.6rem;
    font-weight: 800;
    color: #a5b4fc;
    letter-spacing: 0.02em;
    margin-bottom: 0.75rem;
}

/* Tag de question/réponse */
.qa-row {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(99,102,241,0.15);
    border-radius: 10px;
    padding: 0.6rem 1rem;
    margin: 0.35rem 0;
    font-size: 0.88rem;
    line-height: 1.5;
}
.qa-q { color: #94a3b8; }
.qa-a { color: #34d399; font-weight: 600; }

/* Tableau de comparaison */
.compare-table th {
    background: rgba(99,102,241,0.15) !important;
    color: #818cf8 !important;
    font-weight: 600 !important;
}
.compare-table td {
    background: rgba(255,255,255,0.02) !important;
    color: #e2e8f0 !important;
}
</style>
"""


def apply_theme():
    """Injecter le thème CSS dans la page Streamlit courante."""
    import streamlit as st
    st.markdown(CSS, unsafe_allow_html=True)
