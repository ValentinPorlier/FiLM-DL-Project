"""Streamlit page: Sort of CLEVR — dataset 2D simplifié pour FiLM."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ui_theme import apply_theme

st.set_page_config(
    page_title="Sort of CLEVR — FiLM Explorer",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed",
)
apply_theme()

# ══════════════════════════════════════════════════════════════════════════════
#  Génération d'images Sort of CLEVR 2D
# ══════════════════════════════════════════════════════════════════════════════

COULEURS = {
    "rouge":  "#e74c3c",
    "bleu":   "#3498db",
    "vert":   "#27ae60",
    "orange": "#e67e22",
    "jaune":  "#f1c40f",
    "violet": "#8e44ad",
}
FORMES = ["cercle", "carré"]

# ── Session state ─────────────────────────────────────────────────────────────
for _k, _v in [
    ("soc_active",    False),
    ("soc_queue",     None),
    ("soc_progress",  None),
    ("soc_history",   None),
    ("soc_per_class", None),
    ("soc_fig_preds", None),
]:
    if _k not in st.session_state:
        st.session_state[_k] = _v

def _generer_image(objets: list[dict], taille: int = 224) -> "plt.Figure":
    """Génère une image Sort of CLEVR à partir d'une liste d'objets."""
    fig, ax = plt.subplots(figsize=(3, 3), dpi=taille // 3)
    fig.patch.set_facecolor("#dde1e7")
    ax.set_facecolor("#dde1e7")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.axis("off")

    for obj in objets:
        x, y, r = obj["x"], obj["y"], 0.11
        c = COULEURS[obj["couleur"]]

        # Ombre légère
        if obj["forme"] == "cercle":
            ax.add_patch(plt.Circle((x + 0.012, y - 0.012), r, color="#000", alpha=0.18, zorder=1))
            ax.add_patch(plt.Circle((x, y), r, color=c, zorder=2))
            # Reflet
            ax.add_patch(plt.Circle((x - 0.028, y + 0.028), r * 0.28, color="#fff", alpha=0.35, zorder=3))
        else:
            for dx, dy, alpha in [(0.012, -0.012, 0.18), (0, 0, 1.0)]:
                col = "#000" if alpha < 1 else c
                ax.add_patch(mpatches.FancyBboxPatch(
                    (x - r + dx, y - r + dy), 2 * r, 2 * r,
                    boxstyle="round,pad=0.015",
                    facecolor=col, alpha=alpha,
                    zorder=1 if alpha < 1 else 2,
                ))
            ax.add_patch(mpatches.FancyBboxPatch(
                (x - r + 0.01, y + r * 0.45), r * 0.55, r * 0.22,
                boxstyle="round,pad=0.005",
                facecolor="#fff", alpha=0.3, zorder=3,
            ))

    plt.tight_layout(pad=0)
    return fig


def _generer_scene(seed: int | None = None, n: int = 4) -> tuple[list[dict], "plt.Figure"]:
    """Génère une scène aléatoire et retourne (objets, figure)."""
    rng = np.random.default_rng(seed)
    noms_couleurs = list(COULEURS.keys())
    indices = rng.choice(len(noms_couleurs), n, replace=False)

    objets = []
    for i in range(n):
        forme = rng.choice(FORMES)
        couleur = noms_couleurs[indices[i]]
        # Placement en grille lâche pour éviter les superpositions
        x = 0.2 + (i % 2) * 0.55 + rng.uniform(-0.06, 0.06)
        y = 0.7 - (i // 2) * 0.45 + rng.uniform(-0.06, 0.06)
        objets.append({"forme": forme, "couleur": couleur, "x": x, "y": y})

    return objets, _generer_image(objets)


def _questions_pour(objets: list[dict]) -> list[dict]:
    """Génère des paires (question, réponse) pour une scène."""
    qas = []
    for obj in objets:
        qas.append({
            "q": f"De quelle couleur est le {obj['forme']} ?",
            "r": obj["couleur"],
            "type": "non-relationnel",
        })
        qas.append({
            "q": f"Quelle est la forme de l'objet {obj['couleur']} ?",
            "r": obj["forme"],
            "type": "non-relationnel",
        })
    # Questions relationnelles
    if len(objets) >= 2:
        o1, o2 = objets[0], objets[1]
        direction = "gauche" if o1["x"] < o2["x"] else "droite"
        qas.append({
            "q": f"Quelle forme est à {direction} du {o2['couleur']} ?",
            "r": o1["forme"],
            "type": "relationnel",
        })
        above = "au-dessus" if o1["y"] > o2["y"] else "en-dessous"
        qas.append({
            "q": f"Le {o1['couleur']} est-il {above} du {o2['couleur']} ?",
            "r": "oui",
            "type": "relationnel",
        })
    return qas


# ══════════════════════════════════════════════════════════════════════════════
#  Détection du module sort_of_clevr (du collègue)
# ══════════════════════════════════════════════════════════════════════════════

# Cherche les données dans sortofclevr/ du projet, sinon dans le dossier parent
def _find_data(filename: str) -> Path:
    for candidate in [
        ROOT / "sortofclevr" / filename,
        ROOT.parent / "sortofclevr" / filename,
    ]:
        if candidate.exists():
            return candidate
    return ROOT / "sortofclevr" / filename  # chemin par défaut (affiché dans l'erreur)

TRAIN_H5  = _find_data("data_train.h5")
TRAIN_CSV = _find_data("data_train.csv")
TEST_H5   = _find_data("data_test.h5")
TEST_CSV  = _find_data("data_test.csv")

SOC_MODULE_OK  = False
SOC_DATA_READY = False
try:
    from sortofclevr import (  # type: ignore
        HDF5Dataset, SortOfClevrFiLMModel,
        CLASSES as SOC_CLASSES, NUM_CLASSES as SOC_NUM_CLASSES,
        evaluate as soc_evaluate,
        evaluate_per_class as soc_eval_per_class,
        visualize_predictions as soc_viz_preds,
    )
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    import queue, threading, time
    SOC_MODULE_OK  = True
    SOC_DATA_READY = all(p.exists() for p in [TRAIN_H5, TRAIN_CSV, TEST_H5, TEST_CSV])
except Exception:
    pass

# ── Polling (niveau supérieur — fonctionne quel que soit l'onglet actif) ──────
if SOC_MODULE_OK and st.session_state.soc_active and st.session_state.soc_queue is not None:
    _pq = st.session_state.soc_queue
    _latest = None
    while True:
        try:
            _latest = _pq.get_nowait()
        except queue.Empty:
            break

    if _latest:
        if "error" in _latest:
            st.error(f"Erreur d'entraînement :\n```\n{_latest['error']}\n```")
            st.session_state.soc_active = False
        elif _latest.get("done"):
            st.session_state.soc_history   = _latest["history"]
            st.session_state.soc_per_class  = _latest.get("per_class")
            st.session_state.soc_fig_preds  = _latest.get("fig_preds")
            st.session_state.soc_active     = False
        else:
            st.session_state.soc_progress = _latest

    if st.session_state.soc_active:
        time.sleep(0.5)
        st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
#  UI
# ══════════════════════════════════════════════════════════════════════════════

# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding:2rem 0 0.5rem 0;">
    <div class="hero-title" style="font-size:clamp(1.8rem,4vw,3rem);">
         Sort of CLEVR
    </div>
    <p class="hero-sub">
        Raisonnement visuel 2D — le point de départ avant CLEVR
    </p>
    <span class="badge badge-green">2D</span>
    <span class="badge badge-cyan">Formes simples</span>
    <span class="badge badge-purple">Questions relationnelles</span>
    <span class="badge badge-amber">Santoro et al. 2017</span>
</div>
""", unsafe_allow_html=True)

st.divider()

# ── Onglets ────────────────────────────────────────────────────────────────────
tab_apercu, tab_demo, tab_archi, tab_train, tab_resultats = st.tabs([
    " Présentation",
    " Démo interactive",
    " Architecture",
    " Entraînement",
    " Résultats",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Présentation
# ══════════════════════════════════════════════════════════════════════════════
with tab_apercu:
    c1, c2 = st.columns([3, 2], gap="large")

    with c1:
        st.markdown("""
        <div class="section-header">Pourquoi Sort of CLEVR ?</div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="film-card">
        <p style="color:#cbd5e1; line-height:1.8; font-size:0.9rem;">
        Avant de s'attaquer aux 18 Go de CLEVR et à ses images 3D photoréalistes,
        nous avons d'abord validé l'architecture FiLM sur un dataset bien plus simple :
        <strong style="color:#e2e8f0;">Sort of CLEVR</strong>.
        </p>
        <p style="color:#cbd5e1; line-height:1.8; font-size:0.9rem;">
        Introduit par Santoro et al. (2017) avec les <em>Relation Networks</em>, ce dataset
        2D contient des <strong>formes colorées</strong> (cercles et carrés) sur un fond uni.
        Les questions sont de deux types :
        </p>
        <ul style="color:#cbd5e1; font-size:0.88rem; line-height:2;">
            <li><strong style="color:#818cf8;">Non-relationnelles</strong> — portent sur un seul objet :
                <em>« Quelle est la couleur du carré ? »</em></li>
            <li><strong style="color:#22d3ee;">Relationnelles</strong> — comparent deux objets :
                <em>« Quelle forme est à droite du cercle bleu ? »</em></li>
        </ul>
        <p style="color:#cbd5e1; line-height:1.8; font-size:0.9rem;">
        Ce cadre léger nous a permis de tester rapidement la boucle complète
        (extraction de features → GRU → FiLM → classifieur) avant de passer
        à l'échelle de CLEVR.
        </p>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="section-header">Caractéristiques</div>
        """, unsafe_allow_html=True)

        items = [
            ("", "6 couleurs", "Rouge, bleu, vert, orange, jaune, violet"),
            ("⬛", "2 formes", "Cercle et carré"),
            ("", "2 types de questions", "Non-relationnelles & relationnelles"),
            ("", "Images 128×128", "Fond gris, objets avec ombre"),
            ("", "Dataset léger", "Quelques Mo, générable à la volée"),
            ("", "Même architecture", "FiLM identique à CLEVR, juste plus petit"),
        ]
        for icon, titre, desc in items:
            st.markdown(f"""
            <div class="film-card" style="padding:0.85rem 1rem; margin-bottom:0.5rem;">
                <div style="display:flex; align-items:flex-start; gap:0.75rem;">
                    <span style="font-size:1.2rem; flex-shrink:0;">{icon}</span>
                    <div>
                        <div style="font-weight:600; color:#e2e8f0; font-size:0.85rem;">{titre}</div>
                        <div style="color:#94a3b8; font-size:0.78rem;">{desc}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.divider()

    # Comparaison Sort of CLEVR vs CLEVR
    st.markdown('<div class="section-header">Sort of CLEVR vs CLEVR</div>', unsafe_allow_html=True)

    comp_data = {
        "Critère":          ["Images",      "Résolution", "Questions", "Types de réponse", "Features",        "Taille dataset", "Entraînement"],
        "Sort of CLEVR":    ["2D synthèse", "128×128",    "~15 k",     "Forme / couleur",  "CNN léger / raw",  "< 50 Mo",        "~5 min GPU"],
        "CLEVR":            ["3D render",   "480×320",    "700 k",     "28 classes",       "ResNet101 1024ch", "18 Go",          "~10 h GPU"],
    }

    import pandas as pd
    df = pd.DataFrame(comp_data)
    st.dataframe(df, width="stretch", hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Démo interactive
# ══════════════════════════════════════════════════════════════════════════════
with tab_demo:
    st.markdown('<div class="section-header"> Exemples de scènes générées</div>', unsafe_allow_html=True)

    st.markdown("""
    <p style="color:#94a3b8; font-size:0.88rem; margin-bottom:1rem;">
    Chaque scène est générée aléatoirement : 4 objets (cercles et carrés colorés)
    placés sur un fond gris. Le modèle FiLM doit répondre à des questions sur ces scènes.
    </p>
    """, unsafe_allow_html=True)

    seed_offset = st.slider("Graine aléatoire (change les scènes)", 0, 100, 0, key="demo_seed")

    cols = st.columns(4, gap="medium")
    all_scenes = []
    for i, col in enumerate(cols):
        seed = i * 7 + seed_offset * 13
        objets, fig = _generer_scene(seed=seed, n=4)
        all_scenes.append((objets, fig))
        with col:
            st.pyplot(fig, width="stretch")
            plt.close(fig)
            st.markdown(f"""
            <div style="text-align:center; font-size:0.75rem; color:#64748b; margin-top:0.25rem;">
                Scène {i + 1}
            </div>
            """, unsafe_allow_html=True)

    # Questions/réponses pour la scène sélectionnée
    st.markdown('<div class="section-header"> Questions & Réponses</div>', unsafe_allow_html=True)
    scene_idx = st.radio(
        "Scène à inspecter",
        [f"Scène {i+1}" for i in range(4)],
        horizontal=True,
        key="scene_select",
    )
    idx = int(scene_idx.split()[-1]) - 1
    objets_sel, _ = all_scenes[idx]
    qas = _questions_pour(objets_sel)

    col_nr, col_r = st.columns(2, gap="large")
    with col_nr:
        st.markdown("**Questions non-relationnelles**")
        for qa in qas:
            if qa["type"] == "non-relationnel":
                st.markdown(f"""
                <div class="qa-row">
                    <span class="qa-q">Q : {qa['q']}</span><br>
                    <span class="qa-a">R : {qa['r']}</span>
                </div>
                """, unsafe_allow_html=True)

    with col_r:
        st.markdown("**Questions relationnelles**")
        for qa in qas:
            if qa["type"] == "relationnel":
                st.markdown(f"""
                <div class="qa-row">
                    <span class="qa-q">Q : {qa['q']}</span><br>
                    <span class="qa-a">R : {qa['r']}</span>
                </div>
                """, unsafe_allow_html=True)

    st.divider()

    # Palette de couleurs
    st.markdown('<div class="section-header"> Palette Sort of CLEVR</div>', unsafe_allow_html=True)
    palette_cols = st.columns(len(COULEURS))
    for col, (nom, hex_) in zip(palette_cols, COULEURS.items()):
        with col:
            st.markdown(f"""
            <div style="text-align:center;">
                <div style="width:48px; height:48px; border-radius:50%;
                     background:{hex_}; margin:0 auto 0.4rem auto;
                     box-shadow:0 4px 12px {hex_}55;
                     border:2px solid rgba(255,255,255,0.15);"></div>
                <div style="font-size:0.78rem; color:#94a3b8; font-weight:500;">{nom.capitalize()}</div>
                <div style="font-size:0.68rem; color:#475569;">{hex_}</div>
            </div>
            """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Architecture
# ══════════════════════════════════════════════════════════════════════════════
with tab_archi:
    st.markdown('<div class="section-header"> Architecture pour Sort of CLEVR</div>', unsafe_allow_html=True)

    c_film, c_diff = st.columns([1, 1], gap="large")

    with c_film:
        st.markdown("**Pipeline complet**")
        st.markdown("""
        <div class="film-card" style="font-size:0.82rem; font-family:'Fira Code','Courier New',monospace;
             line-height:2.1; color:#e2e8f0;">
            <span style="color:#94a3b8;">Question</span>
            → Embedding(64) → GRU(256) → Linear
            → <span style="color:#6366f1; font-weight:700;">(γₖ, βₖ)</span><br>

            <span style="color:#94a3b8;">Image 128×128</span>
            → <span style="color:#22d3ee;">CNN léger</span> → (64, 16, 16)<br>

            &emsp;→ 4×
            <span style="color:#f59e0b; font-weight:700;">FiLMedResBlock</span><br>
            &emsp;&emsp;Conv3×3 → BN →
            <span style="color:#6366f1; font-weight:700;">FiLM(γ,β)</span>
            → ReLU + résidu<br>

            &emsp;→ [feat ++ coord] → Conv1×1(→128) → AvgPool<br>
            &emsp;→ FC(128→256) → ReLU → FC(256→<span style="color:#10b981; font-weight:700;">2+6</span>)
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="film-card" style="margin-top:1rem;">
            <div style="font-size:0.82rem; color:#94a3b8; line-height:1.8;">
                <b style="color:#e2e8f0;">Sorties</b><br>
                • <span style="color:#818cf8;">2 formes</span>
                  pour les questions sur la forme (cercle / carré)<br>
                • <span style="color:#22d3ee;">6 couleurs</span>
                  pour les questions sur la couleur<br>
                • Ou <b>une seule tête</b> avec toutes les réponses possibles
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c_diff:
        st.markdown("**Différences clés vs CLEVR**")
        diffs = [
            ("CNN léger", "ResNet101 1024ch",       "Extracteur de features simplifié"),
            ("Coord maps", "Coord maps identiques",  "Même mécanisme de position 2D"),
            ("FiLM ×4",   "FiLM ×4 identique",      "Même nombre de blocs"),
            ("AvgPool",   "Flatten 25 088",           "Classifieur bien plus petit"),
            ("FC(128→256)","FC(25088→1024)",          "Paramètres réduits × 200"),
            ("~50 k params","~28 M params",           "Beaucoup plus rapide à entraîner"),
        ]
        for soc, clevr, note in diffs:
            st.markdown(f"""
            <div class="film-card" style="padding:0.7rem 1rem; margin-bottom:0.4rem;">
                <div style="display:flex; justify-content:space-between; align-items:center; gap:0.5rem;">
                    <span style="color:#818cf8; font-size:0.8rem; font-weight:600; flex:1;">{soc}</span>
                    <span style="color:#475569; font-size:0.75rem;">vs</span>
                    <span style="color:#22d3ee; font-size:0.8rem; flex:1; text-align:right;">{clevr}</span>
                </div>
                <div style="color:#64748b; font-size:0.72rem; margin-top:0.2rem;">{note}</div>
            </div>
            """, unsafe_allow_html=True)

    st.divider()

    # Schéma FiLMedResBlock
    st.markdown('<div class="section-header"> FiLMedResBlock (identique dans les deux datasets)</div>',
                unsafe_allow_html=True)

    st.markdown("""
    <div class="film-card" style="max-width:700px; margin:0 auto;">
        <div style="display:flex; flex-direction:column; align-items:center;
                    gap:0; font-size:0.82rem; font-family:'Fira Code',monospace;
                    color:#e2e8f0; line-height:1;">

            <div style="padding:0.4rem 1.2rem; background:rgba(6,182,212,0.15);
                 border:1px solid rgba(6,182,212,0.3); border-radius:8px;">
                x  (B, C, H, W)
            </div>
            <div style="color:#475569; margin:0.2rem 0;">↓ + coord maps</div>
            <div style="padding:0.4rem 1.2rem; background:rgba(99,102,241,0.1);
                 border:1px solid rgba(99,102,241,0.25); border-radius:8px;">
                Conv 1×1 → ReLU &nbsp;&nbsp;<span style="color:#64748b;">(résidu)</span>
            </div>
            <div style="color:#475569; margin:0.2rem 0;">↓</div>
            <div style="padding:0.4rem 1.2rem; background:rgba(99,102,241,0.1);
                 border:1px solid rgba(99,102,241,0.25); border-radius:8px;">
                Conv 3×3
            </div>
            <div style="color:#475569; margin:0.2rem 0;">↓</div>
            <div style="padding:0.4rem 1.2rem; background:rgba(99,102,241,0.1);
                 border:1px solid rgba(99,102,241,0.25); border-radius:8px;">
                BatchNorm (affine=False)
            </div>
            <div style="color:#475569; margin:0.2rem 0;">↓</div>
            <div style="padding:0.5rem 1.5rem; background:rgba(99,102,241,0.25);
                 border:1px solid rgba(99,102,241,0.5); border-radius:8px;
                 font-weight:700; color:#818cf8;">
                FiLM : γ · x + β
            </div>
            <div style="color:#475569; margin:0.2rem 0;">↓</div>
            <div style="padding:0.4rem 1.2rem; background:rgba(99,102,241,0.1);
                 border:1px solid rgba(99,102,241,0.25); border-radius:8px;">
                ReLU
            </div>
            <div style="color:#475569; margin:0.2rem 0;">↓ + résidu</div>
            <div style="padding:0.4rem 1.2rem; background:rgba(16,185,129,0.15);
                 border:1px solid rgba(16,185,129,0.3); border-radius:8px;">
                x'  (B, C, H, W)
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Entraînement
# ══════════════════════════════════════════════════════════════════════════════
with tab_train:
    if not SOC_MODULE_OK:
        st.error("Impossible d'importer le module `sortofclevr`. "
                 "Vérifiez que le dossier `sortofclevr/` est présent à la racine du projet.")
    elif not SOC_DATA_READY:
        st.markdown("""
        <div class="film-card" style="border-color:rgba(245,158,11,0.4); text-align:center; padding:2.5rem;">
            <div style="font-size:2.5rem; margin-bottom:1rem;"></div>
            <div style="font-weight:700; color:#fbbf24; font-size:1.1rem; margin-bottom:0.75rem;">
                Données introuvables
            </div>
            <div style="color:#94a3b8; font-size:0.88rem; line-height:1.7;">
                Placez les fichiers suivants dans le dossier <code>sortofclevr/</code> :<br><br>
                <code>data_train.h5</code> &nbsp;·&nbsp; <code>data_train.csv</code><br>
                <code>data_test.h5</code> &nbsp;·&nbsp; <code>data_test.csv</code>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # ── Configuration ─────────────────────────────────────────────────────
        st.markdown('<div class="section-header"> Configuration</div>', unsafe_allow_html=True)
        sb = st.columns(3, gap="large")
        with sb[0]:
            n_epochs = st.slider("Époques", 1, 50, 10)
        with sb[1]:
            batch_sz = st.select_slider("Batch size", [32, 64, 128, 256, 512], value=128)
        with sb[2]:
            lr = st.select_slider("Learning rate", [1e-4, 5e-4, 1e-3, 3e-3], value=1e-3,
                                  format_func=lambda x: f"{x:.0e}")

        # ── Statut des données ─────────────────────────────────────────────────
        st.markdown("""
        <div style="display:flex; align-items:center; gap:0.5rem; margin-bottom:1.5rem;">
            <span style="color:#34d399; font-size:1.2rem;"></span>
            <span style="color:#34d399; font-weight:600;">Données détectées</span>
            <span style="color:#475569; font-size:0.85rem;">— prêt à entraîner</span>
        </div>
        """, unsafe_allow_html=True)

        start_btn = st.button(" Lancer l'entraînement",
                              disabled=st.session_state.soc_active)

        if start_btn and not st.session_state.soc_active:
            pq: queue.Queue = queue.Queue()
            st.session_state.soc_queue    = pq
            st.session_state.soc_active   = True
            st.session_state.soc_progress = None
            st.session_state.soc_history  = None

            def _run_soc(q: queue.Queue) -> None:
                try:
                    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    _pin    = torch.cuda.is_available()

                    train_ds = HDF5Dataset(str(TRAIN_H5), "data_train", str(TRAIN_CSV))
                    test_ds  = HDF5Dataset(str(TEST_H5),  "data_test",  str(TEST_CSV))
                    train_loader = DataLoader(train_ds, batch_size=batch_sz, shuffle=True,
                                             num_workers=0, pin_memory=_pin)
                    test_loader  = DataLoader(test_ds,  batch_size=batch_sz, shuffle=False,
                                             num_workers=0, pin_memory=_pin)

                    model     = SortOfClevrFiLMModel(num_answers=SOC_NUM_CLASSES).to(_device)
                    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                    criterion = nn.CrossEntropyLoss()
                    history   = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

                    n_batches = len(train_loader)
                    for epoch in range(n_epochs):
                        model.train()
                        run_loss, correct, total = 0.0, 0, 0
                        print(f"[Sort of CLEVR] Époque {epoch+1}/{n_epochs} — début", flush=True)
                        for batch_i, (_, imgs, labels, encs) in enumerate(train_loader):
                            imgs, encs, labels = imgs.to(_device), encs.to(_device), labels.to(_device)
                            optimizer.zero_grad()
                            out  = model(imgs, encs)
                            loss = criterion(out, labels)
                            loss.backward()
                            optimizer.step()
                            run_loss += loss.item()
                            total    += labels.size(0)
                            correct  += out.argmax(1).eq(labels).sum().item()
                            if (batch_i + 1) % 50 == 0:
                                print(f"  batch {batch_i+1}/{n_batches} | "
                                      f"loss={loss.item():.4f} | acc={correct/total:.1%}", flush=True)

                        t_loss = run_loss / len(train_loader)
                        t_acc  = correct / total
                        v_loss, v_acc = soc_evaluate(model, test_loader, criterion, _device)

                        history["train_loss"].append(t_loss)
                        history["train_acc"].append(t_acc)
                        history["val_loss"].append(v_loss)
                        history["val_acc"].append(v_acc)
                        print(f"[Sort of CLEVR] Époque {epoch+1}/{n_epochs} | "
                              f"Train Loss: {t_loss:.4f} | Train Acc: {t_acc:.2%} | "
                              f"Val Loss: {v_loss:.4f} | Val Acc: {v_acc:.2%}")
                        q.put({"epoch": epoch + 1, "num_epochs": n_epochs,
                               "train_loss": t_loss, "train_acc": t_acc,
                               "val_loss": v_loss, "val_acc": v_acc})

                    per_class = soc_eval_per_class(model, test_loader, _device)
                    fig_preds = soc_viz_preds(model, test_loader, _device, num_images=6)
                    q.put({"done": True, "history": history,
                           "per_class": per_class, "fig_preds": fig_preds})

                except Exception:
                    import traceback
                    q.put({"error": traceback.format_exc(), "done": True})

            threading.Thread(target=_run_soc, args=(pq,), daemon=True).start()

        # ── Progression ────────────────────────────────────────────────────────
        prog = st.session_state.soc_progress
        if prog and not st.session_state.soc_history:
            ep, tot = prog["epoch"], prog["num_epochs"]
            st.progress(ep / tot, text=f"Époque {ep}/{tot}")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Train Loss", f"{prog['train_loss']:.4f}")
            c2.metric("Train Acc",  f"{prog['train_acc']:.1%}")
            c3.metric("Val Loss",   f"{prog['val_loss']:.4f}")
            c4.metric("Val Acc",    f"{prog['val_acc']:.1%}")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Résultats
# ══════════════════════════════════════════════════════════════════════════════
with tab_resultats:
    # ── Résultats de l'entraînement si disponibles ─────────────────────────────
    if st.session_state.soc_history:
        history = st.session_state.soc_history
        st.success(" Entraînement terminé !")
        st.progress(1.0)
        best_val = max(history["val_acc"])
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Meilleure Val Acc", f"{best_val:.1%}")
        c2.metric("Val Acc finale",    f"{history['val_acc'][-1]:.1%}")
        c3.metric("Train Acc finale",  f"{history['train_acc'][-1]:.1%}")
        c4.metric("Époques",           len(history["train_loss"]))

        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            epochs = list(range(1, len(history["train_loss"]) + 1))
            fig = make_subplots(rows=1, cols=2, subplot_titles=("Loss", "Accuracy"))
            for name, y, col in [("Train Loss", history["train_loss"], 1),
                                  ("Val Loss",   history["val_loss"],   1),
                                  ("Train Acc",  history["train_acc"],  2),
                                  ("Val Acc",    history["val_acc"],    2)]:
                fig.add_trace(go.Scatter(x=epochs, y=y, name=name, mode="lines+markers"), row=1, col=col)
            fig.update_layout(template="plotly_dark", height=320,
                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.02)")
            st.plotly_chart(fig, width="stretch")
        except ImportError:
            st.line_chart({"Train Acc": history["train_acc"], "Val Acc": history["val_acc"]})

        if st.session_state.soc_per_class:
            st.markdown('<div class="section-header">Accuracy par classe</div>', unsafe_allow_html=True)
            import pandas as pd
            per = st.session_state.soc_per_class
            df_pc = pd.DataFrame([{"Classe": c, "Accuracy": f"{v:.1%}"} for c, v in
                                   sorted(per.items(), key=lambda x: -x[1])])
            st.dataframe(df_pc, width="stretch", hide_index=True)

        if st.session_state.soc_fig_preds:
            st.markdown('<div class="section-header">Prédictions exemples</div>', unsafe_allow_html=True)
            st.pyplot(st.session_state.soc_fig_preds)

        if st.button(" Réinitialiser"):
            for k in ["soc_active", "soc_queue", "soc_progress", "soc_history", "soc_per_class", "soc_fig_preds"]:
                st.session_state[k] = False if k == "soc_active" else None
            st.rerun()

        st.divider()

    st.markdown('<div class="section-header"> Résultats attendus</div>', unsafe_allow_html=True)

    c_res, c_context = st.columns([3, 2], gap="large")

    with c_res:
        import pandas as pd

        res_data = {
            "Modèle": [
                "MLP seul",
                "CNN sans FiLM",
                "FiLM (non-rel.)",
                "FiLM (rel.)",
                "FiLM (global)",
                "Relation Network",
            ],
            "Acc. non-rel.": ["~76 %", "~84 %", "~96 %", "~88 %", "~94 %", "~95 %"],
            "Acc. rel.":     ["~63 %", "~67 %", "~81 %", "~95 %", "~88 %", "~99 %"],
            "Acc. globale":  ["~70 %", "~76 %", "~89 %", "~92 %", "~91 %", "~97 %"],
        }
        df_res = pd.DataFrame(res_data)
        st.dataframe(df_res, width="stretch", hide_index=True)

        st.markdown("""
        <div style="color:#64748b; font-size:0.75rem; margin-top:0.5rem;">
            * Résultats indicatifs basés sur les benchmarks de la littérature (Santoro et al. 2017,
            Perez et al. 2018). Les chiffres exacts dépendent de l'architecture et du dataset généré.
        </div>
        """, unsafe_allow_html=True)

    with c_context:
        st.markdown('<div class="section-header" style="font-size:1.1rem;"> Interprétation</div>',
                    unsafe_allow_html=True)

        insights = [
            ("", "FiLM excelle sur les questions relationnelles", "badge-cyan"),
            ("", "Entraînement 100× plus rapide que sur CLEVR", "badge-green"),
            ("", "Permet de tester toutes les ablations en minutes", "badge-purple"),
            ("", "Courbe d'apprentissage rapide : ~90 % en 10 epochs", "badge-amber"),
        ]
        for icon, text, badge_cls in insights:
            st.markdown(f"""
            <div class="film-card" style="padding:0.75rem 1rem; margin-bottom:0.5rem;
                 display:flex; align-items:flex-start; gap:0.75rem;">
                <span style="font-size:1.2rem; flex-shrink:0;">{icon}</span>
                <span style="color:#cbd5e1; font-size:0.83rem; line-height:1.6;">{text}</span>
            </div>
            """, unsafe_allow_html=True)

    st.divider()

    # Ablations suggérées
    st.markdown('<div class="section-header"> Ablations suggérées</div>', unsafe_allow_html=True)

    abl_data = {
        "Configuration":   ["FiLM baseline", "Sans BN", "γ fixe = 1", "β fixe = 0", "γ et β fixes", "2 blocs", "6 blocs"],
        "BN":              ["", "", "", "", "", "", ""],
        "γ libre":         ["", "", "", "", "", "", ""],
        "β libre":         ["", "", "", "", "", "", ""],
        "Blocs":           ["4", "4", "4", "4", "4", "2", "6"],
        "Acc. attendue":   ["~91 %", "~85 %", "~88 %", "~89 %", "~82 %", "~87 %", "~93 %"],
    }
    df_abl = pd.DataFrame(abl_data)
    st.dataframe(df_abl, width="stretch", hide_index=True)
