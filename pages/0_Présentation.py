"""FiLM Explorer — Page d'accueil."""

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

st.title("FiLM Explorer")
st.caption("Feature-wise Linear Modulation — Perez et al., AAAI 2018")
st.divider()

# ─── Introduction ─────────────────────────────────────────────────────────────
col_text, col_math = st.columns([3, 2])

with col_text:
    st.markdown("""
En deep learning, le **conditionnement** consiste à adapter le comportement d'un réseau
à une information contextuelle externe : une question, un style, une classe…

L'approche naïve — concaténer le contexte à l'entrée — ne permet pas au modèle de
sélectionner les features pertinentes et l'oblige à traiter le contexte au mauvais niveau
d'abstraction.

**FiLM** résout ce problème élégamment : un réseau auxiliaire (le *FiLM generator*)
transforme le contexte en paramètres **(γ, β)** qui modifient directement les feature maps
du CNN couche par couche — sans augmenter la dimension des activations.
""")

with col_math:
    st.markdown("**Transformation affine feature-wise**")
    st.latex(
        r"\text{FiLM}(F_{i,c} \mid \gamma_{i,c},\,\beta_{i,c})"
        r"= \gamma_{i,c}\, F_{i,c} + \beta_{i,c}"
    )
    st.markdown("""
| Symbole | Rôle |
|---------|------|
| $F_{i,c}$ | $c$-ième feature map de l'entrée $i$ |
| $\\gamma_{i,c}$ | échelle — peut supprimer une feature ($\\gamma \\approx 0$) |
| $\\beta_{i,c}$ | décalage du point de fonctionnement |
| $\\gamma,\\,\\beta$ | prédits par un réseau auxiliaire selon le contexte |
""")

st.divider()

# ─── Navigation ───────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Sort of CLEVR")
    st.markdown(
        "Dataset 2D Kaggle (formes colorées). "
        "Validation rapide de FiLM avant de passer sur CLEVR 3D. "
        "Entraînement interactif et test visuel."
    )
    st.page_link("pages/1_Sort_of_CLEVR.py", label="Ouvrir →")

with col2:
    st.subheader("CLEVR VQA")
    st.markdown(
        "Benchmark officiel 3D — 700 000 questions de raisonnement compositionnel. "
        "Architecture complète, résultats et étude des composants."
    )
    st.page_link("pages/2_CLEVR_VQA.py", label="Ouvrir →")

with col3:
    st.subheader("Style Transfer")
    st.markdown(
        "FiLM appliqué au transfert de style artistique via "
        "Conditional Instance Normalisation (Ghiasi et al., 2017)."
    )
    st.page_link("pages/3_Style_Transfer.py", label="Ouvrir →")

st.divider()

# ─── Démarche ─────────────────────────────────────────────────────────────────
st.markdown("""
**Notre démarche :**

1. **Sort of CLEVR** — validation sur un dataset 2D simple, entraînement en < 5 min CPU (~94 %)
2. **CLEVR VQA** — montée en complexité sur le benchmark officiel (700 k questions, features ResNet101)
3. **Style Transfer** — généralisation de FiLM à une autre modalité via Conditional Instance Normalisation
""")

st.caption(
    "Implémentation de [Perez et al. 2018](https://arxiv.org/abs/1709.07871) · "
    "[ethanjperez/film](https://github.com/ethanjperez/film)"
)
