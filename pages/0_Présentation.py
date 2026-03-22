"""Page d'accueil."""

import subprocess
import sys
from pathlib import Path
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

st.title("FiLM : Feature Wise Transformations")
st.divider()

# ─── Introduction ─────────────────────────────────────────────────────────────
col_text, col_math = st.columns([4, 3])

with col_text:
    st.markdown("""
En deep learning, le conditionnement consiste à adapter le comportement d'un réseau
à une information contextuelle externe : une question, un style, une classe…

L'approche naïve concaténer le contexte à l'entrée ne permet pas au modèle de
sélectionner les features pertinentes et l'oblige à traiter le contexte au mauvais niveau
d'abstraction.

FiLM résout ce problème élégamment : un réseau auxiliaire (le *FiLM generator*)
transforme le contexte en paramètres (γ, β) qui modifient directement les feature maps
du CNN couche par couche sans augmenter la dimension des activations.
""")

with col_math:
    st.markdown("Comment FiLM modifie les feature maps :")
    st.latex(
        r"\text{FiLM}(F_{i,c} \mid \gamma_{i,c},\,\beta_{i,c})"
        r"= \gamma_{i,c}\, F_{i,c} + \beta_{i,c}"
    )
    st.markdown("""
| Symbole | Rôle |
|---------|------|
| $F_{i,c}$ | la feature map numéro $c$ pour l'image $i$ |
| $\\gamma_{i,c}$ |  amplifie, réduit ou éteint la feature map  |
| $\\beta_{i,c}$ | décale les activations (pousse au-dessus ou en-dessous de zéro) |
| $\\gamma,\\,\\beta$ | produits par le FiLM generator à partir de la question |
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
    st.page_link("pages/1_Sort_of_CLEVR.py", label="Ouvrir")

with col2:
    st.subheader("CLEVR VQA")
    st.markdown(
        "Implémentation de l'architecture présentée dans l'article sur le même dataset 3D — 700 000 questions. "
        "Architecture complète, résultats."
    )
    st.page_link("pages/2_CLEVR_VQA.py", label="Ouvrir")

with col3:
    st.subheader("Style Transfer")
    st.markdown(
        "FiLM appliqué au transfert de style artistique via "
        "Conditional Instance Normalisation (Ghiasi et al., 2017)."
    )
    st.page_link("pages/3_Style_Transfer.py", label="Ouvrir")

st.divider()

# ─── Démarche ─────────────────────────────────────────────────────────────────
st.markdown(r"""
**Notre démarche :**

1. **Sort of CLEVR** — essai sur un dataset simple et léger, entraînement en <5 min  sur le CPU (et atteint ~$94\%$ d'accuracy )
2. **CLEVR VQA** — vrai test sur le dataset utilisé dans l'article implémenté
3. **Style Transfer** — généralisation de FiLM via Conditional Instance Normalisation
""")

st.caption(
    "Implémentation de [Perez et al. 2018](https://arxiv.org/abs/1709.07871) · "
    "[ethanjperez/film](https://github.com/ethanjperez/film) · "
    "Ghiasi et al. (2017) -[Exploring the structure of a real-time, arbitrary neuralartistic stylization network](https://arxiv.org/pdf/1705.06830)"
)

st.divider()

# ─── Installation ──────────────────────────────────────────────────────────────
st.subheader("Installation des dépendances")
st.markdown("Installe tous les packages nécessaires depuis `requirements.txt`.")

if st.button("Installer les dépendances"):
    requirements = ROOT / "requirements.txt"
    with st.spinner("Installation en cours..."):
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", str(requirements)],
            capture_output=True,
            text=True,
        )
    if result.returncode == 0:
        st.success("Dépendances installées avec succès.")
    else:
        st.error("Erreur lors de l'installation.")
        st.code(result.stderr)
