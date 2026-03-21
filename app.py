"""FiLM Explorer — Page d'accueil."""

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

st.set_page_config(page_title="FiLM Explorer", layout="wide")

st.title("FiLM Explorer")
st.caption("Feature-wise Linear Modulation — Perez et al., AAAI 2018")

st.divider()

# ─── Navigation ───────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Sort of CLEVR")
    st.markdown(
        "Dataset Kaggle 2D (formes colorées) — étape intermédiaire avant CLEVR 3D. "
        "Entraînement interactif et test visuel du modèle FiLM."
    )
    st.page_link("pages/1_sort_of_clevr.py", label="Ouvrir")

with col2:
    st.subheader("CLEVR VQA")
    st.markdown(
        "Résultats FiLM entraîné sur 700 000 questions CLEVR 3D "
        "avec features ResNet101 pré-extraites."
    )
    st.page_link("pages/2_clevr_vqa.py", label="Ouvrir")

with col3:
    st.subheader("Style Transfer")
    st.markdown(
        "Transfert de style conditionnel via Conditional Instance Normalisation (CIN) — "
        "le même mécanisme que FiLM appliqué au style artistique (Dumoulin et al., 2017)."
    )
    st.page_link("pages/3_style_transfer.py", label="Ouvrir")

st.divider()

# ─── FiLM — Principe ──────────────────────────────────────────────────────────
col_text, col_math = st.columns([3, 2])

with col_text:
    st.subheader("Qu'est-ce que FiLM ?")
    st.markdown("""
**FiLM** (Feature-wise Linear Modulation) est un mécanisme de conditionnement
d'un CNN visuel sur une modalité externe — ici une question en langage naturel.

L'idée clé : au lieu de paramètres fixes, un encodeur de question
(**GRU → Linear**) prédit dynamiquement les paramètres d'échelle et de biais
**(γ, β)** injectés après chaque Batch Normalisation du CNN.
Cela permet au pipeline visuel d'être **modulé par la question** sans partage
de paramètres entre couches.
""")

with col_math:
    st.subheader("Formulation")
    st.latex(r"\text{FiLM}(x_i^c) = \gamma_i^c \cdot x_i^c + \beta_i^c")
    st.markdown("""
| Symbole | Rôle |
|---------|------|
| **x** | feature map CNN (après BN) |
| **γ, β** | paramètres prédits par le GRU |
| **i** | indice du bloc résiduel |
| **c** | canal des feature maps |
""")

st.divider()

# ─── Architecture + Forces / Limites ──────────────────────────────────────────
col_arch, col_sl = st.columns([1, 1])

with col_arch:
    with st.expander("Architecture complète"):
        st.code("""
Question → Embedding → GRU → Linear → (γk, βk) pour k = 0…3

Image → ResNet101 → (1024, 14, 14)
  → Stem : 2x Conv3x3 → (128, 14, 14)
  → 4x FiLMedResBlock :
      [x ++ coord] → Conv1x1 → ReLU   (résidu)
                   → Conv3x3 → BN → FiLM(γ,β) → ReLU
                   + résidu
  → Conv1x1(→512) → ReLU → MaxPool
  → Flatten → FC(25088→1024) → Dropout(0.5) → FC(1024→28)
        """, language="text")

with col_sl:
    st.subheader("Forces et limites")
    c_f, c_l = st.columns(2)
    with c_f:
        st.markdown("""
**Forces**
- Architecture générique et légère
- Conditionnement cross-modal élégant
- Formulation résiduelle γ = 1 + Δγ stable
- État de l'art sur CLEVR (97,7 %)
""")
    with c_l:
        st.markdown("""
**Limites**
- Dépend de features pré-extraites (ResNet101)
- GRU doit être large (hidden_dim ≥ 1024)
- Entraînement long sur CLEVR complet
- Pas de raisonnement multi-saut natif
""")

st.divider()

# ─── Démarche ─────────────────────────────────────────────────────────────────
st.subheader("Notre démarche")
st.markdown("""
1. **Sort of CLEVR** — dataset Kaggle 2D, entraînement rapide (< 5 min CPU)
2. **CLEVR VQA** — dataset 3D photoréaliste, 700k questions, features ResNet101 (~18 Go)
3. **Référence papier** — 97,7 % avec `hidden_dim=4096`, 80 epochs
""")

st.divider()
st.caption(
    "Implémentation de [Perez et al. 2018](https://arxiv.org/abs/1709.07871) · "
    "[ethanjperez/film](https://github.com/ethanjperez/film)"
)
