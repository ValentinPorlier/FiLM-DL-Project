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

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Sort of CLEVR")
    st.write("Version 2D simplifiée de CLEVR pour valider l'architecture FiLM sur un dataset léger.")
    st.page_link("pages/1_sort_of_clevr.py", label="Ouvrir")

with col2:
    st.subheader("CLEVR VQA")
    st.write("Entraînement complet sur CLEVR avec features ResNet101 pré-extraites.")
    st.page_link("pages/2_clevr_vqa.py", label="Ouvrir")

with col3:
    st.subheader("Style Transfer")
    st.write("Conditional Instance Normalisation — même principe que FiLM appliqué au style artistique.")
    st.page_link("pages/3_style_transfer.py", label="Ouvrir")

st.divider()

st.subheader("Comment fonctionne FiLM ?")

st.latex(r"\text{FiLM}(x) = \gamma \cdot x + \beta")

st.markdown("""
`γ` et `β` sont prédits par un GRU qui encode la question.
Ils modulent les feature maps du CNN après chaque Batch Normalisation.
Formulation résiduelle : `γ = 1 + Δγ` → identité par défaut.
""")

with st.expander("Architecture complète"):
    st.code("""
Question → Embedding → GRU → Linear → (γk, βk) pour k = 0…3

Image → ResNet101 → (1024, 14, 14)
  → Stem : 2x Conv3x3 → (128, 14, 14)
  → 4x FiLMedResBlock :
      [x ++ coord] → Conv1x1 → ReLU  (résidu)
                   → Conv3x3 → BN → FiLM(γ,β) → ReLU
                   + résidu
  → Conv1x1(→512) → ReLU → MaxPool
  → Flatten → FC(25088→1024) → Dropout(0.5) → FC(1024→28)
    """, language="text")

st.divider()

st.subheader("Notre démarche")
st.markdown("""
1. **Sort of CLEVR** — dataset 2D synthétique, formes simples, entraînement rapide (~quelques Mo)
2. **CLEVR VQA** — dataset 3D photoréaliste, 700k questions, features ResNet101 (~18 Go)
3. **Référence papier** — 97,7% avec `hidden_dim=4096`, 80 epochs
""")

st.divider()
st.caption("Implémentation de [Perez et al. 2018](https://arxiv.org/abs/1709.07871) · [ethanjperez/film](https://github.com/ethanjperez/film)")
