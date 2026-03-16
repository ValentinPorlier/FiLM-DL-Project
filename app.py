"""FiLM Explorer — Page d'accueil."""

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
 sys.path.insert(0, str(ROOT))

from src.ui_theme import apply_theme

st.set_page_config(
 page_title="FiLM Explorer",
 page_icon="",
 layout="wide",
 initial_sidebar_state="expanded",
)
apply_theme()

# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding: 2.5rem 0 1rem 0;">
 <div class="hero-title">FiLM Explorer</div>
 <p class="hero-sub">
 Feature-wise Linear Modulation — Raisonnement visuel conditionné par le langage naturel
 </p>
 <span class="badge badge-purple">AAAI 2018</span>
 <span class="badge badge-cyan">Perez et al.</span>
 <span class="badge badge-amber">PyTorch</span>
 <span class="badge badge-green">Streamlit</span>
</div>
""", unsafe_allow_html=True)

st.divider()

# ── Navigation cards ───────────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3, gap="large")

with col1:
 st.markdown("""
 <div class="nav-card">
 <div class="nav-title">Sort of CLEVR</div>
 <div class="nav-desc">
 Version 2D simplifiée de CLEVR — formes colorées sur fond uni.
 Point de départ pour valider l'architecture FiLM avant de
 passer au dataset 3D complet.
 </div>
 </div>
 """, unsafe_allow_html=True)
 st.markdown("<div style='height:0.75rem'></div>", unsafe_allow_html=True)
 st.page_link("pages/1_sort_of_clevr.py", label="Ouvrir Sort of CLEVR")

with col2:
 st.markdown("""
 <div class="nav-card">
 <div class="nav-title">CLEVR VQA</div>
 <div class="nav-desc">
 Entraînement complet sur CLEVR avec features ResNet101 pré-extraites.
 Visualisation en direct des courbes, des paramètres γ/β et des
 prédictions du modèle.
 </div>
 </div>
 """, unsafe_allow_html=True)
 st.markdown("<div style='height:0.75rem'></div>", unsafe_allow_html=True)
 st.page_link("pages/2_clevr_vqa.py", label="Ouvrir CLEVR VQA")

with col3:
 st.markdown("""
 <div class="nav-card">
 <div class="nav-title">Style Transfer</div>
 <div class="nav-desc">
 Conditional Instance Normalisation — le même principe que FiLM
 appliqué au transfert de style artistique.
 Uploadez une image et choisissez un style.
 </div>
 </div>
 """, unsafe_allow_html=True)
 st.markdown("<div style='height:0.75rem'></div>", unsafe_allow_html=True)
 st.page_link("pages/3_style_transfer.py", label="Ouvrir Style Transfer")

st.divider()

# ── Comment fonctionne FiLM ────────────────────────────────────────────────────
st.markdown('<div class="section-header">Comment fonctionne FiLM ?</div>', unsafe_allow_html=True)

col_eq, col_arch = st.columns([1, 2], gap="large")

with col_eq:
 st.markdown("""
 <div class="formula-box">
 <div class="formula-text">FiLM(x) = γ · x + β</div>
 <div style="color:#94a3b8; font-size:0.88rem; line-height:1.9;">
 <b style="color:#818cf8;">γ</b> et <b style="color:#22d3ee;">β</b>
 sont prédits par un GRU<br>
 qui encode la question en tokens.<br><br>
 Ils modulent <b style="color:#e2e8f0;">dynamiquement</b><br>
 les feature maps du CNN visuel<br>
 après chaque Batch Normalisation.<br><br>
 Formulation résiduelle :<br>
 <code style="font-size:0.85rem;">γ = 1 + Δγ</code>
 &nbsp;→ identité par défaut.
 </div>
 </div>
 """, unsafe_allow_html=True)

with col_arch:
 st.markdown("""
 <div class="film-card" style="font-size:0.83rem; font-family:'Fira Code','Courier New',monospace;
 line-height:2.1; color:#e2e8f0;">
 <span style="color:#94a3b8;">Question</span>
 → Embedding → GRU → Linear
 → <span style="color:#6366f1; font-weight:700;">(γₖ, βₖ)</span>
 &nbsp;pour k = 0…3<br>

 <span style="color:#94a3b8;">Image</span>
 → ResNet101 → <span style="color:#22d3ee;">(1024, 14, 14)</span><br>

 &emsp;→ <b>Stem</b> : 2× Conv3×3 → <span style="color:#22d3ee;">(128, 14, 14)</span><br>

 &emsp;→ 4×
 <span style="color:#f59e0b; font-weight:700;">FiLMedResBlock</span> :<br>
 &emsp;&emsp;[x ++ coord] → Conv1×1 → ReLU &nbsp;<span style="color:#64748b;">(résidu)</span><br>
 &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&nbsp;→ Conv3×3 → BN →
 <span style="color:#6366f1; font-weight:700;">FiLM(γ,β)</span> → ReLU<br>
 &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&nbsp;+ résidu<br>

 &emsp;→ [features ++ coord] → Conv1×1(→512) → ReLU → MaxPool2×2<br>
 &emsp;→ Flatten → FC(25 088→1 024) → ReLU
 → <span style="color:#94a3b8;">Dropout(0.5)</span>
 → FC(1024→<span style="color:#10b981; font-weight:700;">28</span>)
 </div>
 """, unsafe_allow_html=True)

# ── Trois idées clés ───────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Trois idées clés</div>', unsafe_allow_html=True)

k1, k2, k3 = st.columns(3, gap="large")
with k1:
 st.markdown("""
 <div class="film-card">
 <div style="font-weight:700; color:#818cf8; margin-bottom:0.4rem; font-size:0.95rem;">
 Conditionnement universel
 </div>
 <div style="color:#94a3b8; font-size:0.83rem; line-height:1.7;">
 FiLM peut conditionner <em>n'importe quel</em> réseau sur
 <em>n'importe quelle</em> modalité — texte, son, métadonnées,
 labels de classe.
 </div>
 </div>
 """, unsafe_allow_html=True)

with k2:
 st.markdown("""
 <div class="film-card">
 <div style="font-weight:700; color:#22d3ee; margin-bottom:0.4rem; font-size:0.95rem;">
 Stable par construction
 </div>
 <div style="color:#94a3b8; font-size:0.83rem; line-height:1.7;">
 <code>γ = 1 + Δγ</code> initialise la modulation à l'identité.
 Le réseau peut ignorer FiLM au départ et l'apprendre progressivement.
 </div>
 </div>
 """, unsafe_allow_html=True)

with k3:
 st.markdown("""
 <div class="film-card">
 <div style="font-weight:700; color:#fbbf24; margin-bottom:0.4rem; font-size:0.95rem;">
 97,7 % sur CLEVR
 </div>
 <div style="color:#94a3b8; font-size:0.83rem; line-height:1.7;">
 Avec 700 k questions d'entraînement et <code>hidden_dim=4096</code>,
 FiLM atteint l'état de l'art sur CLEVR VQA en 2018.
 </div>
 </div>
 """, unsafe_allow_html=True)

# ── Progression Sort of CLEVR → CLEVR ─────────────────────────────────────────
st.markdown('<div class="section-header">Notre démarche</div>', unsafe_allow_html=True)

st.markdown("""
<div style="display:flex; align-items:stretch; gap:1rem; flex-wrap:wrap; margin-bottom:1rem;">

 <div class="film-card" style="flex:1; min-width:220px;">
 <div style="display:flex; align-items:center; margin-bottom:0.75rem;">
 <span class="step-pill">1</span>
 <span style="font-weight:700; color:#6366f1; font-size:0.95rem;">Sort of CLEVR</span>
 </div>
 <div style="color:#94a3b8; font-size:0.83rem; line-height:1.7;">
 Images 2D générées synthétiquement.
 Formes simples (cercle, carré), 6 couleurs, fond uni.
 Dataset léger, entraînement rapide.<br>
 <span class="badge badge-green" style="margin-top:0.5rem; display:inline-block;">~quelques MB</span>
 </div>
 </div>

 <div style="display:flex; align-items:center; font-size:1.5rem; color:#6366f1; padding:0 0.25rem;">→</div>

 <div class="film-card" style="flex:1; min-width:220px;">
 <div style="display:flex; align-items:center; margin-bottom:0.75rem;">
 <span class="step-pill">2</span>
 <span style="font-weight:700; color:#22d3ee; font-size:0.95rem;">CLEVR VQA</span>
 </div>
 <div style="color:#94a3b8; font-size:0.83rem; line-height:1.7;">
 Images 3D photoréalistes, questions compositionnelles complexes.
 700 k paires question-réponse, 70 k images uniques.
 Features ResNet101 pré-extraites.<br>
 <span class="badge badge-amber" style="margin-top:0.5rem; display:inline-block;">18 GB</span>
 </div>
 </div>

 <div style="display:flex; align-items:center; font-size:1.5rem; color:#6366f1; padding:0 0.25rem;">→</div>

 <div class="film-card" style="flex:1; min-width:220px;">
 <div style="display:flex; align-items:center; margin-bottom:0.75rem;">
 <span class="step-pill">3</span>
 <span style="font-weight:700; color:#fbbf24; font-size:0.95rem;">97,7 %</span>
 </div>
 <div style="color:#94a3b8; font-size:0.83rem; line-height:1.7;">
 Résultat du papier original avec l'architecture complète
 (hidden_dim 4096, 80 epochs).
 Notre run (hidden_dim 256, 30 epochs) vise ~65-70 %.<br>
 <span class="badge badge-purple" style="margin-top:0.5rem; display:inline-block;">AAAI 2018</span>
 </div>
 </div>

</div>
""", unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.divider()
st.markdown("""
<div style="text-align:center; color:#475569; font-size:0.78rem; padding:0.5rem 0 1rem 0;">
 Implémentation de
 <a href="https://arxiv.org/abs/1709.07871" target="_blank"
 style="color:#6366f1; text-decoration:none;">Perez et al. 2018</a>
 &nbsp;·&nbsp;
 Code original :
 <a href="https://github.com/ethanjperez/film" target="_blank"
 style="color:#6366f1; text-decoration:none;">ethanjperez/film</a>
 &nbsp;·&nbsp;
 Construit avec PyTorch &amp; Streamlit
</div>
""", unsafe_allow_html=True)
