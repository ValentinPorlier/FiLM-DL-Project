"""CLEVR VQA — Architecture FiLM et résultats sur le benchmark CLEVR."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

st.set_page_config(page_title="CLEVR VQA", layout="wide")
st.title("CLEVR VQA")
st.caption("FiLM sur 700 000 questions — features ResNet101 pré-extraites")
st.divider()

# ─── Architecture ─────────────────────────────────────────────────────────────
st.header("Architecture FiLM")

col_q, col_v = st.columns(2)

with col_q:
    st.subheader("Branche textuelle")
    st.markdown("""
La question est encodée en deux étapes :

1. **Embedding** : chaque mot → vecteur de dimension 200 (appris à l'entraînement)
2. **GRU** : lit les mots séquentiellement et produit un état caché final
   $h_T \\in \\mathbb{R}^{4096}$ résumant le sens de la question

Le GRU est préféré au LSTM (2 portes vs 3) car les questions CLEVR sont courtes.

À partir de $h_T$, le **FiLM generator** produit les paramètres de chaque bloc $k$ :
""")
    st.latex(
        r"[\gamma^{(k)},\, \beta^{(k)}] = W_k\, h_T + b_k,"
        r"\quad W_k \in \mathbb{R}^{2C \times d}"
    )
    st.markdown("""
avec $C = 128$ canaux et $d = 4096$.

En pratique, le modèle prédit $\\Delta\\gamma$ plutôt que $\\gamma$ directement :
$$\\gamma_{i,c} = 1 + \\Delta\\gamma_{i,c}$$
Cette paramétrisation résiduelle initialise $\\gamma$ autour de l'identité,
évitant d'annuler les feature maps (et donc les gradients) en début d'entraînement.
""")

with col_v:
    st.subheader("Branche visuelle")
    st.markdown("""
L'image ($224 \\times 224$) passe d'abord dans un **ResNet101 pré-entraîné** tronqué
après `layer3`, produisant des feature maps de forme $(1024, 14, 14)$.

Ces features alimentent **4 blocs résiduels FiLM** dont chaque bloc suit :
""")
    st.code(
        "Conv 1×1  →  BN  →  FiLM(γ,β)  →  ReLU  →  Conv 3×3  →  (+ skip)",
        language="text",
    )
    st.markdown("""
**Pourquoi FiLM est placé après la Batch Normalisation ?**
La BN normalise les activations — si FiLM était avant, la BN annulerait la modulation.
Après la BN, les activations sont centrées et réduites : FiLM peut librement les
rescaler et redécaler.

Après les 4 blocs : **Global Average Pooling** réduit chaque feature map en un scalaire,
puis un MLP produit une distribution sur les **28 réponses** possibles.

L'ensemble (GRU + FiLM generator + CNN + MLP) est entraîné *end-to-end* par Adam
($\\text{lr} = 3 \\times 10^{-4}$, 80 epochs).
""")

st.divider()

# ─── Performances ─────────────────────────────────────────────────────────────
st.header("Performances sur CLEVR")

st.markdown("""
CLEVR contient ~70 000 images 3D photoréalistes et ~700 000 questions de raisonnement
compositionnel (comptage, comparaison, localisation, attributs) — sans biais statistique
exploitable.
""")

m1, m2, m3, m4 = st.columns(4)
m1.metric("FiLM (papier)",              "97,6 %", delta="+5,0 pts vs humains")
m2.metric("Humains",                    "92,6 %")
m3.metric("CNN + LSTM (concaténation)", "52,3 %")
m4.metric("PG+EE (supervision prog.)", "96,9 %")

st.divider()

# ─── Ablations ────────────────────────────────────────────────────────────────
st.subheader("Contribution de chaque composant")

col_tab, col_comment = st.columns([1, 2])

with col_tab:
    st.markdown("""
| Variante | Accuracy |
|---|---|
| Sans FiLM (aucun bloc) | 21,4 % |
| Sans Batch Normalisation | 93,7 % |
| Sans connexion résiduelle | 94,0 % |
| $\\gamma := 1$ — $\\beta$ seul | 95,9 % |
| $\\beta := 0$ — $\\gamma$ seul | 96,9 % |
| **FiLM complet** | **97,4 %** |
""")

with col_comment:
    st.markdown("""
**Observations :**

- **Sans FiLM → 21,4 %** : c'est la modulation conditionnelle qui fait la différence,
  pas l'architecture du CNN en elle-même.

- **γ seul > β seul** (96,9 % vs 95,9 %) : $\\gamma$ peut annuler entièrement une feature
  map ($\\gamma \\approx 0$), offrant un mécanisme de sélection plus puissant que le simple
  décalage de $\\beta$.

- **γ + β > chacun seul** : ils se complètent — $\\gamma$ sélectionne les features
  pertinentes, $\\beta$ déplace le point de fonctionnement.

- Supprimer la BN ou les skip connections dégrade les performances, mais bien moins
  que supprimer FiLM lui-même.

**Analyse d'erreurs :** 96,1 % des erreurs de comptage sont des erreurs *off-by-one*
(le modèle se trompe de 1), ce qui montre que FiLM a appris le concept de comptage —
les erreurs viennent de cas d'occlusion ou de confusion lors de la comparaison finale.
""")

st.divider()

# ─── Résultats de notre entraînement ──────────────────────────────────────────
RESULTS_FILE = ROOT / "data" / "results_clevr.json"

if not RESULTS_FILE.exists():
    st.warning("Fichier results_clevr.json introuvable.")
    st.stop()

with open(RESULTS_FILE) as f:
    history = json.load(f)

val_acc    = history["val_acc"]
train_acc  = history["train_acc"]
val_loss   = history["val_loss"]
train_loss = history["train_loss"]
n_epochs   = len(val_acc)

st.subheader("Courbes d'apprentissage — notre entraînement")

r1, r2, r3 = st.columns(3)
r1.metric("Val Acc finale",    f"{val_acc[-1]:.2%}")
r2.metric("Meilleure Val Acc", f"{max(val_acc):.2%}")
r3.metric("Epochs entraînées", n_epochs)

try:
    from src.visualize import plot_training_curves
    st.plotly_chart(plot_training_curves(history), width="stretch")
except Exception:
    import pandas as pd
    st.line_chart(pd.DataFrame({"Train Acc": train_acc, "Val Acc": val_acc}))
    st.line_chart(pd.DataFrame({"Train Loss": train_loss, "Val Loss": val_loss}))

if history.get("config"):
    with st.expander("Configuration du modèle"):
        cfg = history["config"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Blocs FiLM",    cfg.get("num_blocks", "—"))
        c2.metric("Canaux",         cfg.get("num_channels", "—"))
        c3.metric("GRU hidden dim", cfg.get("hidden_dim", "—"))
        c1.metric("Learning rate",  cfg.get("learning_rate", "—"))
        c2.metric("Batch size",     cfg.get("batch_size", "—"))
        c3.metric(
            "Train samples",
            f"{cfg.get('max_samples_train', 0):,}" if cfg.get("max_samples_train") else "—",
        )
