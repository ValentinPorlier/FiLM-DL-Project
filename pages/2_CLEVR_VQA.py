"""CLEVR VQA — Architecture FiLM et résultats sur le benchmark CLEVR."""


import pandas as pd
import json
import sys
from pathlib import Path
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# st.set_page_config(page_title="CLEVR VQA", layout="wide")
st.title("CLEVR VQA")
st.caption("FiLM sur 700 000 questions — images traitées par ResNet101")
st.divider()

# ─── Architecture ─────────────────────────────────────────────────────────────
st.header("Architecture FiLM")

col_q, col_v = st.columns(2)

with col_q:
    st.subheader("Branche textuelle")
    st.markdown("""
La question passe par deux étapes :

1. **Embedding** : chaque mot → vecteur de dimension 200
2. **GRU** : lit les mots un par un et résume la question dans un vecteur $h_T \\in \\mathbb{R}^{4096}$

Le GRU est plus léger que le LSTM (2 portes au lieu de 3) et suffit pour les questions courtes de CLEVR.

À partir de $h_T$, le FiLM generator produit les paramètres de chaque bloc $k$ :
""")
    st.latex(
        r"[\gamma^{(k)},\, \beta^{(k)}] = W_k\, h_T + b_k,"
        r"\quad W_k \in \mathbb{R}^{2C \times d}"
    )
    st.markdown("""
avec $C = 128$ canaux et $d = 4096$.

En pratique, on prédit $\\Delta\\gamma$ et on pose $\\gamma_{i,c} = 1 + \\Delta\\gamma_{i,c}$.
Comme ça au début de l'entraînement $\\gamma \\approx 1$, FiLM ne modifie rien et les gradients restent stables.
""")

with col_v:
    st.subheader("Branche visuelle")
    st.markdown("""
L'image ($224 \\times 224$) passe dans un **ResNet101 pré-entraîné** coupé après la 3ème couche, ce qui donne des feature maps de taille $(1024, 14, 14)$.

Ces feature maps passent dans **4 blocs résiduels FiLM** :
""")
    st.code(
        "Conv 1×1  →  BN  →  FiLM(γ,β)  →  ReLU  →  Conv 3×3  →  (+ skip)",
        language="text",
    )
    st.markdown("""
FiLM est placé **après** la BN : si c'était avant, la BN annulerait la modulation.

Après les 4 blocs : **Global Average Pooling** → MLP → réponse parmi 28 possibles.

Le tout est entraîné end-to-end avec Adam ($\\text{lr} = 3 \\times 10^{-4}$, 80 epochs).
""")

st.divider()

# ─── Performances ─────────────────────────────────────────────────────────────
st.header("Performances sur CLEVR")

st.markdown("""
CLEVR contient ~70 000 images 3D et ~700 000 questions en plusieurs étapes (comptage, comparaison, localisation, attributs), sans raccourci statistique possible.
""")

m1, m2, m3, m4 = st.columns(4)
m1.metric("FiLM (papier)", "97,6 %", delta="+5,0 % vs humains")
m2.metric("Humains", "92,6 %")
m3.metric("CNN + LSTM (concaténation)", "52,3 %")
m4.metric("PG+EE (supervision prog.)", "96,9 %")

st.divider()

# ─── Résultats de notre entraînement ──────────────────────────────────────────
RESULTS_FILE = ROOT / "clevr" / "data" / "results_clevr.json"

if not RESULTS_FILE.exists():
    st.warning("Fichier results_clevr.json introuvable.")
    st.stop()

with open(RESULTS_FILE) as f:
    history = json.load(f)

val_acc = history["val_acc"]
train_acc = history["train_acc"]
val_loss = history["val_loss"]
train_loss = history["train_loss"]
n_epochs = len(val_acc)

st.subheader("Notre entraînement")

st.info("""
Le dataset CLEVR fait **~18 Go**, donc pas de téléchargement depuis l'app.
Reproduire le papier prend plusieurs jours de GPU (80 epochs, hidden_dim = 4096).

On a quand même lancé un run de notre côté (~40k itérations, quelques heures)
avec un hidden_dim réduit (256 vs 4096), d'où l'accuracy plus basse.
""")

st.subheader("Nos courbes d'entraînement")

r1, r2, r3 = st.columns(3)
r1.metric("Val Acc finale", f"{val_acc[-1]:.2%}")
r2.metric("Meilleure Val Acc", f"{max(val_acc):.2%}")
r3.metric("Checkpoints (toutes les 5000 itérations)", n_epochs)

st.markdown("**Accuracy par checkpoint**")
acc_data = {"Val Acc": val_acc}
if train_acc:
    acc_data["Train Acc"] = train_acc
st.line_chart(pd.DataFrame(acc_data))

st.markdown("**Loss par checkpoint**")
st.line_chart(pd.DataFrame({"Train Loss": train_loss, "Val Loss": val_loss}))
