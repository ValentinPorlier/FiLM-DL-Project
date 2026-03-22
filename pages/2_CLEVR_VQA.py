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
La question est encodée en deux étapes :

1. **Embedding** : chaque mot → vecteur de dimension 200 (appris à l'entraînement)
2. **GRU** : lit les mots un par un et résume la question dans un vecteur
   $h_T \\in \\mathbb{R}^{4096}$

Le GRU est préféré au LSTM (2 portes vs 3) car les questions CLEVR sont courtes et c'est tout autant performant en étant moins coûteux ce qui suffit pour les questions courtes de CLEVR

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
Ça initialise $\\gamma$ autour de l'identité,
évitant d'annuler les feature maps (et donc les gradients) en début d'entraînement.
""")

with col_v:
    st.subheader("Branche visuelle")
    st.markdown("""
L'image ($224 \\times 224$) passe d'abord dans un **ResNet101 pré-entraîné** coupé après la 3ème couche, produisant des feature maps de forme $(1024, 14, 14)$.

Ces feature maps passent dans **4 blocs résiduels FiLM** où chaque bloc suit :
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
puis un MLP choisit parmi 28 réponses possibles.

L'ensemble (GRU + FiLM generator + CNN + MLP) est entraîné *end-to-end* par Adam
($\\text{lr} = 3 \\times 10^{-4}$, 80 epochs).
""")

st.divider()

# ─── Performances ─────────────────────────────────────────────────────────────
st.header("Performances sur CLEVR")

st.markdown("""
CLEVR contient ~70 000 images 3D et ~700 000 questions de raisonnement en plusieurs étapes (comptage, comparaison, localisation, attributs) sans biais statistique.
""")

m1, m2, m3, m4 = st.columns(4)
m1.metric("FiLM (papier)", "97,6 %", delta="+5,0 pts vs humains")
m2.metric("Humains", "92,6 %")
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
| $\\beta$ seul ($\\gamma := 1$) | 95,9 % |
| $\\gamma$ seul ($\\beta := 0$) | 96,9 % |
| **FiLM complet** | **97,4 %** |
""")

with col_comment:
    st.markdown("""
**Observations :**

- **Sans FiLM → 21,4 %** : c'est la modulation conditionnelle qui fait la différence,
  pas l'architecture du CNN en elle-même.

- **γ seul > β seul** (96,9 % vs 95,9 %) : $\\gamma$ peut annuler entièrement une feature
  map ($\\gamma \\approx 0$), ce qui permet de sélectionner les features utiles, alors que $\\beta$ ne peut que les décaler.

- **γ + β > chacun seul** : ils se complètent — $\\gamma$ sélectionne les features
  pertinentes, $\\beta$ ajuste le seuil d'activation.

- Supprimer la BN ou les skip connections dégrade les performances, mais bien moins
  que supprimer FiLM lui-même.

**Analyse d'erreurs :** 96,1 % des erreurs de comptage sont des erreurs *off-by-one*
(le modèle se trompe de 1), ce qui montre que FiLM a appris le concept de comptage —
les erreurs viennent de cas d'occlusion ou de erreurs de comparaison ensuite.
""")

st.divider()

# ─── Résultats de notre entraînement ──────────────────────────────────────────
RESULTS_FILE = ROOT / "data" / "results_clevr.json"

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
**Pourquoi on ne propose pas d'entraînement interactif ici ?**

Le dataset CLEVR pèse **~18 Go** (images + features ResNet101 pré-extraites),
donc on peut pas le télécharger depuis l'app.
De plus, reproduire les résultats du papier prend plusieurs jours de GPU
(80 epochs sur 700 000 questions avec hidden_dim = 4096).

Nous avons tout de même lancé un entraînement de notre côté (~40 000 itérations,
quelques heures), ce qui explique l'accuracy de validation plus basse :
avec seulement 40 % du training complet et un hidden_dim réduit (256 vs 4096),
le modèle n'a pas eu assez de temps pour atteindre les performances du papier.
""")

st.subheader("Courbes d'apprentissage — notre entraînement")

r1, r2, r3 = st.columns(3)
r1.metric("Val Acc finale", f"{val_acc[-1]:.2%}")
r2.metric("Meilleure Val Acc", f"{max(val_acc):.2%}")
r3.metric("Checkpoints (toutes les 5000 itérations)", n_epochs)

acc_data = {"Val Acc": val_acc}
if train_acc:
    acc_data["Train Acc"] = train_acc
st.line_chart(pd.DataFrame(acc_data))
st.line_chart(pd.DataFrame({"Train Loss": train_loss, "Val Loss": val_loss}))
