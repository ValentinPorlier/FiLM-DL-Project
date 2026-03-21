"""CLEVR VQA — Résultats FiLM sur features ResNet101."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

st.set_page_config(page_title="CLEVR VQA", layout="wide")

RESULTS_FILE = ROOT / "data" / "results_clevr.json"

st.title("CLEVR VQA")
st.caption("Résultats FiLM entraîné sur 700 000 questions CLEVR — features ResNet101 pré-extraites")

st.divider()

# ── Chargement résultats ───────────────────────────────────────────────────────
if not RESULTS_FILE.exists():
    st.error("Fichier results_clevr.json introuvable.")
    st.stop()

with open(RESULTS_FILE) as f:
    history = json.load(f)

val_acc   = history["val_acc"]
train_acc = history["train_acc"]
val_loss  = history["val_loss"]
train_loss = history["train_loss"]
n_epochs  = len(val_acc)

# ── Métriques ─────────────────────────────────────────────────────────────────
m1, m2, m3, m4 = st.columns(4)
m1.metric("Val Acc (dernière epoch)", f"{val_acc[-1]:.2%}")
m2.metric("Meilleure Val Acc",        f"{max(val_acc):.2%}")
m3.metric("Epochs entraînées",        n_epochs)
m4.metric("Dataset",                  "700 000 questions")

st.divider()

# ── Courbes ───────────────────────────────────────────────────────────────────
st.subheader("Courbes d'apprentissage")

try:
    from src.visualize import plot_training_curves
    st.plotly_chart(plot_training_curves(history), width="stretch")
except Exception:
    import pandas as pd
    st.line_chart(pd.DataFrame({
        "Train Acc": train_acc,
        "Val Acc":   val_acc,
    }))
    st.line_chart(pd.DataFrame({
        "Train Loss": train_loss,
        "Val Loss":   val_loss,
    }))

# ── Config ────────────────────────────────────────────────────────────────────
if history.get("config"):
    with st.expander("Configuration du modèle"):
        cfg = history["config"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Blocs FiLM",   cfg.get("num_blocks", "—"))
        c2.metric("Canaux",        cfg.get("num_channels", "—"))
        c3.metric("GRU hidden dim",cfg.get("hidden_dim", "—"))
        c1.metric("Learning rate", cfg.get("learning_rate", "—"))
        c2.metric("Batch size",    cfg.get("batch_size", "—"))
        c3.metric("Train samples", f"{cfg.get('max_samples_train', 0):,}" if cfg.get("max_samples_train") else "—")
