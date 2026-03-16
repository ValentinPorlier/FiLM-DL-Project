"""Sort of CLEVR — Entraînement FiLM sur dataset 2D."""

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sortofclevr.train import run

st.set_page_config(page_title="Sort of CLEVR", layout="wide")
st.title("Sort of CLEVR")
st.divider()

data_dir = st.text_input("Chemin vers le dossier des données (contenant data_train.h5, etc.)")

train_h5  = Path(data_dir) / "data_train.h5"
train_csv = Path(data_dir) / "data_train.csv"
test_h5   = Path(data_dir) / "data_test.h5"
test_csv  = Path(data_dir) / "data_test.csv"

if not data_dir:
    st.stop()

if not (train_h5.exists() and train_csv.exists() and test_h5.exists() and test_csv.exists()):
    st.warning("Fichiers introuvables dans ce dossier.")
    st.stop()

st.success("Données détectées")

n_epochs    = st.slider("Epochs", 1, 50, 10)
batch_sz    = st.slider("Batch size", 32, 512, 128, step=32)
lr          = st.number_input("Learning rate", value=0.001, format="%.4f")
max_samples  = st.slider("Samples d'entraînement", 1000, 70000, 10000, step=1000)

if st.button("Lancer l'entraînement"):
    with st.spinner("Entraînement en cours... (progression dans le terminal)"):
        history, per_class = run(train_h5, train_csv, test_h5, test_csv,
                                 epochs=n_epochs, batch_size=batch_sz, lr=lr,
                                 max_samples=max_samples)

    st.subheader("Résultats")
    c1, c2, c3 = st.columns(3)
    c1.metric("Meilleure Val Acc", f"{max(history['val_acc']):.1%}")
    c2.metric("Val Acc finale",    f"{history['val_acc'][-1]:.1%}")
    c3.metric("Train Acc finale",  f"{history['train_acc'][-1]:.1%}")

    st.line_chart({"Train Loss": history["train_loss"], "Val Loss": history["val_loss"]})
    st.line_chart({"Train Acc":  history["train_acc"],  "Val Acc":  history["val_acc"]})

    st.subheader("Accuracy par classe")
    rows = []
    for classe, acc in per_class.items():
        rows.append({"Classe": classe, "Accuracy": f"{acc:.1%}"})
    rows.sort(key=lambda r: r["Accuracy"], reverse=True)
    st.dataframe(rows, width="stretch")
