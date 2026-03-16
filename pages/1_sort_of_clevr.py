"""Sort of CLEVR — Entraînement FiLM sur dataset 2D."""

import sys
from pathlib import Path

import streamlit as st
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sortofclevr import HDF5Dataset, SortOfClevrFiLMModel, NUM_CLASSES
from sortofclevr.train import train_model, evaluate_per_class

st.set_page_config(page_title="Sort of CLEVR", layout="wide")
st.title("Sort of CLEVR")
st.divider()

# ── Chemin des données ─────────────────────────────────────────────────────────
data_dir = st.text_input("Dossier des données", value=str(ROOT.parent / "sortofclevr"))

train_h5  = Path(data_dir) / "data_train.h5"
train_csv = Path(data_dir) / "data_train.csv"
test_h5   = Path(data_dir) / "data_test.h5"
test_csv  = Path(data_dir) / "data_test.csv"

if not (train_h5.exists() and train_csv.exists() and test_h5.exists() and test_csv.exists()):
    st.warning("Fichiers introuvables dans ce dossier.")
    st.stop()

st.success("Données détectées")

# ── Configuration ──────────────────────────────────────────────────────────────
n_epochs = st.slider("Époques", 1, 50, 10)
batch_sz = st.slider("Batch size", 32, 512, 128, step=32)
lr       = st.number_input("Learning rate", value=0.001, format="%.4f")

if st.button("Lancer l'entraînement"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = HDF5Dataset(str(train_h5), "data_train", str(train_csv))
    test_ds  = HDF5Dataset(str(test_h5),  "data_test",  str(test_csv))

    train_loader = DataLoader(train_ds, batch_size=batch_sz, shuffle=True,  num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=batch_sz, shuffle=False, num_workers=0)

    model     = SortOfClevrFiLMModel(num_answers=NUM_CLASSES).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    with st.spinner("Entraînement en cours... (progression dans le terminal)"):
        history = train_model(
            model, train_loader, test_loader,
            optimizer, criterion, device,
            epochs=n_epochs
        )

    per_class = evaluate_per_class(model, test_loader, device)

    # ── Résultats ──────────────────────────────────────────────────────────────
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
    st.dataframe(rows, use_container_width=True)
