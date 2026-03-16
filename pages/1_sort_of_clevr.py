"""Sort of CLEVR — Entraînement FiLM sur dataset 2D."""

import queue
import sys
import threading
import time
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

# ── Session state ──────────────────────────────────────────────────────────────
if "soc_active" not in st.session_state:
    st.session_state.soc_active = False
if "soc_queue" not in st.session_state:
    st.session_state.soc_queue = None
if "soc_progress" not in st.session_state:
    st.session_state.soc_progress = None
if "soc_history" not in st.session_state:
    st.session_state.soc_history = None
if "soc_per_class" not in st.session_state:
    st.session_state.soc_per_class = None

# ── Polling (top-level) ────────────────────────────────────────────────────────
if st.session_state.soc_active and st.session_state.soc_queue is not None:
    latest = None
    while True:
        try:
            latest = st.session_state.soc_queue.get_nowait()
        except queue.Empty:
            break

    if latest is not None:
        if "error" in latest:
            st.error(latest["error"])
            st.session_state.soc_active = False
        elif latest.get("done"):
            st.session_state.soc_history = latest["history"]
            st.session_state.soc_per_class = latest.get("per_class")
            st.session_state.soc_active = False
        else:
            st.session_state.soc_progress = latest

    if st.session_state.soc_active:
        time.sleep(0.5)
        st.rerun()

# ── UI ─────────────────────────────────────────────────────────────────────────
st.title("Sort of CLEVR")
st.divider()

data_dir = st.text_input("Dossier des données", value=str(ROOT.parent / "sortofclevr"))

train_h5  = Path(data_dir) / "data_train.h5"
train_csv = Path(data_dir) / "data_train.csv"
test_h5   = Path(data_dir) / "data_test.h5"
test_csv  = Path(data_dir) / "data_test.csv"

if not (train_h5.exists() and train_csv.exists() and test_h5.exists() and test_csv.exists()):
    st.warning("Fichiers introuvables dans ce dossier.")
    st.stop()

st.success("Données détectées")

n_epochs = st.slider("Époques", 1, 50, 10)
batch_sz = st.slider("Batch size", 32, 512, 128, step=32)
lr       = st.number_input("Learning rate", value=0.001, format="%.4f")

start_btn = st.button("Lancer l'entraînement", disabled=st.session_state.soc_active)

if start_btn:
    pq = queue.Queue()
    st.session_state.soc_queue = pq
    st.session_state.soc_active = True
    st.session_state.soc_progress = None
    st.session_state.soc_history = None

    def run_training(q):
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            train_ds = HDF5Dataset(str(train_h5), "data_train", str(train_csv))
            test_ds  = HDF5Dataset(str(test_h5),  "data_test",  str(test_csv))

            train_loader = DataLoader(train_ds, batch_size=batch_sz, shuffle=True,  num_workers=0)
            test_loader  = DataLoader(test_ds,  batch_size=batch_sz, shuffle=False, num_workers=0)

            model     = SortOfClevrFiLMModel(num_answers=NUM_CLASSES).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            criterion = nn.CrossEntropyLoss()

            history = train_model(
                model, train_loader, test_loader,
                optimizer, criterion, device,
                epochs=n_epochs, progress_queue=q
            )

            per_class = evaluate_per_class(model, test_loader, device)
            q.put({"done": True, "history": history, "per_class": per_class})

        except Exception:
            import traceback
            q.put({"error": traceback.format_exc(), "done": True})

    threading.Thread(target=run_training, args=(pq,), daemon=True).start()

# ── Progression ────────────────────────────────────────────────────────────────
prog = st.session_state.soc_progress
if prog is not None and st.session_state.soc_history is None:
    ep  = prog["epoch"]
    tot = prog["num_epochs"]
    st.progress(ep / tot, text=f"Époque {ep}/{tot}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Train Loss", f"{prog['train_loss']:.4f}")
    c2.metric("Train Acc",  f"{prog['train_acc']:.1%}")
    c3.metric("Val Loss",   f"{prog['val_loss']:.4f}")
    c4.metric("Val Acc",    f"{prog['val_acc']:.1%}")

# ── Résultats ──────────────────────────────────────────────────────────────────
history = st.session_state.soc_history
if history is not None:
    st.divider()
    st.subheader("Résultats")

    best_val = max(history["val_acc"])
    c1, c2, c3 = st.columns(3)
    c1.metric("Meilleure Val Acc", f"{best_val:.1%}")
    c2.metric("Val Acc finale",    f"{history['val_acc'][-1]:.1%}")
    c3.metric("Train Acc finale",  f"{history['train_acc'][-1]:.1%}")

    st.line_chart({
        "Train Loss": history["train_loss"],
        "Val Loss":   history["val_loss"],
    })
    st.line_chart({
        "Train Acc": history["train_acc"],
        "Val Acc":   history["val_acc"],
    })

    if st.session_state.soc_per_class is not None:
        st.subheader("Accuracy par classe")
        rows = []
        for classe, acc in st.session_state.soc_per_class.items():
            rows.append({"Classe": classe, "Accuracy": f"{acc:.1%}"})
        rows.sort(key=lambda r: r["Accuracy"], reverse=True)
        st.dataframe(rows, use_container_width=True)

    if st.button("Réinitialiser"):
        st.session_state.soc_active = False
        st.session_state.soc_queue = None
        st.session_state.soc_progress = None
        st.session_state.soc_history = None
        st.session_state.soc_per_class = None
        st.rerun()
