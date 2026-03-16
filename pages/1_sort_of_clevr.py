"""Sort of CLEVR — Entraînement FiLM sur dataset 2D."""

from __future__ import annotations

import queue
import sys
import threading
import time
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

st.set_page_config(page_title="Sort of CLEVR", layout="wide")

# ── Session state ──────────────────────────────────────────────────────────────
for _k, _v in [
    ("soc_active",    False),
    ("soc_queue",     None),
    ("soc_progress",  None),
    ("soc_history",   None),
    ("soc_per_class", None),
]:
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ── Import modules + détection données ────────────────────────────────────────
def _find(filename: str) -> Path:
    for candidate in [ROOT / "sortofclevr" / filename, ROOT.parent / "sortofclevr" / filename]:
        if candidate.exists():
            return candidate
    return ROOT / "sortofclevr" / filename

TRAIN_H5  = _find("data_train.h5")
TRAIN_CSV = _find("data_train.csv")
TEST_H5   = _find("data_test.h5")
TEST_CSV  = _find("data_test.csv")

SOC_OK   = False
DATA_OK  = False
try:
    from sortofclevr import (  # type: ignore
        HDF5Dataset, SortOfClevrFiLMModel,
        NUM_CLASSES as SOC_NUM_CLASSES,
        evaluate as soc_evaluate,
        evaluate_per_class as soc_eval_per_class,
    )
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    SOC_OK  = True
    DATA_OK = all(p.exists() for p in [TRAIN_H5, TRAIN_CSV, TEST_H5, TEST_CSV])
except Exception:
    pass

# ── Polling (top-level) ────────────────────────────────────────────────────────
if SOC_OK and st.session_state.soc_active and st.session_state.soc_queue is not None:
    _pq = st.session_state.soc_queue
    _latest = None
    while True:
        try:
            _latest = _pq.get_nowait()
        except queue.Empty:
            break
    if _latest:
        if "error" in _latest:
            st.error(f"Erreur :\n```\n{_latest['error']}\n```")
            st.session_state.soc_active = False
        elif _latest.get("done"):
            st.session_state.soc_history  = _latest["history"]
            st.session_state.soc_per_class = _latest.get("per_class")
            st.session_state.soc_active   = False
        else:
            st.session_state.soc_progress = _latest
    if st.session_state.soc_active:
        time.sleep(0.5)
        st.rerun()

# ── UI ─────────────────────────────────────────────────────────────────────────
st.title("Sort of CLEVR")
st.caption("Dataset 2D synthétique — validation de l'architecture FiLM avant CLEVR")

st.divider()

if not SOC_OK:
    st.error("Module `sortofclevr` introuvable. Vérifiez que le dossier est à la racine du projet.")
    st.stop()

if not DATA_OK:
    st.warning(f"Données introuvables. Placez les fichiers `.h5` et `.csv` dans :\n\n`{TRAIN_H5.parent}`")
    st.stop()

# ── Configuration ──────────────────────────────────────────────────────────────
st.subheader("Configuration")
col1, col2, col3 = st.columns(3)
with col1:
    n_epochs = st.slider("Époques", 1, 50, 10)
with col2:
    batch_sz = st.select_slider("Batch size", [32, 64, 128, 256, 512], value=128)
with col3:
    lr = st.select_slider("Learning rate", [1e-4, 5e-4, 1e-3, 3e-3], value=1e-3,
                          format_func=lambda x: f"{x:.0e}")

st.success("Données détectées — prêt à entraîner")

start_btn = st.button("Lancer l'entraînement", disabled=st.session_state.soc_active)

if start_btn and not st.session_state.soc_active:
    pq: queue.Queue = queue.Queue()
    st.session_state.soc_queue    = pq
    st.session_state.soc_active   = True
    st.session_state.soc_progress = None
    st.session_state.soc_history  = None

    def _run_soc(q: queue.Queue) -> None:
        try:
            _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            _pin    = torch.cuda.is_available()

            train_ds = HDF5Dataset(str(TRAIN_H5), "data_train", str(TRAIN_CSV))
            test_ds  = HDF5Dataset(str(TEST_H5),  "data_test",  str(TEST_CSV))
            train_loader = DataLoader(train_ds, batch_size=batch_sz, shuffle=True,
                                      num_workers=0, pin_memory=_pin)
            test_loader  = DataLoader(test_ds,  batch_size=batch_sz, shuffle=False,
                                      num_workers=0, pin_memory=_pin)

            model     = SortOfClevrFiLMModel(num_answers=SOC_NUM_CLASSES).to(_device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            criterion = nn.CrossEntropyLoss()
            history   = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

            n_batches = len(train_loader)
            for epoch in range(n_epochs):
                model.train()
                run_loss, correct, total = 0.0, 0, 0
                print(f"[SoC] Époque {epoch+1}/{n_epochs}", flush=True)
                for batch_i, (_, imgs, labels, encs) in enumerate(train_loader):
                    imgs, encs, labels = imgs.to(_device), encs.to(_device), labels.to(_device)
                    optimizer.zero_grad()
                    out  = model(imgs, encs)
                    loss = criterion(out, labels)
                    loss.backward()
                    optimizer.step()
                    run_loss += loss.item()
                    total    += labels.size(0)
                    correct  += out.argmax(1).eq(labels).sum().item()
                    if (batch_i + 1) % 50 == 0:
                        print(f"  batch {batch_i+1}/{n_batches} | loss={loss.item():.4f} | acc={correct/total:.1%}", flush=True)

                t_loss = run_loss / len(train_loader)
                t_acc  = correct / total
                v_loss, v_acc = soc_evaluate(model, test_loader, criterion, _device)

                history["train_loss"].append(t_loss)
                history["train_acc"].append(t_acc)
                history["val_loss"].append(v_loss)
                history["val_acc"].append(v_acc)
                print(f"  => Époque {epoch+1} | Train {t_acc:.2%} | Val {v_acc:.2%}", flush=True)

                q.put({"epoch": epoch + 1, "num_epochs": n_epochs,
                       "train_loss": t_loss, "train_acc": t_acc,
                       "val_loss": v_loss, "val_acc": v_acc})

            per_class = soc_eval_per_class(model, test_loader, _device)
            q.put({"done": True, "history": history, "per_class": per_class})

        except Exception:
            import traceback
            q.put({"error": traceback.format_exc(), "done": True})

    threading.Thread(target=_run_soc, args=(pq,), daemon=True).start()

# ── Progression ────────────────────────────────────────────────────────────────
prog = st.session_state.soc_progress
if prog and not st.session_state.soc_history:
    ep, tot = prog["epoch"], prog["num_epochs"]
    st.progress(ep / tot, text=f"Époque {ep}/{tot}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Train Loss", f"{prog['train_loss']:.4f}")
    c2.metric("Train Acc",  f"{prog['train_acc']:.1%}")
    c3.metric("Val Loss",   f"{prog['val_loss']:.4f}")
    c4.metric("Val Acc",    f"{prog['val_acc']:.1%}")

# ── Résultats ──────────────────────────────────────────────────────────────────
history = st.session_state.soc_history
if history:
    st.divider()
    st.subheader("Résultats")
    st.progress(1.0)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Meilleure Val Acc", f"{max(history['val_acc']):.1%}")
    c2.metric("Val Acc finale",    f"{history['val_acc'][-1]:.1%}")
    c3.metric("Train Acc finale",  f"{history['train_acc'][-1]:.1%}")
    c4.metric("Époques",           len(history["train_loss"]))

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        epochs = list(range(1, len(history["train_loss"]) + 1))
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Loss", "Accuracy"))
        for name, y, col in [
            ("Train Loss", history["train_loss"], 1),
            ("Val Loss",   history["val_loss"],   1),
            ("Train Acc",  history["train_acc"],  2),
            ("Val Acc",    history["val_acc"],    2),
        ]:
            fig.add_trace(go.Scatter(x=epochs, y=y, name=name, mode="lines+markers"), row=1, col=col)
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
    except ImportError:
        st.line_chart({"Train Acc": history["train_acc"], "Val Acc": history["val_acc"]})

    if st.session_state.soc_per_class:
        st.subheader("Accuracy par classe")
        import pandas as pd
        per = st.session_state.soc_per_class
        df_pc = pd.DataFrame(
            [{"Classe": c, "Accuracy": f"{v:.1%}"} for c, v in sorted(per.items(), key=lambda x: -x[1])]
        )
        st.dataframe(df_pc, use_container_width=True, hide_index=True)

    if st.button("Réinitialiser"):
        for k in ["soc_active", "soc_queue", "soc_progress", "soc_history", "soc_per_class"]:
            st.session_state[k] = False if k == "soc_active" else None
        st.rerun()
