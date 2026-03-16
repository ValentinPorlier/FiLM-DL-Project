"""CLEVR VQA — Entraînement FiLM sur features ResNet101."""

from __future__ import annotations

import queue
import sys
import threading
import time
from pathlib import Path

import streamlit as st
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.train import TrainConfig, train

st.set_page_config(page_title="CLEVR VQA", layout="wide")

# ── Session state ──────────────────────────────────────────────────────────────
for key, default in [
    ("runs", []),
    ("training_active", False),
    ("progress_queue", None),
    ("current_history", None),
    ("last_progress", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── Sidebar — Configuration ────────────────────────────────────────────────────
st.sidebar.subheader("Modèle")
num_blocks   = st.sidebar.slider("Blocs résiduels", 1, 6, 4)
num_channels = st.sidebar.select_slider("Largeur feature maps", [32, 64, 128], value=128)
hidden_dim   = st.sidebar.select_slider(
    "GRU hidden dim", [256, 512, 1024, 2048, 4096], value=256,
    help="Le papier utilise 4096. Pour tests rapides, 256 suffit.",
)
fix_gamma = st.sidebar.checkbox("Fixer γ = 1", value=False)
fix_beta  = st.sidebar.checkbox("Fixer β = 0", value=False)

st.sidebar.subheader("Entraînement")
num_epochs    = st.sidebar.slider("Époques", 1, 50, 20)
batch_size    = st.sidebar.select_slider("Batch size", [16, 32, 64, 128], value=64)
learning_rate = st.sidebar.select_slider(
    "Learning rate", [1e-4, 3e-4, 5e-4, 1e-3], value=3e-4,
    format_func=lambda x: f"{x:.0e}",
)
max_samples = st.sidebar.slider("Questions d'entraînement", 500, 700_000, 20_000, step=500)

st.sidebar.subheader("Données")
data_dir = st.sidebar.text_input(
    "Répertoire CLEVR",
    value="C:/Users/ilies/Documents/cours/projFiLM/CLEVR_v1.0",
)

# ── Statut données ─────────────────────────────────────────────────────────────
data_ready = (
    (Path(data_dir) / "questions" / "CLEVR_train_questions.json").exists()
    and (Path(data_dir) / "images" / "train").exists()
)

def _h5_count(path: Path) -> int:
    try:
        import h5py
        with h5py.File(path, "r") as f:
            return int(f["features"].shape[0])
    except Exception:
        return 0

n_feat_train   = _h5_count(Path(data_dir) / "features_train.h5")
n_feat_val     = _h5_count(Path(data_dir) / "features_val.h5")
features_ready = n_feat_train >= max_samples

if data_ready:
    if features_ready:
        st.sidebar.success(f"Features prêtes — {n_feat_train:,} train / {n_feat_val:,} val")
    elif n_feat_train > 0:
        st.sidebar.warning(f"Features : {n_feat_train:,} images (besoin : {max_samples:,})")
    else:
        st.sidebar.warning("Features non extraites.")
else:
    st.sidebar.error("CLEVR introuvable au chemin indiqué.")

extract_btn = st.sidebar.button(
    "Extraire features ResNet101",
    disabled=not data_ready or features_ready or st.session_state.training_active,
)
start_btn = st.sidebar.button(
    "Lancer l'entraînement",
    disabled=st.session_state.training_active or not data_ready,
    type="primary",
)

# ── Extraction features ────────────────────────────────────────────────────────
if extract_btn:
    from data.extract_features import extract
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for split, n in [("train", max_samples), ("val", min(max_samples // 5, 9_900))]:
        with st.spinner(f"Extraction ResNet101 — {split} ({n:,} images)…"):
            extract(data_dir=data_dir, split=split, max_images=n, batch_size=64, device=device)
    st.success("Extraction terminée !")
    st.rerun()

# ── Lancer l'entraînement ──────────────────────────────────────────────────────
if start_btn and not st.session_state.training_active:
    run_name = (
        f"blocks{num_blocks}_ch{num_channels}"
        f"_g{'fix' if fix_gamma else 'dyn'}"
        f"_b{'fix' if fix_beta else 'dyn'}"
        f"_{int(time.time()) % 100_000}"
    )
    config = TrainConfig(
        data_dir=data_dir,
        max_samples_train=max_samples,
        max_samples_val=min(max_samples // 5, 9_900),
        num_blocks=num_blocks,
        num_channels=num_channels,
        embedding_dim=300,
        hidden_dim=hidden_dim,
        rnn_num_layers=2,
        fix_gamma=fix_gamma,
        fix_beta=fix_beta,
        dropout=0.0,
        classifier_dropout=0.5,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        run_dir=f"runs/{run_name}",
        save_gamma_beta=True,
    )
    pq: queue.Queue = queue.Queue()
    st.session_state.progress_queue  = pq
    st.session_state.training_active = True
    st.session_state.current_history = None
    st.session_state.last_progress   = None

    def _run_training(cfg, q):
        try:
            history = train(cfg, progress_queue=q)
            history["config"] = cfg.__dict__.copy()
            q.put({"done": True, "history": history})
        except Exception:
            import traceback
            q.put({"error": traceback.format_exc(), "done": True})

    threading.Thread(target=_run_training, args=(config, pq), daemon=True).start()

# ── Polling ────────────────────────────────────────────────────────────────────
pq     = st.session_state.progress_queue
active = st.session_state.training_active

if active and pq is not None:
    latest = None
    while True:
        try:
            latest = pq.get_nowait()
        except queue.Empty:
            break
    if latest:
        if "error" in latest:
            st.error(f"Erreur :\n```\n{latest['error']}\n```")
            st.session_state.training_active = False
        elif "history" in latest:
            st.session_state.current_history = latest["history"]
            st.session_state.runs.append(latest["history"])
            st.session_state.training_active = False
        else:
            st.session_state.last_progress = latest
    time.sleep(0.5)
    st.rerun()

# ── UI ─────────────────────────────────────────────────────────────────────────
st.title("CLEVR VQA")
st.caption("Entraînement FiLM sur features ResNet101 pré-extraites — 28 classes de réponses")

st.divider()

prog    = st.session_state.last_progress
history = st.session_state.current_history

# ── Progression en cours ───────────────────────────────────────────────────────
if active:
    st.info("Chargement du dataset et initialisation du modèle…")

if prog:
    epoch    = prog.get("epoch", 0)
    n_epochs = prog.get("num_epochs", num_epochs)
    st.progress(epoch / n_epochs, text=f"Époque {epoch}/{n_epochs}")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Train Loss", f"{prog['train_loss']:.4f}")
    m2.metric("Train Acc",  f"{prog['train_acc']:.2%}")
    m3.metric("Val Loss",   f"{prog['val_loss']:.4f}")
    m4.metric("Val Acc",    f"{prog['val_acc']:.2%}")

# ── Résultats ──────────────────────────────────────────────────────────────────
if history:
    st.success("Entraînement terminé !")
    st.progress(1.0)

    m1, m2, m3 = st.columns(3)
    m1.metric("Val Acc finale",  f"{history['val_acc'][-1]:.2%}")
    m2.metric("Val Loss finale", f"{history['val_loss'][-1]:.4f}")
    m3.metric("Époques",         len(history["train_loss"]))

    st.subheader("Courbes d'apprentissage")
    try:
        from src.visualize import plot_training_curves
        st.plotly_chart(plot_training_curves(history), width="stretch")
    except Exception:
        st.line_chart({"Train Acc": history["train_acc"], "Val Acc": history["val_acc"]})

# ── Comparaison des runs ───────────────────────────────────────────────────────
if st.session_state.runs:
    st.divider()
    st.subheader("Comparaison des runs")
    import pandas as pd
    rows = []
    for i, r in enumerate(st.session_state.runs):
        c = r.get("config", {})
        rows.append({
            "Run":       i + 1,
            "Blocs":     c.get("num_blocks", "?"),
            "Canaux":    c.get("num_channels", "?"),
            "γ fixe":    c.get("fix_gamma", "?"),
            "β fixe":    c.get("fix_beta",  "?"),
            "Époques":   len(r.get("train_loss", [])),
            "Val Acc":   f"{r['val_acc'][-1]:.2%}" if r.get("val_acc") else "?",
            "Val Loss":  f"{r['val_loss'][-1]:.4f}" if r.get("val_loss") else "?",
        })
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    if st.button("Effacer tous les runs"):
        st.session_state.runs            = []
        st.session_state.current_history = None
        st.session_state.last_progress   = None
        st.rerun()

# ── Aucun run, instructions minimales ─────────────────────────────────────────
if not prog and not history and not st.session_state.runs and not active:
    st.markdown("""
1. Indiquez le chemin CLEVR dans la sidebar
2. Cliquez **Extraire features ResNet101** (une seule fois, ~20 min GPU)
3. Configurez et cliquez **Lancer l'entraînement**
    """)
