"""Streamlit page: Entraînement FiLM sur CLEVR VQA."""

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
from src.ui_theme import apply_theme
from src.visualize import (
 plot_gamma_beta_hist,
 plot_predictions_grid,
 plot_training_curves,
 plot_feature_maps,
)

st.set_page_config(
 page_title="CLEVR VQA — FiLM Explorer",
 page_icon="",
 layout="wide",
 initial_sidebar_state="expanded",
)
apply_theme()

# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding:1.5rem 0 0.5rem 0;">
 <div class="hero-title" style="font-size:clamp(1.8rem,4vw,3rem);">
 CLEVR VQA
 </div>
 <p class="hero-sub">
 Entraînement FiLM sur features ResNet101 — visualisation en direct
 </p>
 <span class="badge badge-cyan">700 k questions</span>
 <span class="badge badge-purple">ResNet101 1024ch</span>
 <span class="badge badge-amber">28 classes</span>
</div>
""", unsafe_allow_html=True)

st.divider()

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

# ══════════════════════════════════════════════════════════════════════════════
# Sidebar — Configuration
# ══════════════════════════════════════════════════════════════════════════════
st.sidebar.markdown("""
<div style="padding:0.5rem 0 1rem 0;">
 <div style="font-size:1rem; font-weight:700; color:#818cf8;"> Configuration</div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("**Modèle**")
num_blocks = st.sidebar.slider("Blocs résiduels", 1, 6, 4)
num_channels = st.sidebar.select_slider("Largeur feature maps", [32, 64, 128], value=128)
hidden_dim = st.sidebar.select_slider(
 "GRU hidden dim", [256, 512, 1024, 2048, 4096], value=256,
 help="Le papier utilise 4096, mais nécessite 700 k samples. Pour tests, 256 suffit.",
)
fix_gamma = st.sidebar.checkbox("Fixer γ = 1 (pas de scaling)", value=False)
fix_beta = st.sidebar.checkbox("Fixer β = 0 (pas de biais)", value=False)

st.sidebar.markdown("**Entraînement**")
num_epochs = st.sidebar.slider("Époques", 1, 50, 20)
batch_size = st.sidebar.select_slider("Batch size", [16, 32, 64, 128], value=64)
learning_rate = st.sidebar.select_slider(
 "Learning rate", [1e-4, 3e-4, 5e-4, 1e-3], value=3e-4,
 format_func=lambda x: f"{x:.0e}",
)
max_samples = st.sidebar.slider("Questions d'entraînement", 500, 700_000, 20_000, step=500)

st.sidebar.markdown("**Données**")
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


n_feat_train = _h5_count(Path(data_dir) / "features_train.h5")
n_feat_val = _h5_count(Path(data_dir) / "features_val.h5")
features_ready = n_feat_train >= max_samples

if data_ready:
 if features_ready:
 st.sidebar.success(f" Features prêtes — {n_feat_train:,} train / {n_feat_val:,} val")
 elif n_feat_train > 0:
 st.sidebar.warning(f"Features : seulement {n_feat_train:,} images (besoin : {max_samples:,})")
 else:
 st.sidebar.warning("Features non extraites.")
else:
 st.sidebar.error("CLEVR introuvable au chemin indiqué.")

extract_btn = st.sidebar.button(
 " Extraire features ResNet101",
 disabled=not data_ready or features_ready or st.session_state.training_active,
)
start_btn = st.sidebar.button(
 " Lancer l'entraînement",
 disabled=st.session_state.training_active or not data_ready,
 type="primary",
)

# ── Extraction features ────────────────────────────────────────────────────────
if extract_btn:
 from data.extract_features import extract
 device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 for split, n in [("train", max_samples), ("val", min(max_samples // 5, 9_900))]:
 with st.spinner(f"Extraction features ResNet101 — split {split} ({n:,} images)…"):
 extract(data_dir=data_dir, split=split, max_images=n, batch_size=64, device=device)
 st.success("Extraction terminée ! Vous pouvez lancer l'entraînement.")
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
 st.session_state.progress_queue = pq
 st.session_state.training_active = True
 st.session_state.current_history = None
 st.session_state.last_progress = None
 st.session_state._run_config = config

 def _run_training(cfg, q):
 try:
 history = train(cfg, progress_queue=q)
 history["config"] = cfg.__dict__.copy()
 q.put({"done": True, "history": history})
 except Exception:
 import traceback
 q.put({"error": traceback.format_exc(), "done": True})

 threading.Thread(target=_run_training, args=(config, pq), daemon=True).start()

# ── Polling progression ────────────────────────────────────────────────────────
pq = st.session_state.progress_queue
active = st.session_state.training_active

if active and pq is not None:
 if not st.session_state.last_progress:
 st.info(" Chargement du dataset et initialisation du modèle…")

 latest = None
 while True:
 try:
 latest = pq.get_nowait()
 except queue.Empty:
 break

 if latest:
 if "error" in latest:
 st.error(f"Erreur d'entraînement :\n```\n{latest['error']}\n```")
 st.session_state.training_active = False
 elif "history" in latest:
 st.session_state.current_history = latest["history"]
 st.session_state.runs.append(latest["history"])
 st.session_state.training_active = False
 else:
 st.session_state.last_progress = latest

 time.sleep(0.5)
 st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# État initial — rien en cours
# ══════════════════════════════════════════════════════════════════════════════
prog = st.session_state.last_progress
history = st.session_state.current_history

if not prog and not history and not st.session_state.runs:
 c_info, c_table = st.columns([1, 1], gap="large")

 with c_info:
 st.markdown('<div class="section-header"> Comment démarrer</div>', unsafe_allow_html=True)
 steps = [
 ("1", "Téléchargez CLEVR v1.0 (~18 Go) et indiquez le chemin dans la sidebar.", "#6366f1"),
 ("2", "Cliquez sur **Extraire features ResNet101** (une seule fois, ~20 min GPU).", "#06b6d4"),
 ("3", "Configurez le modèle et cliquez sur **Lancer l'entraînement**.", "#10b981"),
 ("4", "Les courbes, γ/β et prédictions s'affichent automatiquement.", "#f59e0b"),
 ]
 for num, text, color in steps:
 st.markdown(f"""
 <div class="film-card" style="display:flex; align-items:flex-start; gap:0.75rem;
 padding:0.85rem 1rem; margin-bottom:0.5rem;">
 <span style="width:26px; height:26px; border-radius:50%;
 background:{color}22; border:1.5px solid {color};
 color:{color}; font-size:0.78rem; font-weight:700;
 display:inline-flex; align-items:center; justify-content:center;
 flex-shrink:0;">{num}</span>
 <span style="color:#cbd5e1; font-size:0.85rem; line-height:1.6;">{text}</span>
 </div>
 """, unsafe_allow_html=True)

 with c_table:
 st.markdown('<div class="section-header"> Ablations suggérées</div>', unsafe_allow_html=True)
 st.markdown("""
 | Configuration | BN | γ libre | β libre | Blocs |
 |--------------------|----|---------|---------|-------|
 | Baseline FiLM | | | | 4 |
 | Sans BN | | | | 4 |
 | γ fixe = 1 | | | | 4 |
 | β fixe = 0 | | | | 4 |
 | γ et β fixes | | | | 4 |
 | Plus de profondeur | | | | 6 |
 """)

# ══════════════════════════════════════════════════════════════════════════════
# Progression en cours
# ══════════════════════════════════════════════════════════════════════════════
if prog and not history:
 st.markdown('<div class="section-header"> Entraînement en cours</div>', unsafe_allow_html=True)

 epoch = prog.get("epoch", 0)
 n_epochs = prog.get("num_epochs", num_epochs)
 st.progress(epoch / n_epochs, text=f"Époque {epoch}/{n_epochs}")

 m1, m2, m3, m4 = st.columns(4)
 m1.metric("Loss entraîn.", f"{prog['train_loss']:.4f}")
 m2.metric("Acc. entraîn.", f"{prog['train_acc']:.2%}")
 m3.metric("Loss val.", f"{prog['val_loss']:.4f}")
 m4.metric("Acc. val.", f"{prog['val_acc']:.2%}")

# ══════════════════════════════════════════════════════════════════════════════
# Post-entraînement — visualisations
# ══════════════════════════════════════════════════════════════════════════════
if history:
 st.markdown("""
 <div style="background:rgba(16,185,129,0.08); border:1px solid rgba(16,185,129,0.3);
 border-radius:12px; padding:1rem 1.5rem; margin-bottom:1.5rem;
 display:flex; align-items:center; gap:0.75rem;">
 <span style="font-size:1.4rem;"></span>
 <span style="color:#34d399; font-weight:600; font-size:0.95rem;">Entraînement terminé !</span>
 </div>
 """, unsafe_allow_html=True)

 st.progress(1.0, text="Terminé")

 m1, m2, m3 = st.columns(3)
 m1.metric("Précision val. finale", f"{history['val_acc'][-1]:.2%}")
 m2.metric("Loss val. finale", f"{history['val_loss'][-1]:.4f}")
 m3.metric("Époques", len(history["train_loss"]))

 st.divider()

 # Courbes
 st.markdown('<div class="section-header"> Courbes d\'apprentissage</div>', unsafe_allow_html=True)
 st.plotly_chart(plot_training_curves(history), width="stretch")

 # Histogrammes γ/β
 if history.get("film_params"):
 st.markdown('<div class="section-header"> Distributions γ et β</div>', unsafe_allow_html=True)
 st.markdown("""
 <p style="color:#94a3b8; font-size:0.85rem; margin-bottom:1rem;">
 Des γ proches de 1 indiquent peu de modulation ; des valeurs éloignées montrent
 que ce canal est fortement conditionné par la question.
 </p>
 """, unsafe_allow_html=True)
 cfg = history.get("config", {})
 n_blk = cfg.get("num_blocks", num_blocks)
 st.plotly_chart(plot_gamma_beta_hist(history["film_params"], n_blk), width="stretch")

 # Prédictions
 if history.get("sample_predictions"):
 st.markdown('<div class="section-header"> Exemples de prédictions</div>', unsafe_allow_html=True)
 fig = plot_predictions_grid(history["sample_predictions"], cols=5)
 st.pyplot(fig)

 # Feature maps
 st.markdown('<div class="section-header"> Feature Maps avant / après FiLM</div>',
 unsafe_allow_html=True)
 ckpt_path = Path(history.get("run_dir", "runs/default")) / "checkpoint.pt"
 if ckpt_path.exists():
 col_ctrl, _ = st.columns([1, 2])
 with col_ctrl:
 block_idx = st.number_input(
 "Bloc FiLM à visualiser",
 min_value=0,
 max_value=history["config"]["num_blocks"] - 1,
 value=0,
 )
 if st.button(" Afficher les feature maps"):
 from src.model import FiLMModel
 from src.dataset import MiniCLEVR, NUM_ANSWERS

 ckpt = torch.load(ckpt_path, map_location="cpu")
 cfg_d = ckpt["config"]
 device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

 model = FiLMModel(
 vocab_size=cfg_d.get("vocab_size", len(ckpt["vocab"])),
 num_answers=NUM_ANSWERS,
 num_blocks=cfg_d["num_blocks"],
 num_channels=cfg_d["num_channels"],
 embedding_dim=cfg_d["embedding_dim"],
 hidden_dim=cfg_d["hidden_dim"],
 rnn_num_layers=cfg_d.get("rnn_num_layers", 2),
 input_channels=cfg_d.get("input_channels", 1024),
 feature_map_size=cfg_d.get("feature_map_size", 14),
 classifier_proj_dim=cfg_d.get("classifier_proj_dim", 512),
 fix_gamma=cfg_d["fix_gamma"],
 fix_beta=cfg_d["fix_beta"],
 ).to(device)
 model.load_state_dict(ckpt["model_state"])

 val_dataset = MiniCLEVR(
 cfg_d["data_dir"], split="val",
 max_samples=5, vocab=ckpt["vocab"],
 )
 image, question, _, q_str, a_str = val_dataset[0]
 st.markdown(f"""
 <div class="qa-row">
 <span class="qa-q">Question : {q_str}</span>&emsp;
 <span class="qa-a">Réponse : {a_str}</span>
 </div>
 """, unsafe_allow_html=True)
 fig = plot_feature_maps(model, image, question, device, block_idx=int(block_idx))
 st.pyplot(fig)
 else:
 st.info("Checkpoint non trouvé — relancez l'entraînement pour générer les feature maps.")

# ══════════════════════════════════════════════════════════════════════════════
# Tableau de comparaison des runs
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.runs:
 st.divider()
 st.markdown('<div class="section-header"> Comparaison des runs</div>', unsafe_allow_html=True)

 rows = []
 for i, r in enumerate(st.session_state.runs):
 c = r.get("config", {})
 rows.append({
 "Run": i + 1,
 "Blocs": c.get("num_blocks", "?"),
 "Canaux": c.get("num_channels", "?"),
 "γ fixe": "" if c.get("fix_gamma") else "",
 "β fixe": "" if c.get("fix_beta") else "",
 "Époques": len(r.get("train_loss", [])),
 "Acc. val.": f"{r['val_acc'][-1]:.2%}" if r.get("val_acc") else "?",
 "Loss val.": f"{r['val_loss'][-1]:.4f}" if r.get("val_loss") else "?",
 })
 st.dataframe(rows, width="stretch")

 if st.button(" Effacer tous les runs"):
 st.session_state.runs = []
 st.session_state.current_history = None
 st.session_state.last_progress = None
 st.rerun()
