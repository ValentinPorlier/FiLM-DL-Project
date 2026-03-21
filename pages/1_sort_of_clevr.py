"""Sort of CLEVR — Entraînement FiLM sur dataset 2D."""

from __future__ import annotations

import queue
import sys
import threading
import time
from pathlib import Path

import streamlit as st
import torch
from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sortofclevr import CLASSES
from sortofclevr.train import display_image, prepare_objects, run, train_model

st.set_page_config(page_title="Sort of CLEVR", layout="wide")

# ─── Lien de téléchargement du dataset ────────────────────────────────────────
# Remplacez cette URL par votre lien Kaggle / Google Drive / etc.
DATASET_URL = "https://www.kaggle.com/datasets/VOTRE_LIEN_ICI"

st.title("Sort of CLEVR")
st.caption("Dataset Kaggle 2D — étape intermédiaire avant CLEVR 3D")
st.divider()

# ─── Données ──────────────────────────────────────────────────────────────────
st.header("Données")

DATA_DIR  = Path("./sortofclevr")
train_h5  = DATA_DIR / "data_train.h5"
train_csv = DATA_DIR / "data_train.csv"
val_h5    = DATA_DIR / "data_val.h5"
val_csv   = DATA_DIR / "data_val.csv"
test_h5   = DATA_DIR / "data_test.h5"
test_csv  = DATA_DIR / "data_test.csv"

data_ok = all(
    p.exists() for p in [train_h5, train_csv, val_h5, val_csv, test_h5, test_csv]
)

if data_ok:
    total_mb = sum(
        p.stat().st_size for p in [train_h5, train_csv, val_h5, val_csv, test_h5, test_csv]
    ) / 1e6
    st.success(f"Données détectées ({total_mb:.1f} Mo).")
else:
    st.warning(
        "Les fichiers du dataset sont introuvables dans `sortofclevr/`. "
        "Téléchargez le dataset Kaggle et placez les fichiers dans ce dossier."
    )
    col_btn, col_info = st.columns([1, 3])
    with col_btn:
        st.link_button("Télécharger le dataset (Kaggle)", url=DATASET_URL)
    with col_info:
        st.caption(
            "Le dossier `sortofclevr/` doit contenir : "
            "`data_train.h5`, `data_train.csv`, `data_val.h5`, `data_val.csv`, "
            "`data_test.h5`, `data_test.csv`."
        )
    st.stop()

st.divider()

# ─── Principe ─────────────────────────────────────────────────────────────────
st.header("Principe")

col_desc, col_math = st.columns([3, 2])
with col_desc:
    st.markdown("""
**Sort of CLEVR** est un dataset 2D issu de Kaggle, utilisé comme étape
intermédiaire avant d'attaquer CLEVR 3D. Chaque image contient des
**formes colorées** (cercles ou carrés) sur fond blanc ; le modèle répond à
des questions sur leurs propriétés et leurs relations spatiales.

C'est un excellent banc de test pour valider l'architecture FiLM rapidement :
images légères, entraînement en quelques minutes sur CPU, sans features
pré-extraites (ResNet101 non requis ici).

Trois types de questions :
- **Forme** — *What shape is the blue object?* → `circle` / `square`
- **Direction** — *What is the position of red relative to green?* → `right` / `left` / `top` / `bottom`
- **Couleur** — *What color is the object most to the right?* → `blue` / `yellow` / …
""")

with col_math:
    st.markdown("**Conditionnement FiLM par bloc résiduel**")
    st.latex(r"\text{FiLM}(x) = \gamma(\text{question}) \cdot x + \beta(\text{question})")
    st.markdown("""
La question (vecteur 10-dim) est projetée linéairement en (γ, β) pour chaque
bloc résiduel. Le CNN visuel est ainsi **modulé dynamiquement** par la question,
sans partager de paramètres entre les blocs.
""")

col_f, col_l = st.columns(2)
with col_f:
    st.markdown("""
**Forces**
- Dataset léger, entraînement rapide (< 5 min CPU)
- Valide toute l'architecture FiLM end-to-end
- Étape pédagogique avant CLEVR 3D
""")
with col_l:
    st.markdown("""
**Limites**
- Images 2D très simplifiées, non photoréalistes
- 11 classes de réponse seulement
- Ne teste pas ResNet101 ni le GRU profond
""")

with st.expander("Encodage de question (10 dims)"):
    st.markdown("""
| Dims | Contenu |
|------|---------|
| 0–2  | Type de question one-hot : shape / direction / color_at_pos |
| 3–7  | Couleur de l'objet 1 (one-hot, 5 couleurs) |
| 8    | Couleur de l'objet 2 normalisée ∈ [0, 1] (pour direction) |
| 9    | Direction normalisée ∈ [0, 1] (pour color_at_pos) |
""")

st.divider()

# ─── Entraînement ─────────────────────────────────────────────────────────────
st.header("Entraînement")

use_pretrained = st.checkbox("Utiliser un modèle pré-entraîné", value=False)
button_label = (
    "Lancer l'évaluation du modèle pré-entraîné"
    if use_pretrained
    else "Lancer l'entraînement"
)

if not use_pretrained:
    n_epochs    = st.slider("Epochs",                       1,  50,     10)
    batch_sz    = st.slider("Batch size",                  32, 512,    128, step=32)
    lr          = st.number_input("Learning rate", value=0.001, format="%.4f")
    max_samples = st.slider("Samples d'entraînement",   1000, 20_000, 5_000, step=1000)
else:
    n_epochs, batch_sz, lr = 1, 512, 1e-3
    max_samples = st.slider("Samples de test (modèle pré-entraîné)", 1000, 20_000, 10_000, step=1000)

parametres_prepare = {
    "train_h5":    train_h5,
    "train_csv":   train_csv,
    "val_h5":      val_h5,
    "val_csv":     val_csv,
    "test_h5":     test_h5,
    "test_csv":    test_csv,
    "batch_size":  batch_sz,
    "max_samples": max_samples,
}

model, train_loader, val_loader, test_loader, device = prepare_objects(**parametres_prepare)

if "modele_entraine" not in st.session_state:
    st.session_state.modele_entraine = None

# Si modèle pré-entraîné : affichage immédiat des résultats
if use_pretrained:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights_path = Path("sortofclevr/model_weights.pth")
    if not weights_path.exists():
        st.error("Fichier `sortofclevr/model_weights.pth` introuvable.")
        st.stop()
    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    st.session_state.modele_entraine = model
    st.session_state.test_loader = test_loader
    st.session_state.device = device

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    history = train_model(
        model, train_loader, val_loader, optimizer, criterion, device,
        epochs=n_epochs, pretrain=True,
    )

    st.subheader("Résultats du modèle pré-entraîné")
    c1, c2, c3 = st.columns(3)
    c1.metric("Meilleure Val Acc", f"{max(history['val_acc']):.1%}")
    c2.metric("Val Acc finale",    f"{history['val_acc'][-1]:.1%}")
    c3.metric("Train Acc finale",  f"{history['train_acc'][-1]:.1%}")
    st.line_chart({"Train Loss": history["train_loss"], "Val Loss": history["val_loss"]},
                  x_label="Epoch", y_label="Loss")
    st.line_chart({"Train Acc": history["train_acc"], "Val Acc": history["val_acc"]},
                  x_label="Epoch", y_label="Accuracy")

# ─── Bouton lancement ─────────────────────────────────────────────────────────
if st.button(button_label):
    ma_queue          = queue.Queue()
    progress_container = st.container()

    parametres_run = {
        "model":          model,
        "train_loader":   train_loader,
        "val_loader":     val_loader,
        "test_loader":    test_loader,
        "device":         device,
        "epochs":         n_epochs,
        "lr":             lr,
        "pretrain":       use_pretrained,
        "progress_queue": ma_queue,
        "st_container":   progress_container,
    }

    thread = threading.Thread(target=run, kwargs=parametres_run, daemon=False)
    add_script_run_ctx(thread, get_script_run_ctx())
    thread.start()

    barre_progression = st.progress(0.0)
    texte_statut      = st.empty()
    history   = None
    per_class = None

    while thread.is_alive() or not ma_queue.empty():
        try:
            infos = ma_queue.get(timeout=0.1)

            if "epoch" in infos:
                progress = infos["epoch"] / infos["num_epochs"]
                barre_progression.progress(progress)
                texte_statut.text(
                    f"Entraînement : epoch {infos['epoch']}/{infos['num_epochs']} "
                    f"— Train Acc: {infos['train_acc']:.2%} — Val Acc: {infos['val_acc']:.2%}"
                )
            elif "batch" in infos:
                progress = infos["batch"] / infos["batch_tot"]
                barre_progression.progress(progress)
                texte_statut.text(
                    f"Évaluation en cours : batch {infos['batch']}/{infos['batch_tot']}..."
                )

            if "history" in infos:
                history = infos["history"]
            if "per_class" in infos:
                per_class = infos["per_class"]

        except queue.Empty:
            time.sleep(0.05)

    thread.join()
    barre_progression.progress(1.0)
    texte_statut.text("Terminé.")

    st.session_state.modele_entraine = model
    st.session_state.test_loader = test_loader
    st.session_state.device = device

    if not use_pretrained and history is not None:
        st.subheader("Résultats")
        c1, c2, c3 = st.columns(3)
        c1.metric("Meilleure Val Acc", f"{max(history['val_acc']):.1%}")
        c2.metric("Val Acc finale",    f"{history['val_acc'][-1]:.1%}")
        c3.metric("Train Acc finale",  f"{history['train_acc'][-1]:.1%}")
        st.line_chart({"Train Loss": history["train_loss"], "Val Loss": history["val_loss"]},
                      x_label="Epoch", y_label="Loss")
        st.line_chart({"Train Acc": history["train_acc"], "Val Acc": history["val_acc"]},
                      x_label="Epoch", y_label="Accuracy")

    if per_class is not None:
        st.subheader("Accuracy par classe")
        rows = [{"Classe": c, "Accuracy": f"{a:.1%}"} for c, a in per_class.items()]
        rows.sort(key=lambda r: r["Accuracy"], reverse=True)
        st.dataframe(rows, use_container_width=True)

# ─── Test visuel ──────────────────────────────────────────────────────────────
if st.session_state.modele_entraine is not None:
    st.divider()
    st.subheader("Test visuel du modèle")

    if st.button("Charger image"):
        model       = st.session_state.modele_entraine
        test_loader = st.session_state.test_loader
        device      = st.session_state.device
        st.session_state.img_data = display_image(model, test_loader, device)

    if "img_data" in st.session_state:
        model  = st.session_state.modele_entraine
        device = st.session_state.device
        img, questions, encodings = st.session_state.img_data

        col1, col2 = st.columns(2)
        with col1:
            img_np = img.permute(1, 2, 0).cpu().numpy()
            st.image(img_np, caption="Image de test", use_container_width=True)
        with col2:
            index_choisi = st.selectbox(
                "Choisir une question",
                options=list(range(len(questions))),
                format_func=lambda i: questions[i],
            )

        model.eval()
        with torch.no_grad():
            img_in  = img.unsqueeze(0).to(device)
            ques_in = encodings[index_choisi].unsqueeze(0).to(device)
            output  = model(img_in, ques_in)

        st.write(f"Question : {questions[index_choisi]}")
        st.success(f"Réponse prédite : {CLASSES[output.argmax().item()]}")

if st.button("Reset"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()
