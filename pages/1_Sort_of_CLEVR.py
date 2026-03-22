"""Sort of CLEVR — Entraînement FiLM sur dataset 2D."""

from __future__ import annotations

import queue
import sys
import threading
import time
from pathlib import Path

import numpy as np
import streamlit as st
import torch
from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sortofclevr import CLASSES
from sortofclevr.train import display_image, prepare_objects, run, train_model

# st.set_page_config(page_title="Sort of CLEVR", layout="wide")
st.title("Sort of CLEVR")
st.caption("Dataset Kaggle 2D — étape intermédiaire avant CLEVR 3D")
st.divider()

# ─── Présentation du dataset ───────────────────────────────────────────────────
col_desc, col_img, col_enc = st.columns([2, 1.5, 2.5])

with col_desc:
    st.markdown("""
Sort of CLEVR est un dataset 2D issu de Kaggle
(https://www.kaggle.com/datasets/gruberpatrick/sortofclevr). Chaque image contient
des formes colorées (cercles ou carrés) sur fond blanc. Les questions portent sur trois types :

| Type | Exemple |
|------|---------|
| **Forme** | *What shape is the blue object ?* |
| **Direction** | *What is the position of red relative to green ?* |
| **Couleur** | *What color is the object most to the right ?* |

La réponse appartient à l'une des **11 classes** :
`right`, `left`, `top`, `bottom`, `circle`, `square`, `blue`, `red`, `green`, `yellow`, `gray`.
""")

with col_img:
    example_img = ROOT / "assets" / "example_soc.png"
    if example_img.exists():
        st.image(str(example_img), caption="Exemple d'image du dataset", use_container_width=True)

st.divider()

# ─── Architecture du modèle ────────────────────────────────────────────────────
st.subheader("Notre implémentation vs l'article")

st.markdown("""
| | **Article (CLEVR 3D)** | **Nous (Sort of CLEVR 2D)** |
|---|---|---|
| Image | ResNet-101 pré-entraîné | CNN léger (4 conv stride-2 + BN) |
| Question | GRU sur le texte mot par mot | Vecteur numérique de taille 10 |
| FiLM generator | FC séparée par bloc depuis $h_T$ | Linear 10 → 2×128 dans chaque bloc |
| Formule FiLM | $\\hat{x} = (1 + \\gamma) \\cdot \\text{BN}(x) + \\beta$ | Identique |
| Coord maps | Oui | Oui |
| Blocs résiduels | 4 | 4 |
| Classifieur | MLP après Global Average Pooling | MLP après Global Max Pooling |
""")

st.divider()

# ─── Chemins des données ───────────────────────────────────────────────────────
DATA_DIR  = Path("./sortofclevr/data")
train_h5  = DATA_DIR / "data_train.h5"
train_csv = DATA_DIR / "data_train.csv"
val_h5    = DATA_DIR / "data_val.h5"
val_csv   = DATA_DIR / "data_val.csv"
test_h5   = DATA_DIR / "data_test.h5"
test_csv  = DATA_DIR / "data_test.csv"

_EXPECTED_FILES = [
    "data_train.h5", "data_train.csv",
    "data_val.h5",   "data_val.csv",
    "data_test.h5",  "data_test.csv",
    "model_weights.pth",
]

if not (train_h5.exists() and train_csv.exists() and test_h5.exists() and test_csv.exists()):
    st.warning("Données introuvables dans `sortofclevr/data/`.")
    if st.button("Télécharger les données depuis Google Drive"):
        import gdown

        err: list = []

        def _download():
            try:
                gdown.download_folder(
                    id="1R5zFO73ABA0zn5TxvWKm_JeG0iq8WX6t",
                    output=str(ROOT / "sortofclevr" / "data"),
                    quiet=False,
                    use_cookies=False,
                    remaining_ok=True,
                )
            except Exception as e:
                err.append(e)

        t = threading.Thread(target=_download, daemon=True)
        t.start()

        bar  = st.progress(0.0)
        info = st.empty()
        while t.is_alive():
            found = sum(1 for f in _EXPECTED_FILES if (DATA_DIR / f).exists())
            bar.progress(found / len(_EXPECTED_FILES))
            info.text(f"Téléchargement... {found}/{len(_EXPECTED_FILES)} fichiers reçus")
            time.sleep(1)
        t.join()

        if err:
            st.error(f"Erreur lors du téléchargement : {err[0]}")
        else:
            bar.progress(1.0)
            info.text("Téléchargement terminé.")
            st.rerun()
    st.stop()

st.success("Données détectées.")
st.divider()

# ─── Hyperparamètres ───────────────────────────────────────────────────────────
use_pretrained = st.checkbox("Utiliser un modèle pré-entraîné", value=False)
button_label   = "Lancer l'évaluation du modèle pré-entraîné" if use_pretrained else "Lancer l'entraînement"

if not use_pretrained:
    n_epochs    = st.slider("Epochs",                     1,  50,     10)
    batch_sz    = st.slider("Batch size",                32, 512,    128, step=32)
    lr          = st.number_input("Learning rate", value=0.001, format="%.4f")
    max_samples = st.slider("Samples d'entraînement", 1000, 20000, 5000, step=1000)
else:
    n_epochs, batch_sz, lr = 1, 512, 1e-3
    max_samples = st.slider("Samples de test", 1000, 20000, 10000, step=1000)

# ─── Préparation modèle & dataloaders ─────────────────────────────────────────
model, train_loader, val_loader, test_loader, device = prepare_objects(
    train_h5=train_h5, train_csv=train_csv,
    val_h5=val_h5,     val_csv=val_csv,
    test_h5=test_h5,   test_csv=test_csv,
    batch_size=batch_sz, max_samples=max_samples,
)

if not use_pretrained:
    if str(device) == "cpu":
        import platform
        cpu_name = platform.processor() or platform.machine()
        st.info(f"Calculs effectués sur : **{cpu_name}** (CPU)")
    else:
        st.info(f"Calculs effectués sur : **{torch.cuda.get_device_name(device)}** (GPU)")

if "modele_entraine" not in st.session_state:
    st.session_state.modele_entraine = None

# Chargement immédiat du modèle pré-entraîné
if use_pretrained:
    weights_path = DATA_DIR / "model_weights.pth"
    if not weights_path.exists():
        st.error("Fichier `sortofclevr/data/model_weights.pth` introuvable.")
        st.stop()

    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    st.session_state.modele_entraine = model
    st.session_state.test_loader     = test_loader
    st.session_state.device          = device

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    history   = train_model(
        model, train_loader, val_loader, optimizer, criterion,
        device, epochs=n_epochs, pretrain=True,
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
    ma_queue           = queue.Queue()
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
    texte_statut.text("Entraînement en cours...")
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
            continue

    thread.join()
    barre_progression.progress(1.0)
    texte_statut.text("Terminé.")

    st.session_state.modele_entraine = model
    st.session_state.test_loader     = test_loader
    st.session_state.device          = device

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
        rows = [{"Classe": c, "Accuracy": f"{acc:.1%}"} for c, acc in per_class.items()]
        rows.sort(key=lambda r: r["Accuracy"], reverse=True)
        st.dataframe(rows, width='stretch')

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
            st.image(img.permute(1, 2, 0).cpu().numpy(), caption="Image de test",
                     width='stretch')
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