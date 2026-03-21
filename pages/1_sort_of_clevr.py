"""Sort of CLEVR — Entraînement FiLM sur dataset 2D."""

from __future__ import annotations

import queue
import sys
import threading
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
st.title("Sort of CLEVR")
st.caption("Dataset Kaggle 2D — étape intermédiaire avant CLEVR 3D")
st.divider()

# ─── Chemins des données ───────────────────────────────────────────────────────
DATA_DIR  = Path("./sortofclevr")
train_h5  = DATA_DIR / "data_train.h5"
train_csv = DATA_DIR / "data_train.csv"
val_h5    = DATA_DIR / "data_val.h5"
val_csv   = DATA_DIR / "data_val.csv"
test_h5   = DATA_DIR / "data_test.h5"
test_csv  = DATA_DIR / "data_test.csv"

if not (train_h5.exists() and train_csv.exists() and test_h5.exists() and test_csv.exists()):
    st.warning("Données introuvables dans `sortofclevr/`.")
    if st.button("Télécharger les données depuis Google Drive"):
        import gdown
        with st.spinner("Téléchargement en cours..."):
            try:
                gdown.download_folder(
                    id="1R5zFO73ABA0zn5TxvWKm_JeG0iq8WX6t",
                    output=str(ROOT),
                    quiet=True,
                    use_cookies=False,
                )
                st.success("Téléchargement terminé.")
                st.rerun()
            except Exception as e:
                st.error(f"Erreur lors du téléchargement : {e}")
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

if "modele_entraine" not in st.session_state:
    st.session_state.modele_entraine = None

# Chargement immédiat du modèle pré-entraîné
if use_pretrained:
    weights_path = DATA_DIR / "model_weights.pth"
    if not weights_path.exists():
        st.error("Fichier `sortofclevr/model_weights.pth` introuvable.")
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
            st.image(img.permute(1, 2, 0).cpu().numpy(), caption="Image de test",
                     use_container_width=True)
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
