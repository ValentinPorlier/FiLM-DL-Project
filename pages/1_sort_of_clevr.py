"""Sort of CLEVR — Entraînement FiLM sur dataset 2D."""

import queue
import sys
from pathlib import Path

import streamlit as st

import threading

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sortofclevr.train import run

st.set_page_config(page_title="Sort of CLEVR", layout="wide")
st.title("Sort of CLEVR")
st.divider()

data_dir = "./sortofclevr"

train_h5  = Path(data_dir) / "data_train.h5"
train_csv = Path(data_dir) / "data_train.csv"
val_h5    = Path(data_dir) / "data_val.h5"
val_csv   = Path(data_dir) / "data_val.csv"
test_h5   = Path(data_dir) / "data_test.h5"
test_csv  = Path(data_dir) / "data_test.csv"

if not data_dir:
    st.stop()

if not (train_h5.exists() and train_csv.exists() and test_h5.exists() and test_csv.exists()):
    st.warning("Fichiers introuvables dans ce dossier.")
    st.stop()

st.success("Données détectées")


use_pretrained = st.checkbox("Utiliser un modèle pré-entraîné", value=False)
button_label = "Lancer l'evaluation du modèle pré-entraîné" if use_pretrained else "Lancer l'entraînement"


if not use_pretrained:
    n_epochs    = st.slider("Epochs", 1, 50, 10)
    batch_sz    = st.slider("Batch size", 32, 512, 128, step=32)
    lr          = st.number_input("Learning rate", value=0.001, format="%.4f")
    max_samples = st.slider("Samples d'entraînement", 1000, 70000, 10000, step=1000)
else:
    n_epochs, batch_sz, lr, max_samples = 1, 512, 1e-3, 1000 #valeurs de bases pour que la fonction se lance

if st.button(button_label):


    
    ma_queue = queue.Queue()
    parametres_run = {
    "train_h5": train_h5,
    "train_csv": train_csv,
    "val_h5": val_h5,
    "val_csv": val_csv,
    "test_h5": test_h5,
    "test_csv": test_csv,
    "epochs": n_epochs,
    "batch_size": batch_sz,
    "lr": lr,
    "max_samples": max_samples,
    "pretrain": use_pretrained,
    "progress_queue": ma_queue  # On l'ajoute pour le suivi Streamlit
    }
    
    # 3. On lance l'entraînement (via run qui appelle train_model)
    # Note: Idéalement dans un thread comme on a vu avant pour ne pas figer l'UI
    import threading
    thread = threading.Thread(target=run, kwargs=parametres_run)
    thread.start()

    barre_progression = st.progress(0)
    texte_statut = st.empty()


    while thread.is_alive() or not ma_queue.empty():
        try:
            infos = ma_queue.get(timeout=0.5)
            #Ici je veux afficher les epochs si epochs, les batches quand batches

            #Partie train: en premier si epoch
            if "epoch" in infos:
                progress = infos["epoch"] / infos["num_epochs"]
                barre_progression.progress(progress)
                texte_statut.text(f"Entraînement : epoch {infos['epoch']}/{infos['num_epochs']} - Train Acc: {infos['train_acc']:.2%} - Val Acc: {infos['val_acc']:.2%}")
            

            #Partie eval: batch (le premier si pretrained)
            elif "batch" in infos:
                progress = infos["batch"] / infos["batch_tot"]
                barre_progression.progress(progress)
                texte_statut.text(f"📊 Évaluation en cours : Batch {infos['batch']}/{infos['batch_tot']}...")
                if infos["batch"] >= infos["batch_tot"]*0.8:
                    texte_statut.text(f"📊 Évaluation en cours : Batch {infos['batch']}/{infos['batch_tot']}... (peu prendre du temps sur la fin)")

            if "history" in infos:
                history = infos["history"]
            if "per_class" in infos:
                per_class = infos["per_class"]

        except queue.Empty:
            continue
        
        


    """with st.spinner("Entraînement en cours... (progression dans le terminal)"):
        history, per_class = run(train_h5, train_csv, test_h5, test_csv,
                                 epochs=n_epochs, batch_size=batch_sz, lr=lr,
                                 max_samples=max_samples, pretrain=use_pretrained)"""

    st.subheader("Résultats")
    c1, c2, c3 = st.columns(3)
    c1.metric("Meilleure Val Acc", f"{max(history['val_acc']):.1%}")
    c2.metric("Val Acc finale",    f"{history['val_acc'][-1]:.1%}")
    c3.metric("Train Acc finale",  f"{history['train_acc'][-1]:.1%}")

    st.line_chart({"Train Loss": history["train_loss"], "Val Loss": history["val_loss"]}, x_label="Epoch", y_label="Loss")
    st.line_chart({"Train Acc":  history["train_acc"],  "Val Acc":  history["val_acc"]}, x_label="Epoch", y_label="Accuracy")

    st.subheader("Accuracy par classe")
    rows = []
    for classe, acc in per_class.items():
        rows.append({"Classe": classe, "Accuracy": f"{acc:.1%}"})
    rows.sort(key=lambda r: r["Accuracy"], reverse=True)


    st.dataframe(rows, use_container_width=True)


if st.button("Reset"):
    # On vide la mémoire des widgets
    for key in st.session_state.keys():
        del st.session_state[key]
    # On relance
    st.rerun()

