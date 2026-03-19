"""Sort of CLEVR — Entraînement FiLM sur dataset 2D."""

import queue
import sys
from pathlib import Path

import streamlit as st

import threading

import torch
import numpy as np

from sortofclevr import CLASSES

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sortofclevr.train import display_image, run, prepare_objects, train_model

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
    max_samples = st.slider("Samples d'entraînement", 1000, 20000, 5000, step=1000)
else:
    n_epochs, batch_sz, lr = 1, 512, 1e-3 #valeurs de bases pour que la fonction se lance
    max_samples = st.slider("Samples de test (pour le modèle pré-entraîné)", 1000, 20000, 10000, step=1000)

#On défini le modele et les dataloaders pour pouvoir les reutiliser
parametres_prepare = {
    "train_h5": train_h5,
    "train_csv": train_csv,
    "val_h5": val_h5,
    "val_csv": val_csv,
    "test_h5": test_h5,
    "test_csv": test_csv,
    "batch_size": batch_sz,
    "max_samples": max_samples
    }

model, train_loader, val_loader, test_loader, device = prepare_objects(**parametres_prepare)
#pour garder le modele apres
if "modele_entraine" not in st.session_state:
    st.session_state.modele_entraine = None

#Si on utilise pretrained on met le modele dans l'etat de la session pour afficher des images sans avoir à réentraîner à chaque fois
if use_pretrained:
    model, train_loader, val_loader, test_loader, device = prepare_objects(**parametres_prepare)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dict = torch.load("sortofclevr/model_weights.pth", map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    st.session_state.modele_entraine = model
    st.session_state.test_loader = test_loader # On garde aussi le loader pour l'éval
    st.session_state.device = device

    #On affiche directement les résultats d'entrainement pour le modele pré-entrainé
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    history = train_model(model, train_loader, val_loader, optimizer, criterion,device, epochs=n_epochs, pretrain=True)    
    st.subheader("Résultats")
    c1, c2, c3 = st.columns(3)
    c1.metric("Meilleure Val Acc", f"{max(history['val_acc']):.1%}")
    c2.metric("Val Acc finale",    f"{history['val_acc'][-1]:.1%}")
    c3.metric("Train Acc finale",  f"{history['train_acc'][-1]:.1%}")

    st.line_chart({"Train Loss": history["train_loss"], "Val Loss": history["val_loss"]}, x_label="Epoch", y_label="Loss")
    st.line_chart({"Train Acc":  history["train_acc"],  "Val Acc":  history["val_acc"]}, x_label="Epoch", y_label="Accuracy")



if st.button(button_label):


    
    ma_queue = queue.Queue()
    parametres_run = {
    "model": model,
    "train_loader": train_loader,
    "val_loader": val_loader,
    "test_loader": test_loader,
    "device": device,
    "epochs": n_epochs,
    "lr": lr,
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
            infos = ma_queue.get(timeout=.1)
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

            if "history" in infos:
                history = infos["history"]
            if "per_class" in infos:
                per_class = infos["per_class"]

        except queue.Empty:
            continue
    barre_progression.progress(1.0)
    texte_statut.text(f"Terminé")


    st.session_state.modele_entraine = model
    st.session_state.test_loader = test_loader # On garde aussi le loader pour l'éval
    st.session_state.device = device
        


    #with st.spinner("Entraînement en cours... (progression dans le terminal)"):
    #    history, per_class = run(train_h5, train_csv, test_h5, test_csv,
    #                             epochs=n_epochs, batch_size=batch_sz, lr=lr,
    #                             max_samples=max_samples, pretrain=use_pretrained)
    if not use_pretrained:
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



######
# Affichage image
if st.session_state.modele_entraine is not None:
    st.divider()
    st.subheader("Test visuel du modèle")

    if st.button("Charger image"):
        #on récupère le modèle et les données de test depuis la session state
        model = st.session_state.modele_entraine
        test_loader = st.session_state.test_loader
        device = st.session_state.device
        st.session_state.img_data = display_image(model, test_loader, device)
        print("Image chargée dans la session state")

    if "img_data" in st.session_state:
        #lorsque les images sont chargées, on affiche l'image et les questions, et on permet de choisir une question pour afficher la réponse prédite
        model = st.session_state.modele_entraine
        device = st.session_state.device
        img, questions, encodings = st.session_state.img_data
 
        col1, col2 = st.columns(2)

        with col1:
            img_np = img.permute(1, 2, 0).cpu().numpy()
            st.image(img_np, caption="Image de Test", use_container_width=True)

        with col2:
            indices = list(range(len(questions)))
            index_choisi = st.selectbox(
                "Choisir une question", 
                options=indices, 
                format_func=lambda i: questions[i]
            )

        model.eval()
        with torch.no_grad():
            
            img_in = img.unsqueeze(0).to(device)
            ques_in = encodings[index_choisi].unsqueeze(0).to(device)
                        
            output = model(img_in, ques_in)
            st.write(f"Question : {questions[index_choisi]}")
            st.success(f"Réponse prédite : {CLASSES[output.argmax().item()]}")

        


if st.button("Reset"):
    #On vide la mémoire des widgets
    for key in st.session_state.keys():
        del st.session_state[key]
    #On relance
    st.rerun()

