"""Streamlit page: Style Transfer — Conditional Instance Normalisation."""

from __future__ import annotations

import numpy as np
import os
import io
import sys
import threading
import time
from pathlib import Path
import gdown
import zipfile
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from style_transfert.train import charger_image_aleatoire, train_model_styletransfer, prepare_styletransfer_modele, preparer_pour_plot
from style_transfert.modele import StyleTransferNetwork

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


st.set_page_config(
    page_title="Style Transfer — FiLM Explorer",
    page_icon="",
    layout="wide",
    #initial_sidebar_state="collapsed",
)


#data_dir = st.text_input(
#    "Répertoire Images (dossier téléchargé: style_transfert_data)",
#    value="C:/Users/jeejd/Documents/Vscode/FiLM-DL-Project/style_transfert_data",
#)
default_data_dir = ROOT / "style_transfert_data"

# 2. On utilise ce chemin par défaut dans le text_input
# (On convertit en str car text_input attend du texte)
data_dir = st.text_input(
    "Répertoire Images",
    value=str(default_data_dir)
)
DOSSIER_IMG = os.path.join(data_dir, "10k_img_resized")
DOSSIER_STYLE = os.path.join(data_dir, "img_style_resized")
CHEMIN_POIDS = os.path.join(data_dir, "StyleTransfer_weights.pth")





st.header("Conditional Instance Normalisation — le même principe que FiLM appliqué au style artistique.")
st.write("Le transfert de style que nous implémentons dans cette section se base sur l'article de [Ghiasi et al. (2017)](https://arxiv.org/pdf/1705.06830). " \
"Le modèle prend en entrée une image et un style puis génère la même image d'entrée avec le style à appliquer")

st.header("Architecture")
st.write("Le modèle de transfert de style utilise un 'FiLM Generator' pour générer des paramètres $\gamma$ et $\ beta$ à partir de l'image de style. Ces paramètres" \
"seront insérés dans le modèle sous forme de transformation affine avec les features maps issus l'image d'entrée ")
st.subheader("FiLM Generator")
st.write("Pour générer ces paramètres")

st.divider()


st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)


STYLE_NAMES = ["baroque", "Contemporary", "Cubism", "Early_Renaissance", "Impressionism", "Ukiyo_e"]

# ─── Répertoire des données ────────────────────────────────────────────────────
data_dir = st.text_input(
    "Répertoire du dataset (dossier style_transfer_data/)",
    value="./style_transfer_data",
)

DOSSIER_IMG   = str(Path(data_dir) / "10k_img_resized")
DOSSIER_STYLE = str(Path(data_dir) / "img_style_resized")
CHEMIN_POIDS  = str(Path(data_dir) / "StyleTransfer_weights.pth")

_EXPECTED_FILES = [
    "StyleTransfer_weights.pth",
    "10k_img_resized",
    "img_style_resized",
]

if not Path(data_dir).exists():
    st.warning(f"Données introuvables dans `{data_dir}`.")
    if st.button("Télécharger les données depuis Google Drive"):

        err: list = []

        def _download():
            try:
                gdown.download_folder(
                    id="1Lri1gwXKmcKB0xv_-qXeUIUlqoo9Lbom",
                    output=str(ROOT / "style_transfer_data.zip"),
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
            found = sum(1 for f in _EXPECTED_FILES if (Path(data_dir) / f).exists())
            bar.progress(found / len(_EXPECTED_FILES))
            info.text(f"Téléchargement... {found}/{len(_EXPECTED_FILES)} éléments reçus")
            time.sleep(1)
        t.join()

        if err:
            st.error(f"Erreur lors du téléchargement : {err[0]}")
        else:
            bar.progress(1.0)
            info.text("Téléchargement terminé.")
            st.rerun()
    st.stop()

st.title("Style Transfer")
st.caption("Conditional Instance Normalisation — le même principe que FiLM appliqué au style artistique")
st.divider()

# ─── Principe CIN ─────────────────────────────────────────────────────────────
st.header("Principe")
col_text, col_math = st.columns([3, 2])

with col_text:
    st.markdown("""
**Conditional Instance Normalisation (CIN)** (Dumoulin et al., 2017) reprend
exactement l'idée de FiLM : au lieu d'apprendre un seul jeu de paramètres
d'Instance Normalisation, un couple **(γₛ, βₛ)** distinct est prédit pour
chaque image de style **s** via un encodeur Inception.

Changer de style revient simplement à passer une image de style différente —
sans modifier l'architecture du réseau de transfert.
""")

with col_math:
    st.markdown("**Formulation**")
    st.latex(r"\text{CIN}(x, s) = \gamma_s \cdot \frac{x - \mu}{\sigma} + \beta_s")
    st.markdown("""
| Symbole | Rôle |
|---------|------|
| **x** | feature map (après Instance Norm) |
| **γₛ, βₛ** | paramètres prédits par InceptionMixed6e |
| **s** | image de style |
""")

st.divider()

# ─── Interface ────────────────────────────────────────────────────────────────
entrained = st.checkbox("Utiliser un modèle pré-entraîné", value=False)

if not entrained:
    st.warning("L'entraînement du modele est long (10min par epoch voire plus). Il est préférable de prendre le modèle pré-entrainé même si les résultats ne sont pas forcément satisfaisants")
    n_epochs    = st.slider("Epochs", 1, 50, 10)
    batch_sz    = st.slider("Batch size", 32, 512, 128, step=32)
    lr          = st.number_input("Learning rate", value=0.001, format="%.4f")
    lambda_style = st.select_slider("Poids de la loss du style", options=[1e4, 1e5, 1e6], format_func = lambda x: f"{x:.0e}")

    if st.button("lancer l'entrainement du modele"):
        try:
            model, dataloader, device = prepare_styletransfer_modele(data_dir, batch_size=batch_sz)
            st.success("Modèle bien chargé")
        except Exception as e:
            st.write("Erreur dans le chargement du modele")
            print("Erreur dans le chargement du modele")

        st.write("entrainement du modèle...")
        train_model_styletransfer(model=model, dataloader=dataloader,device = device, epochs=n_epochs, lr=lr, lambda_style=lambda_style)
        entrained = True
        
else:
    model, dataloader, device = prepare_styletransfer_modele(data_dir, batch_size=128)

if entrained:
    model.load_state_dict(torch.load(CHEMIN_POIDS, map_location=device, weights_only=True))
    print("Poids du modèle chargés avec succès !")

    col_upload, col_style = st.columns([2, 1], gap="large")

    with col_upload:
        st.markdown('Image aléatoire ou bien uploadez votre image', unsafe_allow_html=True)
        uploaded = st.file_uploader("Uploadez votre image", type=["png", "jpg", "jpeg"])
        if st.button("Image Random"):
            uploaded, nom_content = charger_image_aleatoire(Path(DOSSIER_IMG) / "images")

        
    
    with col_style:
        st.markdown('<div class="section-header"> Choix du style</div>', unsafe_allow_html=True)
        style_choice = st.radio("Style artistique", STYLE_NAMES)
        style_idx = STYLE_NAMES.index(style_choice)

    st.divider()

    if uploaded is not None and style_choice is not None:
        if not torch.is_tensor(uploaded):
            img_pil = Image.open(uploaded).convert('RGB')

            transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
            ])

            content_tensor = transform(img_pil)
            content_tensor = content_tensor.unsqueeze(0)
        else:
            content_tensor = uploaded
            
        style_tensor_img, nom_style = charger_image_aleatoire(Path(DOSSIER_STYLE) / "images", style=style_choice)

        
        #Contournement du Bug de Dimension (Batch de 2)
        #On duplique les images pour créer un batch de 2 [2, 3, 256, 256]
        content_img = torch.cat([content_tensor, content_tensor], dim=0).to(device)
        style_tensor = torch.cat([style_tensor_img, style_tensor_img], dim=0).to(device)


        with torch.no_grad():
            output = model(content_img, style_tensor)  # (2, 3, H, W)


        output_np = preparer_pour_plot(output[0])
        content_img_np = preparer_pour_plot(content_tensor)
        style_tensor_np = preparer_pour_plot(style_tensor_img)


        st.divider()
        col_cont, col_styl, col_out = st.columns(3, gap="large")

        with col_cont:
            st.markdown('<div class="section-header" style="font-size:1rem;"> Image originale</div>',
                        unsafe_allow_html=True)
            st.image(content_img_np, use_container_width=True)

        
        with col_styl:
            st.markdown('<div class="section-header" style="font-size:1rem;"> Image de style</div>',
                        unsafe_allow_html=True)
            st.image(style_tensor_np, use_container_width=True)

        with col_out:
            st.markdown(f'<div class="section-header" style="font-size:1rem;"> Style : {style_choice}</div>',
                        unsafe_allow_html=True)
            st.image(output_np, caption="Image stylisée", use_container_width=True)

            #pour le téléchargement
            img_pour_pil = output_np.copy()
            img_pour_pil = (img_pour_pil * 255).astype(np.uint8)
            #on met l'image en format pil et on met dans la mémoire
            with io.BytesIO() as buf:
                image_pil = Image.fromarray(img_pour_pil)
                image_pil.save(buf, format="PNG")
                donnees_binaires = buf.getvalue()

            st.download_button(
                label="Télécharger l'image générée",
                data=donnees_binaires,
                file_name=f"style_transfert_{style_choice}.png",
                mime="image/png" #Indique au navigateur que c'est une image PNG
            )





