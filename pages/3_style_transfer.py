"""Style Transfer — Conditional Instance Normalisation via FiLM."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st
import torch
from PIL import Image
from torchvision import transforms

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from style_transfert.modele import StyleTransferNetwork
from style_transfert.train import (
    charger_image_aleatoire,
    prepare_styletransfer_modele,
    preparer_pour_plot,
    train_model_styletransfer,
)

st.set_page_config(
    page_title="Style Transfer — FiLM Explorer",
    layout="wide",
)

st.title("Style Transfer")
st.caption("Conditional Instance Normalisation — le même principe que FiLM appliqué au style artistique")
st.divider()

# ─── Répertoire des données ────────────────────────────────────────────────────
data_dir = st.text_input(
    "Répertoire du dataset (dossier style_transfert_data/)",
    value="./style_transfert_data",
)

DOSSIER_IMG   = str(Path(data_dir) / "10k_img_resized")
DOSSIER_STYLE = str(Path(data_dir) / "img_style_resized")
CHEMIN_POIDS  = str(Path(data_dir) / "StyleTransfer_weights.pth")

try:
    model, dataloader, device = prepare_styletransfer_modele(data_dir, batch_size=128)
except Exception as e:
    st.error(f"Erreur dans le chargement du modèle : {e}")
    st.stop()

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

if entrained:
    if not Path(CHEMIN_POIDS).exists():
        st.error(f"Fichier de poids introuvable : `{CHEMIN_POIDS}`")
        st.stop()
    model.load_state_dict(torch.load(CHEMIN_POIDS, map_location=device, weights_only=True))
    model.eval()

    col_upload, _ = st.columns([2, 1])

    if st.button("Image aléatoire"):
        uploaded, _ = charger_image_aleatoire(Path(DOSSIER_IMG) / "images")
    else:
        with col_upload:
            st.subheader("Image source")
            uploaded_file = st.file_uploader("Uploadez votre image", type=["png", "jpg", "jpeg"])
            uploaded = None
            if uploaded_file is not None:
                img_pil = Image.open(uploaded_file).convert("RGB")
                transform = transforms.Compose([
                    transforms.Resize((256, 256)),
                    transforms.ToTensor(),
                ])
                uploaded = transform(img_pil).unsqueeze(0)

    st.divider()

    if uploaded is not None:
        style_tensor_img, nom_style = charger_image_aleatoire(Path(DOSSIER_STYLE) / "images")

        # Batch de 2 pour éviter les erreurs de dimension avec BatchNorm
        content_img  = torch.cat([uploaded, uploaded], dim=0).to(device)
        style_tensor = torch.cat([style_tensor_img, style_tensor_img], dim=0).to(device)

        with torch.no_grad():
            output = model(content_img, style_tensor)  # (2, 3, H, W)

        output_np       = preparer_pour_plot(output[0])
        content_img_np  = preparer_pour_plot(uploaded)
        style_tensor_np = preparer_pour_plot(style_tensor_img)

        col_cont, col_styl, col_out = st.columns(3)
        with col_cont:
            st.subheader("Image originale")
            st.image(content_img_np, use_container_width=True)
        with col_styl:
            st.subheader("Image de style")
            st.image(style_tensor_np, use_container_width=True)
            st.caption(nom_style)
        with col_out:
            st.subheader("Image stylisée")
            st.image(output_np, use_container_width=True)

    else:
        st.info("Uploadez une image ou cliquez sur « Image aléatoire » pour appliquer un style.")

else:
    st.subheader("Entraînement")
    n_epochs      = st.slider("Epochs",     1,  30, 10)
    batch_sz      = st.slider("Batch size", 16, 256, 128, step=16)
    lambda_style  = st.number_input("Lambda style", value=1_000_000, step=100_000)

    if st.button("Lancer l'entraînement"):
        with st.spinner("Entraînement en cours..."):
            history = train_model_styletransfer(
                model, dataloader, device,
                epochs=n_epochs,
                lambda_style=lambda_style,
            )
        st.success("Entraînement terminé.")
        st.line_chart({"Train Loss": history["train_loss"]}, x_label="Epoch", y_label="Loss")
