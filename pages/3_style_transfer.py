"""Style Transfer — Conditional Instance Normalisation via FiLM."""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
import gdown
import zipfile
import streamlit as st
import torch
from PIL import Image
from torchvision import transforms

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from style_transfer.train import (
    charger_image_aleatoire,
    prepare_styletransfer_modele,
    preparer_pour_plot,
    train_model_styletransfer,
)

st.set_page_config(
    page_title="Style Transfer — FiLM Explorer",
    layout="wide",
)

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

_ZIP_FILE_ID = "1qnu_aMUz54F5cGjLYeL2-MYuIGzxODjI"

if not Path(data_dir).exists():
    st.warning(f"Données introuvables dans `{data_dir}`.")
    if st.button("Télécharger les données depuis Google Drive"):

        err: list = []
        zip_path = ROOT / "style_transfer_data.zip"

        def _download():
            try:
                gdown.download(
                    id=_ZIP_FILE_ID,
                    output=str(zip_path),
                    quiet=False,
                    fuzzy=True,
                )
                with zipfile.ZipFile(zip_path, "r") as zf:
                    zf.extractall(str(ROOT))
                zip_path.unlink()
            except Exception as e:
                err.append(e)

        t = threading.Thread(target=_download, daemon=True)
        t.start()

        info = st.empty()
        while t.is_alive():
            size_mb = zip_path.stat().st_size / 1e6 if zip_path.exists() else 0
            info.text(f"Téléchargement... {size_mb:.1f} Mo reçus")
            time.sleep(1)
        t.join()

        if err:
            st.error(f"Erreur lors du téléchargement : {err[0]}")
        else:
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
    st.warning(
        "L'entraînement est long (10 min par epoch voire plus). "
        "Il est préférable d'utiliser le modèle pré-entraîné."
    )
    n_epochs     = st.slider("Epochs",     1,  50,  10)
    batch_sz     = st.slider("Batch size", 32, 512, 128, step=32)
    lr           = st.number_input("Learning rate", value=0.001, format="%.4f")
    lambda_style = st.select_slider(
        "Poids de la loss de style",
        options=[1e4, 1e5, 1e6],
        format_func=lambda x: f"{x:.0e}",
    )

    if st.button("Lancer l'entraînement"):
        try:
            model, dataloader, device = prepare_styletransfer_modele(data_dir, batch_size=batch_sz)
        except Exception as e:
            st.error(f"Erreur dans le chargement du modèle : {e}")
            st.stop()

        with st.spinner("Entraînement en cours..."):
            history = train_model_styletransfer(
                model=model, dataloader=dataloader, device=device,
                epochs=n_epochs, lr=lr, lambda_style=lambda_style,
            )
        st.success("Entraînement terminé.")
        st.line_chart({"Train Loss": history["train_loss"]}, x_label="Epoch", y_label="Loss")

else:
    model, dataloader, device = prepare_styletransfer_modele(data_dir, batch_size=128)

    if not Path(CHEMIN_POIDS).exists():
        st.error(f"Fichier de poids introuvable : `{CHEMIN_POIDS}`")
        st.stop()

    model.load_state_dict(torch.load(CHEMIN_POIDS, map_location=device, weights_only=True))
    model.eval()

    col_upload, col_style = st.columns([2, 1], gap="large")

    with col_upload:
        st.subheader("Image source")
        if st.button("Image aléatoire"):
            uploaded, _ = charger_image_aleatoire(Path(DOSSIER_IMG) / "images")
        else:
            uploaded_file = st.file_uploader("Uploadez votre image", type=["png", "jpg", "jpeg"])
            uploaded = None
            if uploaded_file is not None:
                img_pil = Image.open(uploaded_file).convert("RGB")
                transform = transforms.Compose([
                    transforms.Resize((256, 256)),
                    transforms.ToTensor(),
                ])
                uploaded = transform(img_pil).unsqueeze(0)

    with col_style:
        st.subheader("Choix du style")
        style_choice = st.radio("Style artistique", STYLE_NAMES)
        style_idx    = STYLE_NAMES.index(style_choice)

    st.divider()

    if uploaded is not None:
        style_tensor_img, nom_style = charger_image_aleatoire(
            Path(DOSSIER_STYLE) / "images", style=style_choice
        )

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
            st.subheader(f"Style : {style_choice}")
            st.image(output_np, caption="Image stylisée", use_container_width=True)

        st.divider()
        st.subheader("Comment fonctionne la CIN ici")
        st.markdown(f"""
Pour le style **{style_choice}** (index {style_idx}), chaque bloc résiduel possède
son propre couple **(γ_{style_idx}, β_{style_idx})**.

Ces paramètres décalent et scalent les feature maps normalisées —
exactement comme FiLM, mais pour le style artistique plutôt que le
conditionnement par question.

| Couche | dim γ | dim β |
|--------|-------|-------|
| ResBlock 1–5 | (128,) | (128,) |
| Upsample 1 | (64,) | (64,) |
| Upsample 2 | (32,) | (32,) |
""")

    else:
        st.info("Uploadez une image ou cliquez sur « Image aléatoire » pour appliquer un style.")
