"""Style Transfer — Conditional Instance Normalisation via FiLM."""


import io
import sys
import tempfile
import threading
import time
import zipfile
from pathlib import Path

import gdown
import numpy as np
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

_ZIP_FILE_ID = "1qnu_aMUz54F5cGjLYeL2-MYuIGzxODjI"

# ─── Répertoire des données ────────────────────────────────────────────────────
data_dir = st.text_input(
    "Répertoire du dataset (dossier style_transfer/data/)",
    value="./style_transfer/data",
)

DOSSIER_IMG   = str(Path(data_dir) / "10k_img_resized")
DOSSIER_STYLE = str(Path(data_dir) / "img_style_resized")
CHEMIN_POIDS  = str(Path(data_dir) / "StyleTransfer_weights.pth")

if not Path(data_dir).exists():
    st.warning(f"Données introuvables dans `{data_dir}`.")
    if st.button("Télécharger les données depuis Google Drive"):
        err: list = []
        zip_path = Path(tempfile.gettempdir()) / "style_transfer_data.zip"

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
                # déplace dans style_transfer/data/ quel que soit le nom dans le zip
                target = ROOT / "style_transfer" / "data"
                for candidate in [ROOT / "style_transfer_data", ROOT / "style_transfert_data"]:
                    if candidate.exists() and not target.exists():
                        candidate.rename(target)
                        break
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

# ─── Loss d'entraînement ───────────────────────────────────────────────────────
st.header("Entraînement")

col_cl, col_sl = st.columns(2)

with col_cl:
    st.subheader("Content loss")
    st.markdown(
        "Mesure la fidélité au contenu en comparant les feature maps extraites "
        "par un réseau VGG pré-entraîné à une couche intermédiaire $\\phi$ :"
    )
    st.latex(r"\mathcal{L}_{\text{content}} = \| \phi(\hat{x}) - \phi(x) \|_2^2")

with col_sl:
    st.subheader("Style loss")
    st.markdown(
        "Capture les corrélations entre canaux (textures, motifs) via la **matrice de Gram** — "
        "puis mesure la distance entre les matrices de l'image générée et du style cible :"
    )
    st.latex(
        r"G^l(\hat{x}) = \frac{1}{C_l H_l W_l}\,\phi_l(\hat{x})^\top \phi_l(\hat{x})"
    )
    st.latex(
        r"\mathcal{L}_{\text{style}} = \sum_l \| G^l(\hat{x}) - G^l(x_s) \|_F^2"
    )

st.markdown(
    r"**Loss totale :** $\mathcal{L} = \mathcal{L}_{\text{content}}"
    r"+ \lambda_{\text{style}}\,\mathcal{L}_{\text{style}}$"
    " — $\\lambda_{\\text{style}}$ contrôle l'équilibre contenu / style."
)

st.divider()

# ─── Interface ────────────────────────────────────────────────────────────────
entrained = st.checkbox("Utiliser un modèle pré-entraîné", value=False)

if not entrained:
    st.warning("L'entraînement du modele est long (10min par epoch voire plus). Il est préférable de prendre le modèle pré-entrainé même si les résultats ne sont pas forcément satisfaisants")
    n_epochs     = st.slider("Epochs", 1, 50, 10)
    batch_sz     = st.slider("Batch size", 32, 512, 128, step=32)
    lr           = st.number_input("Learning rate", value=0.001, format="%.4f")
    lambda_style = st.select_slider("Poids de la loss du style", options=[1e4, 1e5, 1e6], format_func=lambda x: f"{x:.0e}")

    if st.button("lancer l'entrainement du modele"):
        try:
            model, dataloader, device = prepare_styletransfer_modele(data_dir, batch_size=batch_sz)
            st.success("Modèle bien chargé")
        except Exception as e:
            st.error(f"Erreur dans le chargement du modèle : {e}")

        st.write("entrainement du modèle...")
        train_model_styletransfer(model=model, dataloader=dataloader, device=device, epochs=n_epochs, lr=lr, lambda_style=lambda_style)
        entrained = True

else:
    model, dataloader, device = prepare_styletransfer_modele(data_dir, batch_size=128)

if entrained:
    model.load_state_dict(torch.load(CHEMIN_POIDS, map_location=device, weights_only=True))

    col_upload, col_style = st.columns([2, 1], gap="large")

    with col_upload:
        st.markdown('Image aléatoire ou bien uploadez votre image', unsafe_allow_html=True)
        uploaded = st.file_uploader("Uploadez votre image", type=["png", "jpg", "jpeg"])
        if st.button("Image Random"):
            uploaded, nom_content = charger_image_aleatoire(Path(DOSSIER_IMG) / "images")

    with col_style:
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

        # Contournement du Bug de Dimension (Batch de 2)
        content_img  = torch.cat([content_tensor, content_tensor], dim=0).to(device)
        style_tensor = torch.cat([style_tensor_img, style_tensor_img], dim=0).to(device)

        with torch.no_grad():
            output = model(content_img, style_tensor)

        output_np      = preparer_pour_plot(output[0])
        content_img_np = preparer_pour_plot(content_tensor)
        style_tensor_np = preparer_pour_plot(style_tensor_img)

        st.divider()
        col_cont, col_styl, col_out = st.columns(3, gap="large")

        with col_cont:
            st.markdown("**Image originale**")
            st.image(content_img_np, use_container_width=True)

        with col_styl:
            st.markdown("**Image de style**")
            st.image(style_tensor_np, use_container_width=True)

        with col_out:
            st.markdown(f"**Style : {style_choice}**")
            st.image(output_np, caption="Image stylisée", use_container_width=True)

            img_pour_pil = (output_np.copy() * 255).astype(np.uint8)
            with io.BytesIO() as buf:
                Image.fromarray(img_pour_pil).save(buf, format="PNG")
                donnees_binaires = buf.getvalue()

            st.download_button(
                label="Télécharger l'image générée",
                data=donnees_binaires,
                file_name=f"style_transfer_{style_choice}.png",
                mime="image/png",
            )
