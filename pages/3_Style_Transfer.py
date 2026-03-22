"""Style Transfer — Conditional Instance Normalisation via FiLM."""


from style_transfer.train import (
    charger_image_aleatoire,
    prepare_styletransfer_modele,
    preparer_pour_plot,
    train_model_styletransfer,
)
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


st.set_page_config(
    page_title="Style Transfer — FiLM Explorer",
    layout="wide",
)

st.title("Style Transfer")
st.divider()

# ─── Principe CIN ─────────────────────────────────────────────────────────────
st.header("Conditional Instance Normalisation — le même principe que FiLM appliqué au style artistique.")
st.write("Le transfert de style que nous implémentons dans cette section se base sur l'article de [Ghiasi et al. (2017)](https://arxiv.org/pdf/1705.06830). "
         "Le modèle prend en entrée une image et un style puis génère la même image d'entrée avec le style à appliquer. Le modèle utilise 10 000 images de content"
         " issues d'ImageNet et 6 000 images de styles issues de WikiArt (kaggle). Les images ont été redimensionnées en 256x256 et mises à disposition sur un drive.")


st.header("Architecture et Fonctionnement")

# Section Générateur
st.subheader("1. FiLM Generator :")
st.markdown(r"""
Le modèle utilise un **FiLM Generator** pour prédire les paramètres $\gamma$ et $\beta$ à partir de l'image de style.
*   **Extraction** : On utilise le bloc `mixed 6e` d'un **Inception-v3** pré-entraîné.
*   **Condensation** : Un **Global Average Pooling** transforme les feature maps en vecteur.
*   **Projection** : Un **MLP** génère enfin les coefficients de la transformation affine appliquée à l'image de contenu.
""")

# Section Réseau Principal
st.subheader("2. Réseau Principal :")
st.markdown("""
Il s'agit d'un auto-encodeur composé d'une succession de blocs résiduels (2 convolutions par bloc avec activation). La reconstruction de l'image est assurée par deux couches d'upsampling et une convolution finale pour obtenir le bon nombre de canaux et l'image stylisée.
""")


# ─── Section Loss ───────────────────────────────────────────────────────
st.subheader("3. Détails du calcul de la Loss (VGG16) :")
st.markdown(r"""
Nous utilisons un modèle VGG16 pré-entraîné comme extracteur de caractéristiques pour comparer les trois images : celle de content, de style et celle générée.
            On note $\mathcal{S}$ les couches de bas niveau et $\mathcal{C}$ les couches intermédiaires (une seule ici) du modèle de classification.
""")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Loss de Contenu")
    st.markdown(
        "Mesure la fidélité au contenu en comparant les feature maps extraites "
        "par un réseau VGG pré-entraîné à une couche intermédiaire:"
    )
    # content loss
    st.latex(
        r"\mathcal{L}_c(x, c) = \sum_{j \in \mathcal{C}} \frac{1}{n_j} \| f_j(x) - f_j(c) \|^2_2"
    )


with col2:
    st.subheader("Loss de Style")
    st.markdown(
        "Capture les corrélations entre canaux (textures, motifs) via la matrice de Gram — "
        "puis mesure la distance entre les matrices de l'image générée et du style cible via la norme de Frobenius:"
    )
    # matrice de gram
    st.latex(r"G^l = F^l (F^l)^T")
    st.markdown(
        r"Où $F^l$ est la matrice des activations de taille $(C \times N)$ obtenue en aplatissant "
        r"les dimensions des matrices, avec $N = H \times W$."
    )
    # style loss
    st.latex(
        r"\mathcal{L}_s(x, s) = \sum_{i \in \mathcal{S}} \frac{1}{n_i} \| \mathcal{G}[f_i(x)] - \mathcal{G}[f_i(s)] \|^2_F"
    )
    st.write("Utilisation des **Matrices de Gram** sur les premières couches et calcul de la distance via la **norme de Frobenius**.")

st.write(
    r"où $f_{l}(x)$ représente les activations du réseau à la couche $l$, $n_{l}$ représente le nombre total de neurones à cette même couche et $\mathcal{G}[f_{l}(x)]$ est la matrice de Gram associée aux activations de la couche l. ")

# ─── Section résumé ───────────────────────────────────────────────────────
st.subheader("4. Visualisation de l'architecture")
style_transfer_arch = ROOT / "assets" / "Style_Transfer_Ghiasi.png"
col1, col2, col3 = st.columns([1, 3, 1])
with col2:
    if style_transfer_arch.exists():
        st.image(
            str(style_transfer_arch),
            caption="Architecture du modèle de Style Transfer, image issu de [Ghiasi et al. (2017)](https://arxiv.org/pdf/1705.06830)",
            use_container_width=True
        )
    else:
        st.error(
            f"Image introuvable, vérifiez qu'elle est bien dans : {style_transfer_arch}")
        if (ROOT / "assets").exists():
            st.write("Fichiers présents dans /assets :",
                     [f.name for f in (ROOT / "assets").iterdir()])


st.divider()


# ─── Interface modèle ────────────────────────────────────────────────────────────────
st.title("Modèle")
st.write("Le modèle n'est pas très performant, la base de donnée que nous avons pris est une fraction de celle utilisé dans l'article car trop lourde (l'article utilise au moins 80 000 images juste pour le style, nous avons 10 000 images de content d'ImageNet et 1000 images par style)."
         " Nous avons tout de même voulu implenter la fonctionnalité dans l'application avec un modèle pré entraîné.")


STYLE_NAMES = ["baroque", "Contemporary", "Cubism",
               "Early_Renaissance", "Impressionism", "Ukiyo_e"]

_ZIP_FILE_ID = "1qnu_aMUz54F5cGjLYeL2-MYuIGzxODjI"

# ─── Répertoire des données ────────────────────────────────────────────────────
data_dir = st.text_input(
    "Répertoire du dataset (dossier style_transfer/data/)",
    value="./style_transfer/data",
)

DOSSIER_IMG = str(Path(data_dir) / "10k_img_resized")
DOSSIER_STYLE = str(Path(data_dir) / "img_style_resized")
CHEMIN_POIDS = str(Path(data_dir) / "StyleTransfer_weights.pth")

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
                target = ROOT / "style_transfer" / "data"
                target.mkdir(parents=True, exist_ok=True)
                with zipfile.ZipFile(zip_path, "r") as zf:
                    # détecte si le zip a un dossier racine unique
                    top_levels = {p.split("/")[0]
                                  for p in zf.namelist() if p.strip("/")}
                    tmp_dir = Path(tempfile.mkdtemp())
                    zf.extractall(str(tmp_dir))
                    src = tmp_dir / \
                        list(top_levels)[0] if len(top_levels) == 1 else tmp_dir
                    import shutil
                    for item in src.iterdir():
                        shutil.move(str(item), str(target / item.name))
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


# ─── Modèle et affichage ────────────────────────────────────────────────────

entrained = st.checkbox("Utiliser un modèle pré-entraîné", value=False)

if not entrained:
    st.warning("L'entraînement du modele est long (10min par epoch voire plus dépendant du gpu utilisé (si gpu il y a)). Il est préférable de prendre le modèle pré-entraîné même si les résultats ne sont pas forcément satisfaisants")
    n_epochs = st.slider("Epochs", 1, 50, 10)
    batch_sz = st.slider("Batch size", 4, 32, 8, step=4)
    lr = st.number_input("Learning rate", value=0.001, format="%.4f")
    lambda_style = st.select_slider("Poids de la loss du style", options=[
                                    1e4, 1e5, 1e6], format_func=lambda x: f"{x:.0e}")

    if st.button("lancer l'entrainement du modele"):
        try:
            model, dataloader, device = prepare_styletransfer_modele(
                data_dir, batch_size=batch_sz)
            st.success("Modèle bien chargé")
        except Exception as e:
            st.error(f"Erreur dans le chargement du modèle : {e}")

        st.write("entrainement du modèle...")
        progress_container = st.empty()
        train_model_styletransfer(
            model=model, dataloader=dataloader, device=device,
            epochs=n_epochs, lr=lr, lambda_style=lambda_style,
            st_container=progress_container)
        entrained = True

else:
    model, dataloader, device = prepare_styletransfer_modele(data_dir, batch_size=8)

if entrained:
    model.load_state_dict(torch.load(
        CHEMIN_POIDS, map_location=device, weights_only=True))

    col_upload, col_style = st.columns([2, 1], gap="large")

    with col_upload:
        st.markdown('Image aléatoire ou bien uploadez votre image',
                    unsafe_allow_html=True)
        uploaded = st.file_uploader("Uploadez votre image", type=["png", "jpg", "jpeg"])
        if st.button("Image Random"):
            uploaded, nom_content = charger_image_aleatoire(
                Path(DOSSIER_IMG) / "images")

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

        style_tensor_img, nom_style = charger_image_aleatoire(
            Path(DOSSIER_STYLE) / "images", style=style_choice)

        # Contournement du Bug de Dimension (Batch de 2)
        content_img = torch.cat([content_tensor, content_tensor], dim=0).to(device)
        style_tensor = torch.cat([style_tensor_img, style_tensor_img], dim=0).to(device)

        with torch.no_grad():
            output = model(content_img, style_tensor)

        output_np = preparer_pour_plot(output[0])
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
