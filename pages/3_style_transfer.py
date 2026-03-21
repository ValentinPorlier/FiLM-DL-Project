"""Style Transfer — Conditional Instance Normalisation via FiLM."""

from __future__ import annotations

<<<<<<< HEAD
import os
import io
=======
>>>>>>> main
import sys
from pathlib import Path

import streamlit as st
import torch
from PIL import Image
from torchvision import transforms

from style_transfert.train import charger_image_aleatoire, train_model_styletransfer, prepare_styletransfer_modele, preparer_pour_plot
from style_transfert.modele import StyleTransferNetwork

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

<<<<<<< HEAD
=======
from style_transfer.train import (
    charger_image_aleatoire,
    prepare_styletransfer_modele,
    preparer_pour_plot,
    train_model_styletransfer,
)
>>>>>>> main

st.set_page_config(
    page_title="Style Transfer — FiLM Explorer",
    layout="wide",
<<<<<<< HEAD
    #initial_sidebar_state="collapsed",
)


data_dir = st.text_input(
    "Répertoire Images (dossier téléchargé: style_transfert_data)",
    value="C:/Users/jeejd/Documents/Vscode/FiLM-DL-Project/style_transfert_data",
)

DOSSIER_IMG = os.path.join(data_dir, "10k_img_resized")
DOSSIER_STYLE = os.path.join(data_dir, "img_style_resized")
CHEMIN_POIDS = os.path.join(data_dir, "StyleTransfer_weights.pth")




=======
)
>>>>>>> main

STYLE_NAMES = ["baroque", "Contemporary", "Cubism", "Early_Renaissance", "Impressionism", "Ukiyo_e"]

# ─── Répertoire des données ────────────────────────────────────────────────────
data_dir = st.text_input(
    "Répertoire du dataset (dossier style_transfert_data/)",
    value="./style_transfert_data",
)

DOSSIER_IMG   = str(Path(data_dir) / "10k_img_resized")
DOSSIER_STYLE = str(Path(data_dir) / "img_style_resized")
CHEMIN_POIDS  = str(Path(data_dir) / "StyleTransfer_weights.pth")

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

<<<<<<< HEAD
st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)


STYLE_NAMES = ["baroque", "Contemporary", "Cubism", "Early_Renaissance", "Impressionism", "Ukiyo_e"]


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
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
    print("✅ Poids du modèle chargés avec succès !")

    col_upload, col_style = st.columns([2, 1], gap="large")

    with col_upload:
        st.markdown('Image aléatoire ou bien uploadez votre image', unsafe_allow_html=True)
        if st.button("Image Random"):
            uploaded, nom_content = charger_image_aleatoire(Path(DOSSIER_IMG) / "images")

        else:
            uploaded = st.file_uploader("Uploadez votre image", type=["png", "jpg", "jpeg"])
            print("Image redimensionnée avec succès")
    
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
        #style_choice = nom_style #temporaire

        
        #style_tensor = torch.tensor([style_idx], dtype=torch.long)
        
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
            """
            buf = io.BytesIO()
            output_img.save(buf, format="PNG")
            st.download_button(
                "⬇ Télécharger l'image stylisée",
                data=buf.getvalue(),
                file_name=f"style_{style_choice.lower()}.png",
                mime="image/png",
            )
            """
        # Explication CIN
        st.divider()
        st.markdown('<div class="section-header"> Comment fonctionne la CIN ici</div>', unsafe_allow_html=True)

        c_exp, c_chart = st.columns([1, 2], gap="large")
        with c_exp:
            st.markdown(f"""
            <div class="film-card">
                <div style="color:#cbd5e1; font-size:0.85rem; line-height:1.9;">
                    Pour le style <strong style="color:#818cf8;">{style_choice}</strong>
                    (index {style_idx}), chaque bloc résiduel possède son propre couple
                    <strong>(γ_{style_idx}, β_{style_idx})</strong>.<br><br>
                    Ces paramètres décalent et scalent les feature maps normalisées —
                    <strong style="color:#e2e8f0;">exactement comme FiLM</strong>,
                    mais pour le style artistique plutôt que le conditionnement par question.<br><br>
                    Changer de style = choisir une autre ligne dans les matrices d'embedding γ/β.
                </div>
            </div>
            <div class="film-card" style="margin-top:0.75rem;">
                <table style="width:100%; font-size:0.8rem; color:#94a3b8; border-collapse:collapse;">
                    <tr><th style="color:#818cf8; text-align:left; padding:0.3rem 0;">Couche</th>
                        <th style="color:#818cf8; text-align:left;">dim γ</th>
                        <th style="color:#818cf8; text-align:left;">dim β</th></tr>
                    <tr><td>ResBlock 1–5</td><td>(128,) × style</td><td>(128,) × style</td></tr>
                </table>
            </div>
            """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div style="text-align:center; padding:3rem 1rem;">
            <div style="font-size:3rem; margin-bottom:1rem;"></div>
            <div style="font-size:1.1rem; font-weight:600; color:#94a3b8; margin-bottom:0.5rem;">
                Uploadez une image pour appliquer un style
            </div>
            <div style="color:#475569; font-size:0.83rem; max-width:480px; margin:0 auto; line-height:1.8;">
                Les poids de style transfer présentés ici sont <strong>initialisés aléatoirement</strong>
                (démo uniquement). Dans un déploiement réel, vous chargeriez des poids pré-entraînés
                sur une collection d'œuvres d'art. L'architecture et le mécanisme CIN sont
                entièrement implémentés.
            </div>
        </div>
        """, unsafe_allow_html=True)
=======
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
>>>>>>> main
