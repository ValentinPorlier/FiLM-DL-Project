"""Streamlit page: Style Transfer — Conditional Instance Normalisation."""

from __future__ import annotations

import os
import io
import sys
from pathlib import Path

import numpy as np
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


data_dir = st.text_input(
    "Répertoire Images (dossier téléchargé: style_transfert_data)",
    value="C:/Users/jeejd/Documents/Vscode/FiLM-DL-Project/style_transfert_data",
)

DOSSIER_IMG = os.path.join(data_dir, "10k_img_resized")
DOSSIER_STYLE = os.path.join(data_dir, "img_style_resized")
CHEMIN_POIDS = os.path.join(data_dir, "StyleTransfer_weights.pth")

try:
    model, dataloader, device = prepare_styletransfer_modele(data_dir, batch_size=128)
except Exception as e:
    st.write("Erreur dans le chargement du modele")
    print("Erreur dans le chargement du modele")




st.markdown("""
<div style="padding:1.5rem 0 0.5rem 0;">
    <div class="hero-title" style="font-size:clamp(1.8rem,4vw,3rem);"> Style Transfer</div>
    <p class="hero-sub">
        Conditional Instance Normalisation — le même principe que FiLM appliqué au style artistique
    </p>
    <span class="badge badge-purple">Dumoulin et al. 2017</span>
    <span class="badge badge-cyan">CIN</span>
    <span class="badge badge-amber">Johnson et al. 2016</span>
</div>
""", unsafe_allow_html=True)

st.divider()

st.markdown("""
<div class="film-card" style="max-width:800px;">
    <div style="color:#cbd5e1; font-size:0.9rem; line-height:1.8;">
        La <strong style="color:#e2e8f0;">Conditional Instance Normalisation (CIN)</strong>
        (Dumoulin et al., 2017) reprend exactement l'idée de FiLM :
        au lieu d'apprendre un seul jeu de paramètres d'Instance Normalisation,
        un couple <strong style="color:#818cf8;">(γₛ, βₛ)</strong> distinct est appris
        pour chaque style <strong>s</strong>.<br><br>
        Changer de style revient simplement à choisir une ligne différente
        dans les matrices d'embedding γ/β — sans modifier l'architecture.
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)


STYLE_NAMES = ["baroque", "Contemporary", "cubism", "early_renaissance", "impressionism", "Ukiyo_e"]


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
entrained = st.checkbox("Utiliser un modèle pré-entraîné", value=False)

if not entrained:
    print('a')

if entrained:
    model.load_state_dict(torch.load(CHEMIN_POIDS, map_location=device, weights_only=True))
    print("✅ Poids du modèle chargés avec succès !")

    col_upload, col_style = st.columns([2, 1], gap="large")


    if st.button("Image Random"):
        uploaded, nom_content = charger_image_aleatoire(Path(DOSSIER_IMG) / "images")

    else:
        with col_upload:
            st.markdown('<div class="section-header"> Image source</div>', unsafe_allow_html=True)
            uploaded = st.file_uploader("Uploadez votre image", type=["png", "jpg", "jpeg"])


            print("Image redimensionnée avec succès !")
    
    #with col_style:
    #    st.markdown('<div class="section-header"> Choix du style</div>', unsafe_allow_html=True)
    #    style_choice = st.radio("Style artistique", STYLE_NAMES)
    #    style_idx = STYLE_NAMES.index(style_choice)
    """
        st.markdown("**Palette de couleurs :**")
        palette_html = " ".join(
            f'<span style="background:{c}; display:inline-block; width:28px; height:28px;'
            f' border-radius:6px; margin:2px; box-shadow:0 2px 8px {c}55;"></span>'
            for c in STYLE_COLORS[style_idx]
        )
        st.markdown(palette_html, unsafe_allow_html=True)
        """
    st.divider()

    if uploaded is not None:
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
            
        style_tensor_img, nom_style = charger_image_aleatoire(Path(DOSSIER_STYLE) / "images")
        style_choice = nom_style #temporaire

        
        
        #Pour le style tenseur je prends aléatoire pour le moment
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
