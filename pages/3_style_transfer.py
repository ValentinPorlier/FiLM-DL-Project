"""Streamlit page: Style Transfer — Conditional Instance Normalisation."""

from __future__ import annotations

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

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ui_theme import apply_theme

st.set_page_config(
    page_title="Style Transfer — FiLM Explorer",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed",
)
apply_theme()

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

# ---------------------------------------------------------------------------
# Minimal fast-style-transfer network (no external weights needed)
# We implement Johnson et al.'s architecture with Instance Normalisation
# and support multiple styles via conditional IN.
# ---------------------------------------------------------------------------

class ConvLayer(nn.Module):
    def __init__(self, in_c, out_c, kernel, stride=1, upsample=None):
        super().__init__()
        self.upsample = upsample
        pad = kernel // 2
        self.reflection_pad = nn.ReflectionPad2d(pad)
        self.conv = nn.Conv2d(in_c, out_c, kernel, stride)

    def forward(self, x):
        if self.upsample:
            x = F.interpolate(x, scale_factor=self.upsample, mode="nearest")
        return self.conv(self.reflection_pad(x))


class ResBlock(nn.Module):
    def __init__(self, channels, n_styles):
        super().__init__()
        self.conv1 = ConvLayer(channels, channels, 3)
        self.conv2 = ConvLayer(channels, channels, 3)
        # CIN params: gamma and beta for each style
        self.gamma1 = nn.Embedding(n_styles, channels)
        self.beta1  = nn.Embedding(n_styles, channels)
        self.gamma2 = nn.Embedding(n_styles, channels)
        self.beta2  = nn.Embedding(n_styles, channels)
        nn.init.ones_(self.gamma1.weight)
        nn.init.ones_(self.gamma2.weight)
        nn.init.zeros_(self.beta1.weight)
        nn.init.zeros_(self.beta2.weight)

    def _cin(self, x, gamma_emb, beta_emb, style_idx):
        g = gamma_emb(style_idx).view(-1, x.shape[1], 1, 1)
        b = beta_emb(style_idx).view(-1, x.shape[1], 1, 1)
        # Instance normalisation
        mean = x.mean(dim=[2, 3], keepdim=True)
        std  = x.std(dim=[2, 3], keepdim=True) + 1e-5
        return g * (x - mean) / std + b

    def forward(self, x, style_idx):
        r = x
        x = F.relu(self._cin(self.conv1(x), self.gamma1, self.beta1, style_idx))
        x = self._cin(self.conv2(x), self.gamma2, self.beta2, style_idx)
        return x + r


class StyleNet(nn.Module):
    """Lightweight multi-style fast neural style-transfer network."""

    N_STYLES = 5

    def __init__(self):
        super().__init__()
        self.enc1 = ConvLayer(3, 32, 9, stride=1)
        self.enc2 = ConvLayer(32, 64, 3, stride=2)
        self.enc3 = ConvLayer(64, 128, 3, stride=2)

        self.res_blocks = nn.ModuleList(
            [ResBlock(128, self.N_STYLES) for _ in range(5)]
        )

        self.dec1 = ConvLayer(128, 64, 3, upsample=2)
        self.dec2 = ConvLayer(64, 32, 3, upsample=2)
        self.dec3 = ConvLayer(32, 3, 9)

        self.gamma_enc = nn.ParameterList([
            nn.Parameter(torch.ones(1, c, 1, 1))
            for c in [32, 64, 128]
        ])
        self.beta_enc = nn.ParameterList([
            nn.Parameter(torch.zeros(1, c, 1, 1))
            for c in [32, 64, 128]
        ])

    def _cin_fixed(self, x, gamma, beta):
        mean = x.mean(dim=[2, 3], keepdim=True)
        std  = x.std(dim=[2, 3], keepdim=True) + 1e-5
        return gamma * (x - mean) / std + beta

    def forward(self, x, style_idx: torch.Tensor) -> torch.Tensor:
        x = F.relu(self._cin_fixed(self.enc1(x), self.gamma_enc[0], self.beta_enc[0]))
        x = F.relu(self._cin_fixed(self.enc2(x), self.gamma_enc[1], self.beta_enc[1]))
        x = F.relu(self._cin_fixed(self.enc3(x), self.gamma_enc[2], self.beta_enc[2]))

        for blk in self.res_blocks:
            x = blk(x, style_idx)

        x = F.relu(self.dec1(x))
        x = F.relu(self.dec2(x))
        x = torch.sigmoid(self.dec3(x))
        return x


# ---------------------------------------------------------------------------
# Load / create model (random weights — demo only)
# ---------------------------------------------------------------------------

@st.cache_resource
def get_style_model():
    """Return a StyleNet. In a real app you'd load pre-trained weights here."""
    model = StyleNet()
    model.eval()
    return model


STYLE_NAMES = [
    "Impressionniste",
    "Cubiste",
    "Abstrait",
    "Aquarelle",
    "Esquisse",
]

STYLE_COLORS = [
    ["#FF6B6B", "#FFE66D", "#4ECDC4"],   # warm
    ["#2C3E50", "#E74C3C", "#ECF0F1"],   # cool
    ["#8E44AD", "#3498DB", "#2ECC71"],   # vivid
    ["#AED6F1", "#A9DFBF", "#FAD7A0"],   # soft
    ["#2C2C2C", "#888888", "#FFFFFF"],   # greyscale
]

# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

col_upload, col_style = st.columns([2, 1], gap="large")

with col_upload:
    st.markdown('<div class="section-header"> Image source</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Uploadez votre image", type=["png", "jpg", "jpeg"])

with col_style:
    st.markdown('<div class="section-header"> Choix du style</div>', unsafe_allow_html=True)
    style_choice = st.radio("Style artistique", STYLE_NAMES)
    style_idx = STYLE_NAMES.index(style_choice)

    st.markdown("**Palette de couleurs :**")
    palette_html = " ".join(
        f'<span style="background:{c}; display:inline-block; width:28px; height:28px;'
        f' border-radius:6px; margin:2px; box-shadow:0 2px 8px {c}55;"></span>'
        for c in STYLE_COLORS[style_idx]
    )
    st.markdown(palette_html, unsafe_allow_html=True)

st.divider()

if uploaded is not None:
    content_img = Image.open(uploaded).convert("RGB")

    # Resize for speed
    max_size = 512
    w, h = content_img.size
    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        content_img = content_img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    content_tensor = transform(content_img).unsqueeze(0)  # (1, 3, H, W)

    model = get_style_model()
    style_tensor = torch.tensor([style_idx], dtype=torch.long)

    with torch.no_grad():
        output = model(content_tensor, style_tensor)  # (1, 3, H, W)

    output_np = output.squeeze(0).permute(1, 2, 0).numpy()
    output_np = (output_np * 255).clip(0, 255).astype(np.uint8)
    output_img = Image.fromarray(output_np)

    st.divider()
    col_orig, col_out = st.columns(2, gap="large")

    with col_orig:
        st.markdown('<div class="section-header" style="font-size:1rem;"> Image originale</div>',
                    unsafe_allow_html=True)
        st.image(content_img, width="stretch")

    with col_out:
        st.markdown(f'<div class="section-header" style="font-size:1rem;"> Style : {style_choice}</div>',
                    unsafe_allow_html=True)
        st.image(output_img, width="stretch")
        buf = io.BytesIO()
        output_img.save(buf, format="PNG")
        st.download_button(
            "⬇ Télécharger l'image stylisée",
            data=buf.getvalue(),
            file_name=f"style_{style_choice.lower()}.png",
            mime="image/png",
        )

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

    with c_chart:
        if st.checkbox("Afficher les valeurs γ/β brutes pour ce style"):
            blk_idx = st.slider("Bloc résiduel", 0, 4, 0)
            blk = model.res_blocks[blk_idx]
            g = blk.gamma1.weight[style_idx].detach().numpy()
            b = blk.beta1.weight[style_idx].detach().numpy()
            import plotly.graph_objects as go
            fig = go.Figure()
            fig.add_trace(go.Bar(x=list(range(len(g))), y=g.tolist(), name="γ",
                                 marker_color="#6366f1"))
            fig.add_trace(go.Bar(x=list(range(len(b))), y=b.tolist(), name="β",
                                 marker_color="#06b6d4"))
            fig.update_layout(
                title=f"γ et β — style '{style_choice}', bloc {blk_idx}",
                template="plotly_dark",
                height=320,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#94a3b8"),
            )
            st.plotly_chart(fig, width="stretch")
        else:
            st.markdown("""
            <div class="film-card" style="text-align:center; padding:2rem; color:#64748b; font-size:0.85rem;">
                Cochez la case ci-dessus pour visualiser<br>les paramètres γ/β de ce style.
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
