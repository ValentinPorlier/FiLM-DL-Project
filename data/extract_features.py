"""Extract CNN features from CLEVR images using pretrained ResNet101.

Reproduces the preprocessing from Perez et al. 2018:
  - ResNet101 truncated after layer3
  - Output: (N, 1024, 14, 14) per image at 224x224 input
  - Saved to HDF5 for fast loading during training

Usage
-----
    python data/extract_features.py --split train --max-images 5000
    python data/extract_features.py --split val   --max-images 1000

Output
------
    data/clevr/features_train.h5  — datasets: "features" (N,1024,14,14), "image_ids" (N,)
    data/clevr/features_val.h5
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Build the feature extractor (ResNet101 up to layer3)
# ---------------------------------------------------------------------------

def build_extractor(device: torch.device) -> nn.Module:
    """Return ResNet101 truncated after layer3, pretrained on ImageNet."""
    resnet = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
    # Keep: conv1, bn1, relu, maxpool, layer1, layer2, layer3
    extractor = nn.Sequential(
        resnet.conv1,
        resnet.bn1,
        resnet.relu,
        resnet.maxpool,
        resnet.layer1,
        resnet.layer2,
        resnet.layer3,
    )
    extractor.eval()
    for p in extractor.parameters():
        p.requires_grad = False
    return extractor.to(device)


# Exact preprocessing used in the FiLM paper
TRANSFORM = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ---------------------------------------------------------------------------
# Main extraction loop
# ---------------------------------------------------------------------------

def extract(
    data_dir: str | Path,
    split: str,
    max_images: int,
    batch_size: int,
    device: torch.device,
) -> None:
    """Extrait les features ResNet101 des images CLEVR et les sauvegarde en HDF5.

    Si le fichier de sortie existe déjà avec suffisamment d'images, l'extraction
    est ignorée. Sinon, le fichier est recréé depuis zéro.

    Parameters
    ----------
    data_dir : str ou Path
        Dossier racine des données CLEVR (contient ``images/train``, ``images/val``…).
    split : str
        Découpage à traiter : ``"train"``, ``"val"`` ou ``"test"``.
    max_images : int
        Nombre maximum d'images à traiter.
    batch_size : int
        Taille des batchs pour l'inférence ResNet101.
    device : torch.device
        Appareil de calcul (CPU ou GPU).

    Returns
    -------
    None
        Les features sont sauvegardées dans ``data_dir/features_{split}.h5``.
    """
    data_dir = Path(data_dir)
    images_dir = data_dir / "images" / split
    out_path = data_dir / f"features_{split}.h5"

    if out_path.exists():
        with h5py.File(out_path, "r") as f:
            existing_n = f["features"].shape[0]
        if existing_n >= max_images:
            print(f"[{split}] H5 already has {existing_n} images (>= {max_images}), skipping.")
            return
        print(f"[{split}] H5 has {existing_n} images but {max_images} requested — re-extracting.")
        out_path.unlink()

    # Collect image filenames (sorted for reproducibility)
    all_images = sorted(images_dir.glob("*.png"))[:max_images]
    n = len(all_images)
    print(f"[{split}] Extracting features for {n} images → {out_path}")

    extractor = build_extractor(device)

    # First pass: determine output shape with one image
    sample = TRANSFORM(Image.open(all_images[0]).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        sample_out = extractor(sample)
    C, H, W = sample_out.shape[1:]
    print(f"[{split}] Feature shape per image: ({C}, {H}, {W})")

    with h5py.File(out_path, "w") as f:
        feat_ds = f.create_dataset("features", shape=(n, C, H, W), dtype="float32")
        id_ds = f.create_dataset(
            "image_ids", shape=(
                n,), dtype=h5py.special_dtype(
                vlen=str))

        idx = 0
        for batch_start in tqdm(range(0, n, batch_size), desc=f"Extracting {split}"):
            batch_paths = all_images[batch_start: batch_start + batch_size]
            imgs = torch.stack([
                TRANSFORM(Image.open(p).convert("RGB"))
                for p in batch_paths
            ]).to(device)

            with torch.no_grad():
                feats = extractor(imgs).cpu().numpy()

            batch_len = len(batch_paths)
            feat_ds[idx: idx + batch_len] = feats
            id_ds[idx: idx + batch_len] = [p.name for p in batch_paths]
            idx += batch_len

    print(f"[{split}] Done. Saved {n} feature tensors to {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """Point d'entrée en ligne de commande pour l'extraction de features."""
    parser = argparse.ArgumentParser(
        description="Extract ResNet101 features from CLEVR images.")
    parser.add_argument("--data-dir", default="data/clevr")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--max-images", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Using device: {device}")

    extract(
        data_dir=args.data_dir,
        split=args.split,
        max_images=args.max_images,
        batch_size=args.batch_size,
        device=device,
    )


if __name__ == "__main__":
    main()
