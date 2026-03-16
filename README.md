# FiLM — Feature-wise Linear Modulation sur CLEVR VQA

Implémentation de [Perez et al. 2018](https://arxiv.org/abs/1709.07871) — *FiLM: Visual Reasoning with a General Conditioning Layer* — avec une interface Streamlit pour l'entraînement interactif et la visualisation.

---

## Qu'est-ce que FiLM ?

FiLM conditionne un CNN visuel sur une question en langage naturel en appliquant une transformation affine par canal sur les feature maps intermédiaires :

```
FiLM(x) = γ(question) · x + β(question)
```

Un GRU encode la question en paramètres `(γ, β)` pour chaque bloc résiduel. Cela permet au pipeline visuel d'être modulé dynamiquement par le langage.

---

## Architecture

```
Question → Embedding → GRU → Linear → (γ_k, β_k) pour k=0..3

Features image (ResNet101, 1024×14×14)
  → Stem : 2×Conv3×3
  → 4× FiLMedResBlock :
       [x ++ coord] → Conv1×1 → ReLU  (résidu)
                    → Conv3×3 → BN(affine=False) → FiLM(γ,β) → ReLU
                    + résidu
  → [features ++ coord] → Conv1×1(→512) → ReLU → MaxPool2×2
  → Flatten → FC(25088→1024) → ReLU → Dropout(0.5) → FC(1024→28)
```

28 classes de sortie (vocabulaire de réponses CLEVR).

---

## Structure du projet

```
FiLM-DL-Project/
├── app.py                    # Page d'accueil Streamlit
├── pages/
│   ├── 1_training.py         # Interface d'entraînement interactive
│   └── 2_style_transfer.py   # Démo style transfer avec CIN
├── src/
│   ├── film_layer.py         # FiLM(x) = γx + β
│   ├── film_generator.py     # Encodeur GRU → (γ, β)
│   ├── model.py              # Modèle FiLM complet
│   ├── dataset.py            # Dataset CLEVR (features H5 ou images brutes)
│   ├── train.py              # Boucle d'entraînement + CLI
│   └── visualize.py          # Helpers Plotly/matplotlib
├── data/
│   ├── download_clevr.py     # Téléchargement d'un sous-ensemble CLEVR
│   └── extract_features.py   # Extraction features ResNet101 → HDF5
├── configs/
│   └── default.yaml          # Hyperparamètres par défaut
└── requirements.txt
```

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Préparation des données

**1. Télécharger CLEVR v1.0**

Télécharger depuis [cs.stanford.edu/people/jcjohns/clevr](https://cs.stanford.edu/people/jcjohns/clevr/) et extraire dans un dossier, par exemple `C:/data/CLEVR_v1.0`.

**2. Extraire les features ResNet101** (une seule fois, ~20 min sur GPU)

```bash
python data/extract_features.py --data-dir C:/data/CLEVR_v1.0 --split train --max-images 70000
python data/extract_features.py --data-dir C:/data/CLEVR_v1.0 --split val   --max-images 1000
```

Cela génère `CLEVR_v1.0/features_train.h5` et `features_val.h5` de forme `(N, 1024, 14, 14)`.

---

## Entraînement

**CLI :**

```bash
python -m src.train --config configs/default.yaml
```

**Application Streamlit :**

```bash
streamlit run app.py
```

La page d'entraînement permet de configurer le modèle, lancer l'entraînement en arrière-plan, et visualiser en direct les courbes de loss, les histogrammes γ/β et des exemples de prédictions.

---

## Configuration (`configs/default.yaml`)

| Paramètre | Défaut | Notes |
|-----------|--------|-------|
| `max_samples_train` | 700000 | Toutes les questions CLEVR (~70k images uniques) |
| `max_samples_val` | 9900 | Limité par le nombre d'images val extraites |
| `num_blocks` | 4 | Blocs résiduels FiLM |
| `num_channels` | 128 | Largeur des feature maps |
| `hidden_dim` | 256 | Dimension cachée GRU (papier : 4096) |
| `embedding_dim` | 300 | Dimension des embeddings de mots |
| `learning_rate` | 3e-4 | Adam |
| `num_epochs` | 30 | Avec early stopping (patience=10) |

---

## Résultats

| Questions d'entraînement | hidden_dim | Précision val |
|--------------------------|-----------|---------------|
| 700k | 256 | ~65-70% (30 epochs) |
| 700k | 4096 | 97,7% (papier, 80 epochs) |

L'écart avec le papier vient du `hidden_dim=256` vs 4096 dans l'encodeur GRU. Augmenter à 1024–2048 permet de se rapprocher des résultats originaux.

---

## Références

- Perez et al. (2018) — [FiLM: Visual Reasoning with a General Conditioning Layer](https://arxiv.org/abs/1709.07871)
- Code original : [github.com/ethanjperez/film](https://github.com/ethanjperez/film)
