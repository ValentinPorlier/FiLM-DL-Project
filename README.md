# FiLM — Feature-wise Linear Modulation sur CLEVR VQA

Implémentation de [Perez et al. (2018)](https://arxiv.org/abs/1709.07871) —
*FiLM: Visual Reasoning with a General Conditioning Layer* —
avec une application Streamlit interactive : entraînement sur Sort of CLEVR (Kaggle),
CLEVR VQA, et transfert de style artistique via Conditional Instance Normalisation (CIN).

---

## Principe

FiLM conditionne un CNN visuel sur une question en langage naturel en appliquant
une **transformation affine par canal** sur les feature maps intermédiaires :

$$\text{FiLM}(x_i^c) = \gamma_i^c(\text{question}) \cdot x_i^c + \beta_i^c(\text{question})$$

Un GRU encode la question en paramètres $(\gamma, \beta)$ pour chaque bloc
résiduel. Le pipeline visuel est ainsi modulé dynamiquement par le langage.
La formulation résiduelle $\gamma = 1 + \Delta\gamma$ garantit que le modèle
converge vers l'identité à l'initialisation.

---

## Architecture

```
Question → Embedding → GRU → Linear → (γk, βk) pour k = 0…3

Image → ResNet101 → (1024, 14, 14)
  → Stem : 2x Conv3x3 → (128, 14, 14)
  → 4x FiLMedResBlock :
       [x ++ coord] → Conv1x1 → ReLU   (résidu)
                    → Conv3x3 → BN(affine=False) → FiLM(γ,β) → ReLU
                    + résidu
  → [features ++ coord] → Conv1x1(→512) → ReLU → MaxPool
  → Flatten → FC(25088→1024) → ReLU → Dropout(0.5) → FC(1024→28)
```

28 classes de sortie (vocabulaire de réponses CLEVR).

---

## Forces et limites

| Forces | Limites |
|--------|---------|
| Architecture générique et légère | Dépend de features pré-extraites (ResNet101) |
| Conditionnement cross-modal élégant | GRU doit être large (hidden\_dim ≥ 1024) |
| Formulation résiduelle γ = 1 + Δγ stable | Entraînement long sur CLEVR complet |
| État de l'art sur CLEVR (97,7 %) | Pas de raisonnement multi-saut natif |

---

## Datasets

### Sort of CLEVR (Kaggle)

Dataset 2D issu de Kaggle, utilisé comme étape intermédiaire avant CLEVR 3D.
Chaque image contient des formes colorées (cercles ou carrés) sur fond blanc.
Trois types de questions :

| Type | Exemple | Réponses possibles |
|------|---------|-------------------|
| Forme | *What shape is the blue object?* | `circle`, `square` |
| Direction | *What is the position of red relative to green?* | `right`, `left`, `top`, `bottom` |
| Couleur | *What color is the object most to the right?* | `blue`, `yellow`, `green`, `red`, `gray` |

**Encodage de question (10 dims) :**

| Dims | Contenu |
|------|---------|
| 0–2  | Type one-hot : shape / direction / color_at_pos |
| 3–7  | Couleur objet 1 (one-hot, 5 couleurs) |
| 8    | Couleur objet 2 normalisée ∈ [0, 1] |
| 9    | Direction normalisée ∈ [0, 1] |

### CLEVR VQA

Dataset 3D photoréaliste officiel (~18 Go). 700 000 questions de raisonnement
sur des scènes générées par Blender. Features ResNet101 pré-extraites
(1024 × 14 × 14) nécessaires pour l'entraînement.

---

## Application Streamlit

L'application couvre l'ensemble du pipeline depuis l'interface :

| Page | Contenu |
|------|---------|
| **Accueil** | Théorie FiLM, formulation, architecture, forces/limites |
| **Sort of CLEVR** | Entraînement interactif sur dataset Kaggle 2D, test visuel |
| **CLEVR VQA** | Visualisation des courbes d'apprentissage, métriques |
| **Style Transfer** | Transfert de style artistique via CIN (Dumoulin et al., 2017) |

### Page Sort of CLEVR — fonctionnalités

- **Entraînement** : hyperparamètres configurables (epochs, batch size,
  learning rate, nombre de samples), suivi en direct via stqdm
- **Modèle pré-entraîné** : chargement des poids `model_weights.pth` en un clic
- **Test visuel** : sélection d'une image de test, choix d'une question,
  affichage de la réponse prédite

### Page Style Transfer — fonctionnalités

- **Principe CIN** : même formulation que FiLM — un encodeur Inception prédit
  (γₛ, βₛ) par style pour moduler les feature maps de l'Instance Normalisation
- **6 styles artistiques** : baroque, contemporary, cubism, early renaissance,
  impressionism, ukiyo-e
- **Modèle pré-entraîné** : chargement immédiat depuis `style_transfert_data/`
- **Inférence interactive** : upload d'image ou image aléatoire, choix du style

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Lancement

```bash
streamlit run app.py
```

Tout est accessible depuis l'interface.

### Données requises

Les données sont disponibles sur Google Drive :
**[Télécharger les données (Google Drive)](https://drive.google.com/drive/folders/1iDCvrEsCxbZnzT8MIaPBQtsrr4NGieKj)**

- **Sort of CLEVR** : placer `data_train.h5`, `data_train.csv`, `data_val.h5`,
  `data_val.csv`, `data_test.h5`, `data_test.csv` et `model_weights.pth`
  dans le dossier `sortofclevr/`
- **Style Transfer** : placer le dossier `style_transfert_data/` à la racine
  du projet (contient `10k_img_resized/`, `img_style_resized/` et
  `StyleTransfer_weights.pth`)

---

## CLEVR VQA

Pour reproduire les résultats sur le CLEVR complet :

**1. Télécharger CLEVR v1.0**

Télécharger depuis [cs.stanford.edu/people/jcjohns/clevr](https://cs.stanford.edu/people/jcjohns/clevr/)
et extraire dans `C:/data/CLEVR_v1.0`.

**2. Extraire les features ResNet101** (une seule fois, ~20 min sur GPU)

```bash
python data/extract_features.py --data-dir C:/data/CLEVR_v1.0 --split train --max-images 70000
python data/extract_features.py --data-dir C:/data/CLEVR_v1.0 --split val   --max-images 1000
```

Génère `CLEVR_v1.0/features_train.h5` et `features_val.h5` de forme `(N, 1024, 14, 14)`.

**3. Entraîner via CLI**

```bash
python -m src.train --config configs/default.yaml
```

---

## Configuration (`configs/default.yaml`)

| Paramètre | Défaut | Notes |
|-----------|--------|-------|
| `max_samples_train` | 700 000 | Toutes les questions CLEVR (~70k images) |
| `max_samples_val` | 9 900 | Limité par le nombre d'images val extraites |
| `num_blocks` | 4 | Blocs résiduels FiLM |
| `num_channels` | 128 | Largeur des feature maps |
| `hidden_dim` | 256 | Dimension cachée GRU (papier : 4096) |
| `embedding_dim` | 300 | Dimension des embeddings de mots |
| `learning_rate` | 3e-4 | Optimiseur Adam |
| `num_epochs` | 30 | Avec early stopping (patience = 10) |

---

## Résultats

| Dataset | Questions | hidden\_dim | Val Accuracy |
|---------|-----------|-------------|--------------|
| Sort of CLEVR | ~70 000 | 128 | ~94 % (10 epochs) |
| CLEVR VQA | 700 000 | 256 | ~65–70 % (30 epochs) |
| CLEVR VQA | 700 000 | 4 096 | **97,7 %** (papier, 80 epochs) |

L'écart avec le papier vient du `hidden_dim` réduit dans l'encodeur GRU.
Augmenter à 1024–2048 permet de se rapprocher significativement des résultats originaux.

---

## Structure du projet

```
FiLM-DL-Project/
├── app.py                    # Page d'accueil Streamlit
├── pages/
│   ├── 1_sort_of_clevr.py   # Entraînement interactif + test visuel
│   ├── 2_clevr_vqa.py        # Résultats CLEVR VQA
│   └── 3_style_transfer.py  # Transfert de style CIN
├── sortofclevr/
│   ├── dataset.py            # HDF5Dataset + CLASSES
│   ├── model.py              # SortOfClevrFiLMModel
│   └── train.py              # Boucle d'entraînement
├── style_transfer/
│   ├── dataset.py            # Dataset images + styles
│   ├── model.py              # StyleTransferNetwork + VGGExtractor
│   └── train.py              # Entraînement + inférence
├── clevr/                    # Pipeline CLEVR complet (papier)
│   ├── models/
│   │   ├── filmed_net.py     # FiLMedNet (architecture papier)
│   │   ├── film_gen.py       # Générateur GRU → (γ, β)
│   │   └── layers.py
│   └── scripts/
├── data/
│   ├── download_clevr.py     # Téléchargement subset CLEVR
│   └── extract_features.py   # Extraction features ResNet101 → HDF5
├── configs/
│   └── default.yaml          # Hyperparamètres par défaut
└── requirements.txt
```

---

## Références

- Perez et al. (2018) — [FiLM: Visual Reasoning with a General Conditioning Layer](https://arxiv.org/abs/1709.07871)
- Johnson et al. (2017) — [CLEVR: A Diagnostic Dataset for Compositional Language and Elementary Visual Reasoning](https://arxiv.org/abs/1612.06890)
- Dumoulin et al. (2017) — [A Learned Representation For Artistic Style](https://arxiv.org/abs/1610.07629)
- Ghiasi et al. (2017) — [Exploring the structure of a real-time, arbitrary neural artistic stylization network](https://arxiv.org/pdf/1705.06830)
- Code original : [github.com/ethanjperez/film](https://github.com/ethanjperez/film)
