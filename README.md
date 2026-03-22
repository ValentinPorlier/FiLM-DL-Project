# FiLM — Feature-wise Linear Modulation

Implémentation de [Perez et al. (2018)](https://arxiv.org/abs/1709.07871) avec une application Streamlit interactive.
On couvre trois cas d'usage : Sort of CLEVR (Kaggle), CLEVR VQA (le vrai benchmark du papier), et du transfert de style via Conditional Instance Normalisation.

## C'est quoi FiLM ?

FiLM conditionne un CNN visuel sur un contexte externe (une question, un style…) en appliquant une transformation affine sur ses feature maps :

$$\text{FiLM}(F_{i,c}) = \gamma_{i,c} \cdot F_{i,c} + \beta_{i,c}$$

Un réseau auxiliaire (le *FiLM generator*) produit les paramètres $(\gamma, \beta)$ à partir du contexte.
On utilise $\gamma = 1 + \Delta\gamma$ pour que le modèle démarre proche de l'identité.

## Lancement

```bash
pip install -r requirements.txt
streamlit run app.py
```

Tout se fait depuis l'interface — pas besoin de toucher au terminal après ça.

## Données

Les données et les poids ne sont **pas dans le dépôt** (trop lourds), ils sont sur Google Drive et téléchargeables depuis l'app :
**[(Google Drive)](https://drive.google.com/drive/folders/1iDCvrEsCxbZnzT8MIaPBQtsrr4NGieKj)**

| Dataset | Taille | Comment l'obtenir |
|---------|--------|-------------------|
| Sort of CLEVR | ~200 Mo | Bouton dans l'app |
| Style Transfer | ~400 Mo | Bouton dans l'app |
| CLEVR VQA | ~18 Go | Manuel (voir plus bas) |

## Pages de l'app

### Sort of CLEVR
Dataset 2D Kaggle, formes colorées, 11 classes de réponses. La question est encodée en 10 dimensions et passée au FiLM generator. On peut entraîner depuis l'app, charger un modèle pré-entraîné, et tester visuellement.

### CLEVR VQA
Implémentation de l'architecture complète du papier. Le dataset fait ~18 Go donc on ne propose pas d'entraînement interactif — on affiche les courbes de notre propre run (~40k itérations).

Pour reproduire :

```bash
# Prétraitement
python -m clevr.scripts.preprocess_questions \
  --input_questions_json CLEVR_v1.0/questions/CLEVR_train_questions.json \
  --output_h5_file data/train_questions.h5 --output_vocab_json data/vocab.json

# Extraction features ResNet101
python data/extract_features.py --data-dir CLEVR_v1.0 --split train

# Entraînement
python -m clevr.scripts.train_model --model_type FiLM \
  --checkpoint_path data/film_checkpoint.pth --batch_size 64 \
  --num_iterations 100000 --loader_num_workers 0
```

### Style Transfer
Implémentation de [Ghiasi et al. (2017)](https://arxiv.org/pdf/1705.06830) — même idée que FiLM appliquée au style via Conditional Instance Normalisation. 6 styles disponibles, inférence interactive depuis l'app.

## Résultats

| Dataset | Val Accuracy |
|---------|--------------|
| Sort of CLEVR | ~94 % (10 epochs) |
| CLEVR VQA (notre run, 40k iters) | ~51 % |
| CLEVR VQA (papier, 80 epochs complets) | 97,7 % |

L'écart avec le papier vient surtout du `hidden_dim` réduit (256 vs 4096) et du nombre d'itérations limité.

## Structure

```
FiLM-DL-Project/
├── app.py
├── pages/
│   ├── 0_Présentation.py
│   ├── 1_Sort_of_CLEVR.py
│   ├── 2_CLEVR_VQA.py
│   └── 3_Style_Transfer.py
├── sortofclevr/          # dataset, model, train
├── style_transfer/       # dataset, model, train
├── clevr/
│   ├── core/             # data, embedding, preprocess, programs, utils
│   ├── models/           # filmed_net, film_gen, baselines, layers...
│   └── scripts/          # train_model, preprocess_questions, extract_features
├── data/                 # vocab, questions h5, résultats json
├── assets/
├── configs/
└── requirements.txt
```

## Branches

- `main` — version stable
- `ilies` — Sort of CLEVR + CLEVR VQA
- `Valentin` — Style Transfer

## Références

- Perez et al. (2018) — [FiLM: Visual Reasoning with a General Conditioning Layer](https://arxiv.org/abs/1709.07871)
- Johnson et al. (2017) — [CLEVR: A Diagnostic Dataset](https://arxiv.org/abs/1612.06890)
- Ghiasi et al. (2017) — [Arbitrary neural artistic stylization](https://arxiv.org/pdf/1705.06830)
- Code original CLEVR : [github.com/ethanjperez/film](https://github.com/ethanjperez/film)
