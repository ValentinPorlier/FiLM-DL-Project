"""Présentation du projet FiLM Explorer.

Ce projet implémente FiLM (Feature-wise Linear Modulation),
un mécanisme de conditionnement d'un CNN visuel par une modalité externe
(question en langage naturel ou image de style).

L'idée : un encodeur prédit dynamiquement les paramètres gamma et beta
injectés après chaque Batch Normalisation du CNN.
Cela permet au pipeline visuel d'être modulé par la question
sans partager de paramètres entre couches.

On a validé l'architecture sur trois cas d'usage :

- Sort of CLEVR : dataset Kaggle 2D, entraînement rapide sur CPU (~5 min)
- CLEVR VQA : dataset 3D photoréaliste, 700k questions, features ResNet101
- Style Transfer : transfert de style conditionnel via Conditional Instance
  Normalisation (Ghiasi et al. 2017), une image de style encodée par InceptionV3
  génère les gamma/beta du réseau de transfert.

Références :
    - Perez et al. (2018), FiLM: Visual Reasoning with a General Conditioning Layer
    - Johnson et al. (2017), CLEVR: A Diagnostic Dataset
    - Ghiasi et al. (2017), Exploring the structure of a real-time, arbitrary neural artistic stylization network
"""
