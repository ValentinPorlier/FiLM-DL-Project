import random
import os

from tqdm import tqdm

from .modele import VGGExtractor, compute_loss, StyleTransferNetwork
from .dataset import Dataset_ImageAndStyle

import torch

from torchvision import transforms

from torch.utils.data import DataLoader

from PIL import Image




def prepare_styletransfer_modele(path_to_data, batch_size=128):
    """Prepare les datasets, dataloaders, modèle et device pour l'entraînement."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Calculs effectués sur: {device}")

    path_to_data = "./style_transfert_data/"
    lesdatasets = Dataset_ImageAndStyle(path_to_data, transform=transforms.ToTensor())

    dataloader = DataLoader(
        lesdatasets, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = StyleTransferNetwork().to(device)
    

    return model, dataloader, device


def train_model_styletransfer(model, dataloader, device, epochs:int = 10, lr : float = 1e-3, lambda_style = 1000000):
    print("Lancement de l'entraînement...")

    history = {"train_loss": []}
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    vgg = VGGExtractor().to(device)

    for epoch in range(epochs):
        run_loss = 0.0
        model.train()

        loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")

        for images, styles in loop:

            images = images.to(device)
            styles = styles.to(device)

            optimizer.zero_grad()

            outputs = model(images, styles)


            content_loss, style_loss = compute_loss(vgg, outputs, images, styles)
            loss = content_loss + (lambda_style * style_loss)

            loss.backward()

            optimizer.step()

            run_loss += loss.item()

            loop.set_postfix(loss=f"{loss.item():.4f}")

        t_loss = run_loss / len(dataloader)
        history["train_loss"].append(t_loss)
        print(f"  => Train loss {t_loss:.4f}", flush=True)

def charger_image_aleatoire(dossier, style=None):
    #Lister les fichiers images valides
    extensions = ('.jpg', '.jpeg', '.png')
    pattern = f"{style}*" if style else "*"
    tous_les_fichiers = list(dossier.glob(pattern))
        
    liste_fichiers = [f for f in tous_les_fichiers if f.suffix.lower() in extensions]
    
    if not liste_fichiers:
        raise FileNotFoundError(f"Aucune image trouvée dans le dossier : {dossier}")
    
    #Choisir un fichier au hasard
    chemin_image = random.choice(liste_fichiers)

    
    #Ouvrir, convertir en RGB et appliquer le transform
    img = Image.open(chemin_image).convert('RGB')
    img_tensor = transforms.ToTensor()(img)
    
    #Ajouter la dimension batch [1, 3, 256, 256]
    return img_tensor.unsqueeze(0), chemin_image.name

def preparer_pour_plot(tenseur):
    #On enlève le batch, on ramène sur CPU, on borne entre 0 et 1 les couleurs
    t = tenseur.squeeze(0).cpu().clamp(0, 1)
    #On déplace les canaux (C, H, W) -> (H, W, C)
    t = t.permute(1, 2, 0)
    return t.numpy()