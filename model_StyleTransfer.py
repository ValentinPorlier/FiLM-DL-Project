"""Architecture FiLM pour Style Transfert, issu de l'article de Ghiasi et al. : https://arxiv.org/pdf/1705.06830

Pipeline :
    Image RGB →  FiLMed Network  →  Image stylisée RGB
    Style RGB -> InceptionV3 (jusqu'à Mixed_6e) -> Bottleneck linéaire -> Gammas/Betas pour FiLM


"""


import random

from PIL import Image


from numpy.char import index
import torch
import torch.nn as nn
import queue as _queue

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import random
from torch.utils.data import Dataset

from torchvision.datasets import ImageFolder


import torch
import torchvision.models as models


class InceptionMixed6e(nn.Module):
    def __init__(self):
        super().__init__()
        #On charge le modèle avec les poids officiels
        inception = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)
        inception.eval() #Mode évaluation (important pour Dropout/BatchNorm)
        
        #On regroupe les couches dans l'ordre exact jusqu'à Mixed_6e
        self.features = nn.Sequential(
            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2),
            inception.Conv2d_3b_1x1,
            inception.Conv2d_4a_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2),
            inception.Mixed_5b,
            inception.Mixed_5c,
            inception.Mixed_5d,
            inception.Mixed_6a,
            inception.Mixed_6b,
            inception.Mixed_6c,
            inception.Mixed_6d,
            inception.Mixed_6e
        )
        
        #On gèle les poids puisque le modèle utilise des poids préentrainés
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        #on passe l'image 256x256 qui sortira en 14x14 et 768 canaux
        return self.features(x)

class FiLMGenerator(nn.Module):
    def __init__(self):
        super(FiLMGenerator, self).__init__()
        self.inception = InceptionMixed6e()
        self.bottleneck = nn.Sequential(
            nn.Linear(768, 100),    #On compresse (Bottleneck)
            nn.ReLU(),
            nn.Linear(100, 2758)    #On déploie vers les gammas/betas
        )

    def forward(self, x):
        parms = self.inception(x) #(B, 768, 14, 14)
        parms = torch.mean(parms, dim=[2, 3], keepdim=True) #Global Average Pooling (B, 768, 1, 1)
        parms = self.bottleneck(parms.squeeze()) #On enlève les dimensions 1,1 pour le Linear (B, 768) -> (B, 2758)
        return parms

class StyleTransferNetwork(nn.Module):
    def __init__(self):
        super(StyleTransferNetwork, self).__init__()
        self.FiLMGenerator = FiLMGenerator()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=9, stride=1, padding='same')
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)

        self.res1conv1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding='same')
        self.res1conv2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding='same')

        self.res2conv1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding='same')
        self.res2conv2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding='same')

        self.res3conv1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding='same')
        self.res3conv2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding='same')

        self.res4conv1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding='same')
        self.res4conv2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding='same')

        self.res5conv1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding='same')
        self.res5conv2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding='same')

        self.upsample1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.convUpsample1 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding='same')

        self.upsample2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.convUpsample2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding='same')
        
        self.convfinal = nn.Conv2d(in_channels=32, out_channels=3, kernel_size=9, stride=1, padding='same')

    def apply_film(self, x, gamma, beta):
            #Normalisation d'instance
            eps = 1e-5
            mean = x.mean(dim=(2, 3), keepdim=True)
            var = x.var(dim=(2, 3), keepdim=True)
            x_norm = (x - mean) / torch.sqrt(var + eps)
            return gamma * x_norm + beta
    

    def forward(self, x, vecteur_style):
        params       = self.FiLMGenerator(vecteur_style).unsqueeze(2).unsqueeze(3)  #(B, 2C, 1, 1)
        #gamma, beta  = params.chunk(2, dim=1)
        
        #On crée une fonction qui "consomme" le vecteur au fur et à mesure
        pointer = 0
        
        def get_next_params(num_channels, current_pointer):
            #On prend 2 * num_channels (moitié gamma, moitié beta)
            start = current_pointer
            mid = start + num_channels
            end = mid + num_channels
            
            gamma = params[:,start:mid,:,:] #On prend les gammas
            beta = params[:,mid:end,:,:]   #On prend les betas
            return gamma, beta, end

        #Premieres convolutions
        x = self.conv1(x)
        x = torch.relu(x)

        x = self.conv2(x)
        x = torch.relu(x)

        x = self.conv3(x)
        x = torch.relu(x)

        #ResBlocks

        #Block1
        res = x

        x = self.res1conv1(x)
        gamma, beta, pointer = get_next_params(128, pointer)
        x = self.apply_film(x, gamma, beta)
        x = torch.relu(x)

        x = self.res1conv2(x)
        gamma, beta, pointer = get_next_params(128, pointer)
        x = self.apply_film(x, gamma, beta)
        x = x + res

        #Block2
        res = x

        x = self.res2conv1(x)
        gamma, beta, pointer = get_next_params(128, pointer)
        x = self.apply_film(x, gamma, beta)
        x = torch.relu(x)

        x = self.res2conv2(x)
        gamma, beta, pointer = get_next_params(128, pointer)
        x = self.apply_film(x, gamma, beta)
        x = x + res

        #Block3
        res = x

        x = self.res3conv1(x)
        gamma, beta, pointer = get_next_params(128, pointer)
        x = self.apply_film(x, gamma, beta)
        x =torch.relu(x)

        x = self.res3conv2(x)
        gamma, beta, pointer = get_next_params(128, pointer)
        x = self.apply_film(x, gamma, beta)
        x = x + res

        #Block4
        res = x

        x = self.res4conv1(x)
        gamma, beta, pointer = get_next_params(128, pointer)
        x = self.apply_film(x, gamma, beta)
        x = torch.relu(x)

        x = self.res4conv2(x)
        gamma, beta, pointer = get_next_params(128, pointer)
        x = self.apply_film(x, gamma, beta)
        x = x + res

        #Block5
        res = x

        x = self.res5conv1(x)
        gamma, beta, pointer = get_next_params(128, pointer)
        x = self.apply_film(x, gamma, beta)
        x = torch.relu(x)

        x = self.res5conv2(x)
        gamma, beta, pointer = get_next_params(128, pointer)
        x = self.apply_film(x, gamma, beta)
        x = x + res

        #Upsampling
        x = self.upsample1(x)
        x = self.convUpsample1(x)
        gamma, beta, pointer = get_next_params(64, pointer)
        x = self.apply_film(x, gamma, beta)
        x = torch.relu(x)

        x = self.upsample2(x)
        x = self.convUpsample2(x)
        gamma, beta, pointer = get_next_params(32, pointer)
        x = self.apply_film(x, gamma, beta)
        x = torch.relu(x)

        x = self.convfinal(x)
        gamma, beta, pointer = get_next_params(3, pointer)
        x = self.apply_film(x, gamma, beta)
        x = torch.sigmoid(x)

        return x


class VGGExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        # On charge VGG16 préentraîné
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        self.vgg = vgg[:23] #On s'arrête après relu4_2
        
        #On fige les poids puisque on veut pas l'entrainer
        for param in self.parameters():
            param.requires_grad = False
        
        self.eval()
            
        #indices des couches auxquelles on veut acceder
        self.style_idx = ['3', '8', '15', '22']
        self.content_idx = ['22'] #relu4_2

    def forward(self, x):
        features = {}
        for name, layer in self.vgg._modules.items():
            x = layer(x)
            if name in self.style_idx or name in self.content_idx:
                features[name] = x
        return features
    
def get_gram_matrix(tensor):
    #Batch, Canaux, Hauteur, Largeur
    b, c, h, w = tensor.size()

    #On aplatit les dimensions spatiales (Hauteur et Largeur)
    #(b, c, h, w) à (b, c, h * w)
    features = tensor.view(b, c, h * w)

    #On multiplie la matrice par sa transposée
    #torch.bmm = Batch Matrix Multiplication
    gram = torch.bmm(features, features.transpose(1, 2))

    #On normalise
    return gram / (c * h * w)



def compute_loss(vgg, gen_img, content_img, style_img):
    #Extraire les features pour les 3 images
    gen_feats = vgg(gen_img)
    cont_feats = vgg(content_img)
    style_feats = vgg(style_img)
    
    mse = nn.MSELoss()
    
    #LOSS DE CONTENU
    #On compare relu4_2 de l'image générée et de l'image de contenu
    content_loss = mse(gen_feats['22'], cont_feats['22'])
    
    #LOSS DE STYLE
    style_loss = 0
    style_layers_idx = ['3', '8', '15', '22']
    
    #On parcours la liste d'indice et on extrait les matrices de gram et on calcule la MSE
    for idx in style_layers_idx:
        gen_gram = get_gram_matrix(gen_feats[idx])
        style_gram = get_gram_matrix(style_feats[idx])
        
        style_loss += mse(gen_gram, style_gram)
        

    style_loss /= len(style_layers_idx)
    
    return content_loss, style_loss


def train_model_styletransfer(model, dataloader, optimizer, device, epochs=10, lambda_style = 1e5):

    history = {"train_loss": []}
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

        checkpoint_path = f"checkpoints/styletranfer_lam1e5_weights_{epoch+1}.pth"
        torch.save(model.state_dict(), checkpoint_path)
        print(f"✅ Epoch {epoch+1} terminée ! Modèle sauvegardé sous : {checkpoint_path}")







class Dataset_ImageAndStyle(Dataset):
    def __init__(self, path_to_data, transform, max_samples=None):
        image_path = path_to_data + "10k_img_resized"
        style_path = path_to_data + "img_style_resized"

        if max_samples is not None:
            indices = list(range(max_samples))

        self.image = datasets.ImageFolder(root=image_path, transform=transform)

        self.style_image = datasets.ImageFolder(root=style_path, transform=transform)


    def __len__(self):
        return len(self.image)
    
    def __getitem__(self, idx):
        img_content, _ = self.image[idx]

        random_style_idx = random.randint(0, len(self.style_image) - 1)
        image_style,_ = self.style_image[random_style_idx]
        return img_content, image_style


if __name__ == '__main__':
    transform = transforms.Compose([
    transforms.ToTensor(),
    ])

    path_to_data = "./style_transfert_data/"
    lesdatasets = Dataset_ImageAndStyle(path_to_data, transform)

    dataloader = DataLoader(
        lesdatasets, 
        batch_size=8,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = StyleTransferNetwork().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    print("Lancement de l'entraînement...")
    train_model_styletransfer(model, dataloader, optimizer, device, epochs=10, lambda_style=1000000)

    #torch.save(model.state_dict(), "StyleTransfer_weights.pth")

    print("Entraînement terminé !")
    print("Poids du modèle sauvegardés dans StyleTransfer_weights.pth")

    model.eval()
    img = Image.open("style_transfert_data/img_test/img_test.jpg").convert('RGB')
    style = Image.open("style_transfert_data/img_test/cubism_test.jpg").convert('RGB')

    img = transform(img).unsqueeze(0)  #Ajouter une dimension batch
    style = transform(style).unsqueeze(0)

    img = img.to(device)
    style = style.to(device)

    with torch.no_grad():
        output = model(img, style)

    output = output.squeeze(0).cpu().clamp(0, 1)
    output = transforms.ToPILImage()(output)

    #on l'affiche
    output.show() 
    #on l'enregistre
    output.save("resultat_stylise.jpg")

"""
img = Image.open("style_transfert_data/img_test/img_test.jpg").convert('RGB')
baroque = Image.open("style_transfert_data/img_test/baroque_test.jpg").convert('RGB')
Contemporary = Image.open("style_transfert_data/img_test/Contemporary_test.jpg")
cubism = Image.open("style_transfert_data/img_test/cubism_test.jpg").convert('RGB')
early_renaissance = Image.open("style_transfert_data/img_test/early_renaissance test.jpg")
impressionism = Image.open("style_transfert_data/img_test/impressionism_test.jpg").convert('RGB')
Ukiyo_e = Image.open("style_transfert_data/img_test/Ukiyo_e_test.jpg").convert('RGB')

style_names = ["baroque", "Contemporary", "cubism", "early_renaissance", "impressionism", "Ukiyo_e"]
style_test = [baroque, Contemporary, cubism, early_renaissance, impressionism, Ukiyo_e]
style_list =[]
for style in style_test:
    style_list.append(transforms.ToTensor()(style).unsqueeze(0))


img = transforms.ToTensor()(img).unsqueeze(0)  #Ajouter une dimension batch

img = torch.cat([img, img, img, img, img, img], dim=0)
style = torch.cat(style_list, dim=0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
img = img.to(device)
style = style.to(device)


model = StyleTransferNetwork().to(device)

path_weights = "StyleTransfer_weights.pth"
weights = torch.load(path_weights, map_location=device)

model.load_state_dict(weights)
model.eval()

print("Modèle chargé et prêt pour l'inférence !")

with torch.no_grad():
    # img et style doivent être des tenseurs [1, 3, 256, 256]
    test_output = model(img, style)
    print("Le modèle répond correctement !")

generated_list = []
for i in range(len(test_output)):
    genimg = test_output[i].cpu().clamp(0, 1)
    genimg = transforms.ToPILImage()(genimg)
    generated_list.append(genimg)
    genimg.save(f"resultat_{style_names[i]}.jpg")





"""
import torch
from PIL import Image
from torchvision import transforms
import os
import random
import matplotlib.pyplot as plt
from datetime import datetime


#Dossiers images
DOSSIER_CONTENU = "./style_transfert_data/10k_img_resized/images"  #dataset de 10k images
DOSSIER_STYLE = "./style_transfert_data/img_style_resized/images/"     #images de style

#Chemin de poids de modèle entraîné (.pth)
CHEMIN_POIDS = "checkpoint"
dossier = "checkpoints/lam1e5"
[f for f in os.listdir(dossier)]


#Fonction pour charger UNE image aléatoire et la préparer pour le modèle
def charger_image_aleatoire(dossier):


    #Lister les fichiers images valides
    extensions = ('.jpg', '.jpeg', '.png')
    liste_fichiers = [f for f in os.listdir(dossier) if f.lower().endswith(extensions)]
    
    if not liste_fichiers:
        raise FileNotFoundError(f"Aucune image trouvée dans le dossier : {dossier}")
    
    #Choisir un fichier au hasard
    nom_fichier = random.choice(liste_fichiers)
    chemin_complet = os.path.join(dossier, nom_fichier)
    
    #Ouvrir, convertir en RGB et appliquer le transform
    img = Image.open(chemin_complet).convert('RGB')
    img_tensor = transforms.ToTensor()(img)
    
    #Ajouter la dimension batch [1, 3, 256, 256]
    return img_tensor.unsqueeze(0), nom_fichier

#Fonction pour préparer un tenseur pour Matplotlib (H, W, C, détache du GPU)
def preparer_pour_plot(tenseur):
    #On enlève le batch, on ramène sur CPU, on borne entre 0 et 1 les couleurs
    t = tenseur.squeeze(0).cpu().clamp(0, 1)
    #On déplace les canaux (C, H, W) -> (H, W, C)
    t = t.permute(1, 2, 0)
    return t.numpy()


def generer_collage_aleatoire(poids):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Utilisation du périphérique : {device}")
    model = StyleTransferNetwork().to(device)
    
    try:
        #model.load_state_dict(torch.load(poids, map_location=device, weights_only=True))
        checkpoint = torch.load(poids, map_location=device, weights_only=True)

        # Si c'est un dictionnaire de sauvegarde complet, on extrait juste les poids
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            # Si c'est déjà le state_dict, on utilise strict=False pour voir ce qui passe
            model.load_state_dict(checkpoint, strict=False)
        print("✅ Poids du modèle chargés avec succès !")
    except FileNotFoundError:
        print(f"⚠️ Fichier de poids non trouvé à : {CHEMIN_POIDS}. Le résultat sera aléatoire.")
    
    model.eval()

    #Sélection Aléatoire des Images
    try:
        img_content_tensor, nom_content = charger_image_aleatoire(DOSSIER_CONTENU)
        img_style_tensor, nom_style = charger_image_aleatoire(DOSSIER_STYLE)
        print(f"🎨 Combinaison aléatoire : {nom_content} ➕ {nom_style}")
    except FileNotFoundError as e:
        print(f"❌ Erreur : {e}")
        return

    #Contournement du Bug de Dimension (Batch de 2)
    #On duplique les images pour créer un batch de 2 [2, 3, 256, 256]
    content_batch = torch.cat([img_content_tensor, img_content_tensor], dim=0).to(device)
    style_batch = torch.cat([img_style_tensor, img_style_tensor], dim=0).to(device)

    #Génération du Style
    with torch.no_grad():
        output_batch = model(content_batch, style_batch)
    
    print("✨ Transfert de style terminé !")

    #Assemblage Visuel: Collage 3x1 avec Matplotlib
    fig, axes = plt.subplots(1, 3, figsize=(15, 6)) #1 ligne, 3 colonnes
    
    #Image de Contenu
    axes[0].imshow(preparer_pour_plot(img_content_tensor))
    axes[0].set_title(f"Contenu\n({nom_content})")
    axes[0].axis('off')

    #Image de Style
    axes[1].imshow(preparer_pour_plot(img_style_tensor))
    axes[1].set_title(f"Style\n({nom_style})")
    axes[1].axis('off')

    #Image Résultat (On prend que la première image du batch de 2)
    output_final = preparer_pour_plot(output_batch[0])
    axes[2].imshow(output_final)
    axes[2].set_title("Résultat Final")
    axes[2].axis('off')
    plt.suptitle(f"Poids associé a l'epoch : {poids}")
    plt.tight_layout()

    #Sauvegarde Automatique du Collage
    #On crée un nom unique avec la date et l'heure
    maintenant = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    nom_ext = os.path.basename(poids)
    nom_seul, extension = os.path.splitext(nom_ext)
    nom_sauvegarde = f"./style_transfert_data/img_test/imgs_generees/collage_style_{nom_seul}.jpg"
    
    #sauvegarde la figure Matplotlib
    plt.savefig(nom_sauvegarde, dpi=150, bbox_inches='tight')
    print(f"💾 Collage sauvegardé sous : {nom_sauvegarde}")

    plt.close(fig)

for chemin_poid in os.listdir(dossier):
    chemin_complet = os.path.join(dossier, chemin_poid)
    
    generer_collage_aleatoire(chemin_complet)


