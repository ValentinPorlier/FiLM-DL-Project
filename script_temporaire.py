import numpy as np
import pandas as pd

import h5py
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import tensordict

from torch.utils.data import DataLoader, Dataset

import torchvision
import torchvision.transforms as transforms

import opendatasets as od

import os

import ast
"""
dataset_url = 'https://www.kaggle.com/datasets/timoboz/clevr-dataset'
od.download(dataset_url)

data_dir = './sortofclevr/'
images_dir = os.path.join(data_dir, "images/train")
qst_dir = os.path.join(data_dir, "questions/CLEVR_train_questions.json")

val_dir = os.path.join(data_dir, "data_val")
test_dir = os.path.join(data_dir, "data_test")

"""

#fonctions issues de https://github.com/mvsjober/pytorch-hdf5/blob/master/pytorch_dvc_cnn.py
class HDF5Dataset(Dataset):
    def __init__(self, h5_path, dataset_name,csv_path, transform=None):
        self.h5_path = h5_path
        self.dataset_name = dataset_name
        self.transform = transform
        
        self.df = pd.read_csv(csv_path)
        self.length = len(self.df)

        self.ds_qst = self.df['question']
        self.ds_ans = self.df['answer']
        self.ds_enc = self.df['encoding']

    def __len__(self):
        return self.length
        

    def _open_hdf5(self):
        self._hf = h5py.File(self.h5_path, 'r')
        self._ds_img = self._hf[self.dataset_name]

    def __getitem__(self, index):
        if not hasattr(self, '_hf'):
            self._open_hdf5()
        
        data = self._ds_img[index] 
        img = torch.from_numpy(data).permute(2, 0, 1).float() / 255.0

        #if self.transform:
        #    img = self.transform(img)

        qst = self.df['question'].iloc[index]
        enc = self.df['encoding'].iloc[index]
        if isinstance(enc, str):
            enc = ast.literal_eval(enc)
        enc = torch.tensor(enc, dtype=torch.float32)
        ans = torch.tensor(self.df['answer'].iloc[index], dtype=torch.long)
        
      
        return qst, img, ans, enc

#dataset = HDF5Dataset(h5_path='./sortofclevr/data_train.h5', dataset_name='data_train', csv_path='./sortofclevr/data_train.csv')

#qst, img, ans, enc = dataset[0]
#print(f"Question : {enc}")
"""
img_for_plot = img.permute(1, 2, 0) 
plt.xlabel(qst)
plt.imshow(img_for_plot.numpy()) 
plt.show()
"""

    
    
"""def get_train_loader_hdf5(batch_size=64):
    print('Train: ', end="")
    train_dataset = HDF5Dataset(os.path.join(DATADIR, 'dogs-vs-cats.hdf5'), 'train',
                                transform=data_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers)
    print('Found', len(train_dataset), 'images belonging to',
          len(train_dataset.classes), 'classes')
    return train_loader"""

"""
with h5py.File('./sortofclevr/data_train.h5', 'r') as f:
    # On accède juste aux métadonnées du dataset
    dataset = f['data_train']
    print(f"Shape: {dataset.shape}")
    print(f"Type des données: {dataset.dtype}")
"""


class CNN_feature_map(nn.Module):
    def __init__(self, in_channels, out_channels=128):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels,out_channels, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(out_channels,out_channels, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(out_channels,out_channels, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(out_channels,out_channels, kernel_size=4, stride=2, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(out_channels)
        self.batchnorm2 = nn.BatchNorm2d(out_channels)
        self.batchnorm3 = nn.BatchNorm2d(out_channels)
        self.batchnorm4 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.relu(x)
        
        x = self.conv3(x)
        x = self.batchnorm3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.batchnorm4(x)
        x = self.relu(x)

        return x
    
def add_coordinate_maps(x):
    """ Ajoute deux canaux (X et Y) allant de -1 à 1 au tenseur d'entrée """
    bs, _, h, w = x.shape
    # Création de la grille de coordonnées
    y_coords = torch.linspace(-1, 1, h).view(1, 1, h, 1).expand(bs, 1, h, w).to(x.device)
    x_coords = torch.linspace(-1, 1, w).view(1, 1, 1, w).expand(bs, 1, h, w).to(x.device)
    # On les colle aux 128 canaux existants -> total 130 canaux
    return torch.cat([x, y_coords, x_coords], dim=1)

class FiLMResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, qst_dim=10):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels+2, out_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
        self.batchnorm = nn.BatchNorm2d(out_channels, affine=False)

        self.film_generator = nn.Linear(qst_dim, out_channels * 2)

    def forward(self, x, encoding):
        x = add_coordinate_maps(x)
        x = torch.relu(self.conv1(x))
        id = x
        x = self.conv2(x)
        x = self.batchnorm(x)
        
        params = self.film_generator(encoding).unsqueeze(2).unsqueeze(3)
        gamma, beta = params.chunk(2, dim=1)
        x = gamma * x + beta
        
        x = torch.relu(x) + id
        return x


class FiLMClassifier(nn.Module):
    def __init__(self, in_channels=128, num_answers=11):
        super().__init__()
        #Conv 1x1 pour monter à 512 channels
        self.conv_1x1 = nn.Conv2d(in_channels, 512, kernel_size=1)
        self.bn = nn.BatchNorm2d(512)
        
        #MLP à deux couches
        self.mlp = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_answers)
        )

    def forward(self, x):
        #x shape: (batch, 128, H, W)
        x = torch.relu(self.bn(self.conv_1x1(x)))
        
        #Global Max Pooling : (batch, 512, H, W) -> (batch, 512, 1, 1)
        x = torch.max(x.view(x.size(0), x.size(1), -1), dim=2)[0]
        
        #Passage dans le MLP
        out = self.mlp(x)
        return out # PyTorch CrossEntropyLoss appliquera le Softmax lui-même
    

class SortOfClevrFiLMModel(nn.Module):
    def __init__(self, num_answers):
        super().__init__()
        
        # 1. Extraction initiale des features (CNN)
        # On passe de 3 (RGB) à 128 feature maps
        self.conv_input = CNN_feature_map(in_channels=3, out_channels=128)
        
        # 2. Blocs Résiduels avec FiLM (Raisonnement Visuel)
        # On peut en mettre 4 comme dans le papier original
        self.res_block1 = FiLMResBlock(128, 128)
        self.res_block2 = FiLMResBlock(128, 128)
        self.res_block3 = FiLMResBlock(128, 128)
        self.res_block4 = FiLMResBlock(128, 128)
        
        # 3. Le Classifieur Final
        self.classifier = FiLMClassifier(in_channels=128, num_answers=num_answers)

    def forward(self, img, q_enc):
        # A. Feature maps de base
        x = self.conv_input(img) # Sortie: (batch, 128, 64, 64)
        
        # B. Passage dans les blocs FiLM (la question module l'image)
        x = self.res_block1(x, q_enc)
        x = self.res_block2(x, q_enc)
        x = self.res_block3(x, q_enc)
        x = self.res_block4(x, q_enc)
        
        # C. Décision finale
        logits = self.classifier(x)
        return logits


import torch.optim as optim

# 1. Calcul du nombre de classes (réponses)
num_classes = 11

# 2. Instanciation du modèle
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SortOfClevrFiLMModel(num_answers=11).to(device)
device
# 3. Loss et Optimiseur
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)



from tqdm import tqdm

def train_model(model, dataloader, optimizer, criterion, epochs=10):
    model.train()
    
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        # On entoure le dataloader avec tqdm pour avoir la barre
        # desc permet d'afficher le numéro de l'époque à gauche
        loop = tqdm(dataloader, desc=f"Epoch [{epoch+1}/{epochs}]", unit="batch")
        
        for question_string, images, labels, questions in loop:
            # Envoi sur GPU/CPU
            images = images.to(device)
            questions = questions.to(device)
            labels = labels.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(images, questions)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Calcul des stats
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Mise à jour des informations à DROITE de la barre
            current_acc = 100 * correct / total
            loop.set_postfix(loss=loss.item(), acc=f"{current_acc:.2f}%")
        
        # Log de fin d'époque
        epoch_loss = running_loss / len(dataloader)
        epoch_acc = 100 * correct / total
        print(f"--- Fin Epoch {epoch+1} | Loss Moyenne: {epoch_loss:.4f} | Acc: {epoch_acc:.2f}% ---")


def evaluate_per_class(model, test_loader, device, classes):
    model.eval()
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    print("\n--- Analyse des performances par classe ---")
    
    with torch.no_grad():
        for _, images, labels, questions in test_loader:
            # TRANSFERT VERS GPU (Crucial !)
            images = images.to(device)
            questions = questions.to(device)
            labels = labels.to(device)

            outputs = model(images, questions)
            predictions = torch.argmax(outputs, 1)

            for target, prediction in zip(labels, predictions):
                # .item() est important pour transformer le tensor en index Python
                class_name = classes[target.item()] 
                if target == prediction:
                    correct_pred[class_name] += 1
                total_pred[class_name] += 1

    # Affichage des résultats
    print(correct_pred)
    """
    for classname, correct_count in correct_pred.items():
        if total_pred[classname] > 0:
            accuracy = 100 * float(correct_count) / total_pred[classname]
            print(f'Accuracy for class: {classname:10s} is {accuracy:.1f} %')
        else:
            print(f'Accuracy for class: {classname:10s} is N/A (pas de données)')
    """


if __name__ == '__main__':
    # Tout ce qui lance du calcul doit être ICI
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Utilisation de : {device}")

    # Initialisation
    train_dataset = HDF5Dataset('./sortofclevr/data_train.h5', 'data_train', './sortofclevr/data_train.csv')
    train_loader = DataLoader(train_dataset, batch_size=512, num_workers=12, pin_memory=True,shuffle=True)



    test_dataset = HDF5Dataset('./sortofclevr/data_test.h5', 'data_test', './sortofclevr/data_test.csv')
    test_loader = DataLoader(test_dataset, batch_size=512,num_workers=12,shuffle=True)

    #model = SortOfClevrFiLMModel(num_answers=11).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    # Lancement
    classes = ['right', 'blue', 'circle', 'left', 'bottom', 'yellow', 'square', 'green', 'red', 'top', 'gray']
    train_model(model, train_loader, optimizer, criterion, epochs= 3)
    evaluate_per_class(model, test_loader, device, classes)

def predict_img(img, question):
    """
    question: line
    """
    model.eval()
    with torch.no_grad():
        ### TODO: perform a classification of a given tex
        # Create a vectorize version of your text
        qst_enc = question["encoding"]
        print(qst_enc)
        if isinstance(qst_enc, str):
            qst_enc = ast.literal_eval(qst_enc)
        test_vector =  torch.tensor(qst_enc, dtype=torch.float32).unsqueeze(0).to(device)
        # Perform a prediction
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        img = img.unsqueeze(0)
        prediction = model(img,test_vector)
        # get the label
        ans = question["answer"]
        pred_label = torch.argmax(prediction)
        ###
        labels = ['right', 'blue', 'circle', 'left', 'bottom', 'yellow', 'square',
       'green', 'red', 'top', 'gray']
        qst_txt = question["question"]
        for i, lab in enumerate(labels):
           if pred_label == i:
              print(f"{qst_txt}, le modèle dit {lab}, la vrai réponse est {ans}")



# Initialisation

batch_sz = 512
lr=1e-3

train_dataset = HDF5Dataset('./sortofclevr/data_train.h5', 'data_train', './sortofclevr/data_train.csv')
train_loader = DataLoader(train_dataset, batch_size=batch_sz)

test_dataset = HDF5Dataset('./sortofclevr/data_test.h5', 'data_test', './sortofclevr/data_test.csv')
test_loader = DataLoader(test_dataset, batch_size=batch_sz)


train_dataset[675][2]
# Modèle
model = SortOfClevrFiLMModel(num_answers=11).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

# GO !
train_model(model, train_loader, optimizer, criterion, epochs=10)
classes = ['right', 'blue', 'circle', 'left', 'bottom', 'yellow', 'square', 'green', 'red', 'top', 'gray']
evaluate_per_class(model, test_loader, device, classes)


labels = pd.read_csv('./sortofclevr/data_train.csv')["answer"].unique()
labels


import h5py
idx = 5
with h5py.File('./sortofclevr/data_test.h5', 'r') as f:
    # On accède juste aux métadonnées du dataset
    img_test = f['data_test'][5]
    print(f"Shape: {img_test.shape}")
    print(f"Type des données: {img_test.dtype}")




img = torch.from_numpy(img_test).permute(2, 0, 1).float() / 255.0
img = img.unsqueeze(0)
img.shape

qst_ds_test = pd.read_csv('./sortofclevr/data_test.csv')
qst_ds_test["answer"].nunique()

predict_img(img_test, qst_ds_test)

plt.imshow(img_test/255.0)
plt.show()

"""


def evaluate_model(model, test_loader, device):
    model.eval() # On passe en mode évaluation
    correct = 0
    total = 0
    
    with torch.no_grad(): # On désactive le calcul des gradients (gain de mémoire)
        for _, images, labels, questions in test_loader:
            # TRANSFERT VERS GPU (L'étape qui manquait !)
            images = images.to(device)
            questions = questions.to(device)
            labels = labels.to(device)
            
            outputs = model(images, questions)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    accuracy = 100 * correct / total
    print(f"✅ Accuracy sur le test set : {accuracy:.2f}%")

#num_examples
#predicted_labels
#label

#print("Accuracy: ", np.round(float((correct_pred/num_examples)),4) * 100, "%")






"""
classes = ['right', 'blue', 'circle', 'left', 'bottom', 'yellow', 'square',
       'green', 'red', 'top', 'gray']
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}

with torch.no_grad():
    for question_string, images, labels, questions in test_loader:
        
        questions = questions.to(device)
        print(questions.shape, images.shape)
        labels = labels.to(device)
        ### TODO: compute the messages labels
        outputs = model(images, questions)
        predictions = torch.argmax(outputs, 1)
        # collect the correct predictions for each class within the dictionnary correct pred
        for target, prediction in zip(labels, predictions):
            if target == prediction:
                correct_pred[classes[target]] += 1
            total_pred[classes[target]] += 1
        ###

### TODO: print accuracy for each class
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')
###

#correct_pred
#test_dataset
