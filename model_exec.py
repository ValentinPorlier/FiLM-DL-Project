"""Architecture FiLM pour Sort of CLEVR.

Pipeline :
    Image RGB  →  CNN_feature_map  →  4 × FiLMResBlock  →  FiLMClassifier  →  logits

La question est encodée en un vecteur numérique de taille 10.
Chaque FiLMResBlock prédit localement ses propres γ/β via une couche Linear.
"""


import torch
import torch.nn as nn
from sortofclevr.dataset import CLASSES, NUM_CLASSES
from sortofclevr.dataset import HDF5Dataset 
import queue as _queue

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from sortofclevr.dataset import CLASSES, HDF5Dataset, NUM_CLASSES


def add_coordinate_maps(x: torch.Tensor) -> torch.Tensor:
    """Ajoute deux canaux de coordonnées spatiales (x, y ∈ [−1, 1]) au tenseur.

    Parameters
    ----------
    x : (B, C, H, W)

    Returns
    -------
    (B, C+2, H, W)
    """
    B, _, H, W = x.shape
    y_coords = torch.linspace(-1, 1, H, device=x.device).view(1, 1, H, 1).expand(B, 1, H, W)
    x_coords = torch.linspace(-1, 1, W, device=x.device).view(1, 1, 1, W).expand(B, 1, H, W)
    return torch.cat([x, y_coords, x_coords], dim=1)


class CNN_feature_map(nn.Module):
    """Extracteur de features CNN — 4 convolutions stride-2 avec BatchNorm.

    Réduit la résolution spatiale par un facteur 16 (ex. 128→8).

    Parameters
    ----------
    in_channels  : canaux en entrée (3 pour RGB)
    out_channels : largeur des feature maps (128 par défaut)
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 128) -> None:
        super().__init__()
        layers = []
        ch = in_channels
        for _ in range(4):
            layers += [
                nn.Conv2d(ch, out_channels, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ]
            ch = out_channels
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FiLMResBlock(nn.Module):
    """Bloc résiduel FiLM pour Sort of CLEVR.

    Architecture :
        [x ++ coord]  →  Conv1×1  →  ReLU  (= résidu)
                      →  Conv3×3  →  BN(affine=False)  →  FiLM(γ,β)  →  ReLU
                      +  résidu

    Le générateur FiLM est ici une simple couche Linear(qst_dim → 2C)
    car la question est déjà sous forme d'un vecteur numérique.

    Parameters
    ----------
    in_channels : nombre de canaux en entrée (= out_channels du CNN)
    out_channels: nombre de canaux en sortie
    qst_dim     : dimension de l'encodage de question (10 par défaut)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        qst_dim: int = 10,
    ) -> None:
        super().__init__()
        self.conv1         = nn.Conv2d(in_channels + 2, out_channels, kernel_size=1)
        self.conv2         = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn            = nn.BatchNorm2d(out_channels, affine=False)
        self.film_generator = nn.Linear(qst_dim, out_channels * 2)

    def forward(self, x: torch.Tensor, encoding: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x        : (B, C, H, W)
        encoding : (B, qst_dim)
        """
        x   = add_coordinate_maps(x)        # (B, C+2, H, W)
        x   = torch.relu(self.conv1(x))     # (B, C, H, W)  ← résidu
        res = x

        x = self.conv2(x)
        x = self.bn(x)

        # Paramètres FiLM prédits par le générateur linéaire
        params       = self.film_generator(encoding).unsqueeze(2).unsqueeze(3)  # (B, 2C, 1, 1)
        gamma, beta  = params.chunk(2, dim=1)
        x = (1 + gamma) * x + beta  # formulation résiduelle : gamma=0 → identité au départ

        x = torch.relu(x) + res
        return x


class FiLMClassifier(nn.Module):
    """Tête de classification après les blocs FiLM.

    Pipeline : Conv1×1(C→512) → BN → ReLU → MaxPool global → MLP(512→1024→1024→K)

    Parameters
    ----------
    in_channels : canaux en entrée (128)
    num_answers : nombre de classes de sortie (11 pour Sort of CLEVR)
    """

    def __init__(self, in_channels: int = 128, num_answers: int = 11) -> None:
        super().__init__()
        self.conv_1x1 = nn.Conv2d(in_channels, 512, kernel_size=1)
        self.bn       = nn.BatchNorm2d(512)
        self.mlp      = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_answers),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.bn(self.conv_1x1(x)))            # (B, 512, H, W)
        x = x.view(x.size(0), x.size(1), -1).max(dim=2)[0]  # Global Max Pool → (B, 512)
        return self.mlp(x)


class SortOfClevrFiLMModel(nn.Module):
    """Modèle FiLM complet pour Sort of CLEVR.

    Parameters
    ----------
    num_answers : nombre de classes (11 par défaut)
    """

    def __init__(self, num_answers: int = 11) -> None:
        super().__init__()
        self.featuremap = CNN_feature_map(in_channels=3, out_channels=128)
        self.res_blocks = nn.ModuleList([FiLMResBlock(128, 128) for _ in range(4)])
        self.classifier  = FiLMClassifier(in_channels=128, num_answers=num_answers)

    def forward(self, img: torch.Tensor, encoding: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        img      : (B, 3, H, W)
        encoding : (B, 10)

        Returns
        -------
        logits : (B, num_answers)
        """
        x = self.featuremap(img)
        for block in self.res_blocks:
            x = block(x, encoding)
        return self.classifier(x)






"""Fonctions d'entraînement et d'évaluation pour Sort of CLEVR."""





def train_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device,
    epochs: int = 10,
    progress_queue: _queue.Queue | None = None,
) -> dict:
    """Entraîne le modèle, évalue sur val à chaque epoch, retourne l'historique.

    Si progress_queue est fourni, envoie un dict par epoch :
        {"epoch": int, "num_epochs": int, "train_loss", "train_acc", "val_loss", "val_acc"}
    Et à la fin :
        {"done": True, "history": dict}
    """
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(epochs):
        model.train()
        run_loss, correct, total = 0.0, 0, 0

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for _, images, labels, encodings in loop:
            images    = images.to(device)
            encodings = encodings.to(device)
            labels    = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images, encodings)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            run_loss += loss.item()
            total    += labels.size(0)
            correct  += (torch.argmax(outputs, 1) == labels).sum().item()

            loop.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct/total:.1%}")

        t_loss = run_loss / len(train_loader)
        t_acc  = correct / total
        v_loss, v_acc = evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(t_loss)
        history["train_acc"].append(t_acc)
        history["val_loss"].append(v_loss)
        history["val_acc"].append(v_acc)
        print(f"  => Train {t_acc:.2%} | Val {v_acc:.2%}", flush=True)


        if progress_queue is not None:
            progress_queue.put({
                "epoch": epoch + 1, "num_epochs": epochs,
                "train_loss": t_loss, "train_acc": t_acc,
                "val_loss": v_loss, "val_acc": v_acc,
            })

    if progress_queue is not None:
        progress_queue.put({"done": True, "history": history})
    print(history)
    return history


def evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Retourne (loss_moyenne, accuracy) sur le dataloader."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for _, images, labels, encodings in dataloader:
            images    = images.to(device)
            encodings = encodings.to(device)
            labels    = labels.to(device)
            outputs   = model(images, encodings)
            total_loss += criterion(outputs, labels).item()
            correct    += (torch.argmax(outputs, 1) == labels).sum().item()
            total      += labels.size(0)

    return total_loss / len(dataloader), correct / total


def evaluate_per_class(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """Retourne {classe: accuracy} pour chaque classe."""
    model.eval()
    correct_pred = {c: 0 for c in CLASSES}
    total_pred   = {c: 0 for c in CLASSES}

    with torch.no_grad():
        for _, images, labels, encodings in dataloader:
            images    = images.to(device)
            encodings = encodings.to(device)
            labels    = labels.to(device)
            preds     = torch.argmax(model(images, encodings), 1)

            for target, pred in zip(labels, preds):
                cls = CLASSES[target.item()]
                total_pred[cls]   += 1
                if target == pred:
                    correct_pred[cls] += 1

    return {cls: correct_pred[cls] / total_pred[cls] for cls in CLASSES if total_pred[cls] > 0}


def run(train_h5, train_csv, test_h5, test_csv, epochs=10, batch_size=128, lr=0.001, max_samples=None):
    """Lance un entraînement complet et retourne (history, per_class).

    C'est la fonction à appeler depuis la page Streamlit.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Calculs effectués sur: {device}")

    train_ds = HDF5Dataset(str(train_h5), "data_train", str(train_csv), max_samples=max_samples)
    test_ds  = HDF5Dataset(str(test_h5),  "data_test",  str(test_csv))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=0)

    model     = SortOfClevrFiLMModel(num_answers=NUM_CLASSES).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    history   = train_model(model, train_loader, test_loader, optimizer, criterion, device, epochs=epochs)
    per_class = evaluate_per_class(model, test_loader, device)

    return history, per_class



batch_size = 512
lr = 1e-3
epoch = 10
num_images = 6 #pour la visualisaiton
num_workers = 16

if __name__ == '__main__':
    #ligne qui execute sur gpu avec num_workers qui marche
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Utilisation de : {device}")
    
    #Initialisation
    train_dataset = HDF5Dataset('./sortofclevr/data_train.h5', 'data_train', './sortofclevr/data_train.csv')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True,shuffle=True)

    val_dataset = HDF5Dataset('./sortofclevr/data_val.h5', 'data_val', './sortofclevr/data_val.csv')
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True,shuffle=True)

    test_dataset = HDF5Dataset('./sortofclevr/data_test.h5', 'data_test', './sortofclevr/data_test.csv')
    test_loader = DataLoader(test_dataset, batch_size=batch_size,num_workers=num_workers,shuffle=True)

    model = SortOfClevrFiLMModel(num_answers=11).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    #Lancement
    classes = ['right', 'blue', 'circle', 'left', 'bottom', 'yellow', 'square', 'green', 'red', 'top', 'gray']
    history = train_model(model = model,train_loader= train_loader, val_loader= val_loader, optimizer=optimizer, criterion=criterion, epochs= epoch, device=device)
    print(history)
    torch.save(model.state_dict(), "sortofclevr/model_weights.pth")
    pc = evaluate_per_class(model, test_loader, device)
    print("Accuracy par classe :")
    for cls, acc in pc.items():
        print(f"  {cls}: {acc:.2%}")
