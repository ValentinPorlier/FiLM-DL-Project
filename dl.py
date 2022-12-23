import argparse
import os

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st


def get_FashionMNIST_datasets(batch_size=64, only_loader=True):
    # Download training data from open datasets.
    training_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
    )

    # Download test data from open datasets.
    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor(),
    )

    # Create data loaders.
    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    if only_loader:
        return train_dataloader, test_dataloader
    else:
        return train_dataloader, test_dataloader, training_data, test_data


# Define model
class FMNIST_MLP(nn.Module):
    def __init__(self, hidden_layers=2):
        super().__init__()
        self.flatten = nn.Flatten()
        list_hidden = []
        for _ in range(hidden_layers - 1):
            list_hidden.append(nn.Linear(512, 512))
            list_hidden.append(nn.ReLU())
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            *list_hidden,
            nn.Linear(512, 10),
        )
        self.metrics = pd.DataFrame(
            columns=["train_loss", "train_acc", "test_loss", "test_acc"]
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

    def set_metrics(self, df):
        self.metrics = df

    def update_metrics(self, series):
        self.metrics = pd.concat([self.metrics, series.to_frame().T], ignore_index=True)


def train(dataloader, model, loss_fn, optimizer, device, mode=None):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.train()
    train_loss, correct = 0, 0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        train_loss += loss.item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            if mode == "script":
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    train_loss /= num_batches
    correct /= size
    return train_loss, correct


def test(dataloader, model, loss_fn, device, mode=None):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    if mode == "script":
        print(
            f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
        )
    return test_loss, correct


def get_and_train_model(
    train_dataloader, test_dataloader, hidden_layers=2, epochs=5, mode=None
):
    if not os.path.exists("saved_models"):
        os.mkdir("saved_models")

    # Get cpu or gpu device for training.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if mode == "script":
        print(f"Using {device} device")
    model = FMNIST_MLP(hidden_layers)
    base_name = "saved_models/fmnist_mlp_hidden=" + str(hidden_layers)
    path = base_name + ".pth"
    path_metrics = base_name + "_metrics.csv"
    if os.path.exists(path):
        if mode == "script":
            print("model already exists, let us just load it")
        elif mode == "st":
            st.write("Found a saved model with given config")
        model.load_state_dict(torch.load(path))
        metrics = pd.read_csv(path_metrics, index_col=0)
        model.set_metrics(metrics)
    model = model.to(device)
    if mode == "script":
        print(model)
    elif mode == "st":
        st.text("Model architecture:")
        st.text(model)
    if not os.path.exists(path):
        if mode == "script":
            print("no existing model found")
        elif mode == "st":
            st.write("Didn't find an existing model, training a new one")
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        for t in range(epochs):
            if mode == "script":
                print(f"Epoch {t+1}\n-------------------------------")
            train_loss, train_acc = train(
                train_dataloader, model, loss_fn, optimizer, device, mode
            )
            test_loss, test_acc = test(test_dataloader, model, loss_fn, device, mode)
            new_row = pd.Series(
                {
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "test_loss": test_loss,
                    "test_acc": test_acc,
                }
            )
            model.update_metrics(new_row)
            if (mode == "st") & ((t + 1) % 10 == 0):
                st.text(
                    f"End of epoch {t+1}, Test Error:\n Accuracy: {(100 * new_row['test_acc']):>0.1f}%, Avg loss: {new_row['test_loss']:>8f}"
                )

        if mode == "script":
            print("Done!")
        torch.save(model.state_dict(), path)
        model.metrics.to_csv(path_metrics)
        if mode == "script":
            print("Saved PyTorch Model State to " + path)
    if mode == "script":
        print(model.metrics)

    fig = plt.figure()
    fig.set_figheight(10)
    padding = 2
    model.metrics.index = np.array(model.metrics.index) + 1
    plt.subplot(2, 1, 1)
    plt.tight_layout(pad=padding)
    plt.plot(model.metrics.index, "train_loss", data=model.metrics)
    plt.plot(model.metrics.index, "test_loss", data=model.metrics)
    plt.legend()
    plt.title("Train and test loss during training")
    plt.subplot(2, 1, 2)
    plt.tight_layout(pad=padding)
    plt.plot(model.metrics.index, "train_acc", data=model.metrics)
    plt.plot(model.metrics.index, "test_acc", data=model.metrics)
    plt.legend()
    plt.title("Train and test accuracy during training")
    if mode == "script":
        plt.show()
    elif mode == "st":
        st.pyplot(fig)
    return model


if __name__ == "__main__":

    mode = "script"

    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--epochs", type=int, default=5)
    parser.add_argument("--hidden", type=int, default=2)
    args = parser.parse_args()

    train_dataloader, test_dataloader = get_FashionMNIST_datasets(64)

    for X, y in test_dataloader:
        print(f"Shape of X [N, C, H, W]: {X.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        break

    mod = get_and_train_model(
        train_dataloader,
        test_dataloader,
        hidden_layers=args.hidden,
        epochs=args.epochs,
        mode=mode,
    )
