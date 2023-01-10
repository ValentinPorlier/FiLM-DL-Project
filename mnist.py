"""This module contains the function used for visualization of the MNIST dataset.

Notes
-----
Inspired from https://nextjournal.com/gkoehler/pytorch-mnist.
"""

import streamlit as st
import torch
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt


def mnist_viz():
    """Selects randomly 6 images from the training MNIST dataset and displays them.

    Returns
    -------
    None

    """

    train_data = MNIST("data", train=True, download=True, transform=ToTensor())

    fig = plt.figure()
    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.tight_layout()
        sample_idx = torch.randint(len(train_data), size=(1,)).item()
        plt.imshow(train_data[sample_idx][0][0], cmap="gray", interpolation="none")
        plt.title("Ground Truth: {}".format(train_data[sample_idx][1]))
        plt.xticks([])
        plt.yticks([])
    st.pyplot(fig)
