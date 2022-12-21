import streamlit as st
import torch
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt


def mnist_viz():
    # batch_size_train = 64
    # batch_size_test = 1000

    train_data = MNIST("mnist-data/", train=True, download=True, transform=ToTensor())
    # test_data = MNIST("mnist-data/"", train=False, download=True, transform=ToTensor())

    # train_loader = torch.utils.data.DataLoader(
    #     train_data, batch_size=batch_size_train, shuffle=True
    # )
    # test_loader = torch.utils.data.DataLoader(
    #     test_data, batch_size=batch_size_test, shuffle=True
    # )
    # examples = enumerate(train_loader)
    # batch_idx, (example_data, example_targets) = next(examples)
    # fig = plt.figure()
    # for i in range(6):
    #     plt.subplot(2,3,i+1)
    #     plt.tight_layout()
    #     plt.imshow(example_data[i][0], cmap="gray", interpolation="none")
    #     plt.title("Ground Truth: {}".format(example_targets[i]))
    #     plt.xticks([])
    #     plt.yticks([])

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
