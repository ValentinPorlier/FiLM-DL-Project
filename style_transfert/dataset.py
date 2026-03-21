

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