import os
import torchvision.datasets as datasets
from PIL import Image
import numpy as np


data_dir = "./data/cifar10"
os.makedirs(data_dir, exist_ok=True)

cifar10_train = datasets.CIFAR10(root=data_dir, train=True, download=True)
cifar10_test = datasets.CIFAR10(root=data_dir, train=False, download=True)

image_dir = "./data/cifar10_images"
os.makedirs(image_dir, exist_ok=True)

for idx, (img, label) in enumerate(cifar10_train):
    img_path = os.path.join(image_dir, f"train_{idx}.png")
    img.save(img_path)

for idx, (img, label) in enumerate(cifar10_test):
    img_path = os.path.join(image_dir, f"test_{idx}.png")
    img.save(img_path)
