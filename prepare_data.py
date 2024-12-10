import os
import torchvision
import numpy as np
from PIL import Image

def download_and_organize_cifar10(root_dir='./data/cifar10'):
    os.makedirs(root_dir, exist_ok=True)
    train_dir = os.path.join(root_dir, 'train')
    os.makedirs(train_dir, exist_ok=True)

    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    for cls in classes:
        os.makedirs(os.path.join(train_dir, cls), exist_ok=True)
    cifar_dataset = torchvision.datasets.CIFAR10(
        root=root_dir, 
        train=True, 
        download=True
    )

    for idx, (image, label) in enumerate(cifar_dataset):
        if not isinstance(image, np.ndarray):
            image = np.array(image)
        class_name = classes[label]
        image_path = os.path.join(train_dir, class_name, f'{class_name}_{idx}.jpg')
        Image.fromarray(image).save(image_path)

    print(f"dataset has been downloaded and saved to {train_dir}")

if __name__ == '__main__':
    download_and_organize_cifar10()