from PIL import Image
from torchvision import transforms, datasets
import torch
import random
import os


def build_balanced_augmented_tensor_dataset(image_folder_path, image_size=224, grayscale=False, normalize_mode="minmax"):
    """
    Carica immagini da una cartella organizzata per classi (ImageFolder-style),
    bilancia le classi mediante data augmentation statica (in memoria), e restituisce:
    - tensore immagini [N, C, H, W]
    - tensore etichette [N]
    - nomi delle classi

    Parametri:
    - image_folder_path: percorso alle cartelle con sottocartelle per classe
    - image_size: dimensione di resize quadrata (default 224)
    - grayscale: se True, carica immagini in scala di grigi
    - normalize_mode: "minmax" (default), "standard", "none"
    """
    base_transform = transforms.Resize((image_size, image_size))
    to_tensor = transforms.ToTensor()

    augment_transforms = [
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.RandomVerticalFlip(p=1.0),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
    ]

    raw_dataset = datasets.ImageFolder(image_folder_path, transform=None)
    class_to_paths = {}
    for path, label in raw_dataset.samples:
        class_to_paths.setdefault(label, []).append(path)

    max_class_count = max(len(paths) for paths in class_to_paths.values())
    augmented_images, augmented_labels = [], []

    for label, paths in class_to_paths.items():
        for p in paths:
            img = Image.open(p).convert('L' if grayscale else 'RGB')
            tensor_img = to_tensor(base_transform(img))
            augmented_images.append(tensor_img)
            augmented_labels.append(label)

        to_add = max_class_count - len(paths)
        for _ in range(to_add):
            p = random.choice(paths)
            img = Image.open(p).convert('L' if grayscale else 'RGB')
            aug = transforms.Compose([base_transform,
                                      random.choice(augment_transforms),
                                      to_tensor])
            augmented_images.append(aug(img))
            augmented_labels.append(label)

    images_tensor = torch.stack(augmented_images)

    if normalize_mode == "minmax":
        min_val = images_tensor.min()
        max_val = images_tensor.max()
        images_tensor = (images_tensor - min_val) / (max_val - min_val)
        images_tensor = torch.clamp(images_tensor, 0, 1)
    elif normalize_mode == "standard":
        if grayscale:
            mean = torch.tensor([0.5])
            std = torch.tensor([0.5])
        else:
            mean = torch.tensor([0.485, 0.456, 0.406])
            std = torch.tensor([0.229, 0.224, 0.225])
        mean = mean[:, None, None]
        std = std[:, None, None]
        images_tensor = (images_tensor - mean) / std

    return images_tensor, torch.tensor(augmented_labels), raw_dataset.classes
