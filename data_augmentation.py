import os
import random
import torch
from PIL import Image
from torchvision import transforms, datasets


def build_balanced_augmented_tensor_dataset(
    image_folder_path,
    image_size=224,
    grayscale=False,
    normalize_mode="minmax",
    mode="balance",          # 'balance' o 'augment'
    augment_factor=2         # usato solo se mode == 'augment'
):
    """
    Costruisce un dataset tensoriale bilanciato o aumentato a partire da immagini organizzate per classe.

    Args:
        image_folder_path (str): Path alla cartella principale (ImageFolder-style).
        image_size (int): Dimensione a cui ridimensionare le immagini.
        grayscale (bool): Se True, converte le immagini in scala di grigi.
        normalize_mode (str): "minmax", "standard", "zscore".
        mode (str): "balance" per bilanciare classi al massimo, "augment" per moltiplicare i dati.
        augment_factor (int): Fattore di aumento nel caso mode == "augment".

    Returns:
        torch.Tensor: immagini [N, C, H, W]
        torch.Tensor: etichette [N]
        list: nomi delle classi
    """

    assert mode in ['balance', 'augment'], "mode deve essere 'balance' o 'augment'"

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

    if mode == 'balance':
        target_per_class = max(len(paths) for paths in class_to_paths.values())
    else:  # mode == 'augment'
        target_per_class = max(int(len(paths) * augment_factor) for paths in class_to_paths.values())

    augmented_images, augmented_labels = [], []

    for label, paths in class_to_paths.items():
        orig_len = len(paths)
        extra_needed = target_per_class - orig_len
        paths = list(paths)  # sicurezza

        # Aggiunge immagini originali
        for p in paths:
            img = Image.open(p).convert('L' if grayscale else 'RGB')
            tensor_img = to_tensor(base_transform(img))
            augmented_images.append(tensor_img)
            augmented_labels.append(label)

        # Aggiunge immagini aumentate
        for _ in range(extra_needed):
            p = random.choice(paths)
            img = Image.open(p).convert('L' if grayscale else 'RGB')
            aug = transforms.Compose([
                base_transform,
                random.choice(augment_transforms),
                to_tensor
            ])
            augmented_images.append(aug(img))
            augmented_labels.append(label)

    images_tensor = torch.stack(augmented_images)

    # --- Normalizzazione ---
    if normalize_mode == "minmax":
        min_val = images_tensor.min()
        max_val = images_tensor.max()
        images_tensor = (images_tensor - min_val) / (max_val - min_val)
        images_tensor = torch.clamp(images_tensor, 0, 1)

    elif normalize_mode == "standard":
        mean = torch.tensor([0.5]) if grayscale else torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.5]) if grayscale else torch.tensor([0.229, 0.224, 0.225])
        mean = mean[:, None, None]
        std = std[:, None, None]
        images_tensor = (images_tensor - mean) / std

    elif normalize_mode == "zscore":
        mean = images_tensor.mean(dim=(0, 2, 3), keepdim=True)
        std = images_tensor.std(dim=(0, 2, 3), keepdim=True)
        images_tensor = (images_tensor - mean) / (std + 1e-8)

    return images_tensor, torch.tensor(augmented_labels), raw_dataset.classes
