
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, ConcatDataset, Dataset
import sys
import os
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
import torch.optim as optim
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score
import modules.strings as strings
import numpy as np
from pprint import pprint
import modules.custom_transforms as custom
import matplotlib.pyplot as plt 

def augment_dataset(dataset_path, times=2)-> list[ImageFolder]:
    default = ImageFolder(root=dataset_path, transform=custom.base_transform)

    datasets = [default]

    for _ in range(times):
        aug_dataset = ImageFolder(root=dataset_path, transform=custom.augment_transform)
        datasets.append(aug_dataset)


    # print(datasets[0][0][0][0][100,:])
    # print(datasets[1][0][0][0][100,:])
    # print(datasets[2][0][0][0][100,:])
    # Stanmpa cose diverse per i tre dataset
    # Gli indici sono: - Il primo per quale dataset
    #                  - Il secondo per Immagine o label
    #                  - Il terzo per l'indice della immagine nel dataset
    #                  - Il quarto per il canale (RGB)
    #                  - Gli ultimi selezionano la riga e la colonna dell'immagine nel canale
    # Nota: i target sono uguali per ogni dataset, cambiano solo gli input

    return datasets

def show_batch(dataset, N=8):
    loader =  DataLoader(dataset, batch_size=N, shuffle=True, num_workers=0)

    classes = list(dataset.class_to_idx.keys())

    images, labels = next(iter(loader))
    plt.figure(figsize=(12, 8))
    for i in range(N):
        ax = plt.subplot(2, (N+1)//2, i+1)
        img = images[i].permute(1, 2, 0).numpy() * 0.5 + 0.5  # denormalize
        plt.imshow(img)
        plt.title(classes[labels[i]])
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def map_datasets(datasets: list[ImageFolder]):

    model = resnet18(weights=ResNet18_Weights.DEFAULT) # trasnfer learning (pre-trained model)

    # congela i layer convoluzionali per fare fine-tuning solo sull'ultimo layer
    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Identity()  # rimuove l'ultimo layer FC
    model.eval()

    results = []
    targets = []

    with torch.no_grad():
        for dataset in datasets:
            loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False, num_workers=0)
            for inputs, labels in loader:
                outputs = model(inputs)
                results.append(outputs)
                targets.append(labels)

    return torch.cat(results, dim=0), torch.cat(targets, dim=0)