import os
from torch.utils.data import DataLoader
from pprint import pprint
import numpy as np


def find(path, name):
    """ 
    Searches for a file or directory with the given name in the specified path.
    Args:
        path (str): The directory path to search in (recursive).
        name (str): The name of the file or directory to search for.
    Returns:
        str: The full path to the file or directory if found, otherwise None.
    """
    for root, dirs, files in os.walk(path):
        if name in files or name in dirs:
            return os.path.join(root, name)


def classes_nums(dataset, pretty_classes=None):
    """
    Prints a summary of the dataset, including class distribution and size.
    Args:
        dataset (torch.utils.data.Dataset): The dataset to summarize.
        pretty_classes (list, optional): A list of class names to display. 
                                          If None, uses dataset.class_to_idx keys.
    """

    if pretty_classes is None:
        # Se non sono specificate classi, usa quelle del dataset
        pretty_classes = list(dataset.class_to_idx.keys())

    # class_to_idx mappa le classi in indici numerici
    idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}
    # idx_to_class mappa gli indici numerici alle classi

    # Ottiene la lista di target come stringhe (verisone brutta)
    targets_str = [idx_to_class[idx] for idx in dataset.targets]

    classes, values = np.unique_counts(targets_str)
    

    return dict(zip(pretty_classes, values.tolist()))

    
    