# dataset_utils.py
import os
import json
import random
from typing import List, Dict, Tuple, Any

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image


IGNORE_INDEX = -100


class MultiTaskDataset(Dataset):
    """
    Dataset multi-tâches, basé sur :
      - data_json : annotations/images
      - classes_json : tâches -> liste de classes
    Option search_folder : dossier où se trouvent directement les images
    (on remplace alors les paths présents dans data_json).
    """
    def __init__(self, data_json: str, classes_json: str,
                 transform=None, search_folder: str | None = None):
        with open(data_json, 'r') as f:
            self.data = json.load(f)
        with open(classes_json, 'r') as f:
            self.classes = json.load(f)

        self.transform = transform
        self.search_folder = search_folder
        self.samples: List[Tuple[str, Dict[str, int | None]]] = []
        self.class_to_idx: Dict[str, Dict[str, int]] = {}
        self.task_classes: Dict[str, List[str]] = {}

        # mapping classes -> indices (en lowercase)
        for task, class_list in self.classes.items():
            self.task_classes[task] = class_list
            self.class_to_idx[task] = {cls_name.lower(): idx
                                       for idx, cls_name in enumerate(class_list)}

        # construction des échantillons
        for folder, images in self.data.items():
            for img_name, img_info in images.items():
                if self.search_folder:
                    image_identifier = os.path.join(
                        self.search_folder,
                        os.path.basename(img_info['image_path'])
                    )
                else:
                    image_identifier = img_info['image_path']

                labels: Dict[str, int | None] = {}
                for task in self.classes.keys():
                    if task in img_info:
                        label = img_info[task].lower()
                        if label in self.class_to_idx[task]:
                            labels[task] = self.class_to_idx[task][label]
                        else:
                            print(f"[WARN] label '{label}' for task "
                                  f"'{task}' not found in class_to_idx")
                            labels[task] = None
                    else:
                        labels[task] = None

                self.samples.append((image_identifier, labels))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, labels = self.samples[idx]
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Chemin de l'image '{img_path}' introuvable.")
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, labels


def multitask_collate(batch, task_names: List[str], ignore_index: int = IGNORE_INDEX):
    """
    Collate pour dataset multi-tâches.
    batch : liste de (image_tensor, dict_task->label_ou_None)
    """
    imgs, lbls = zip(*batch)
    imgs = torch.stack(imgs, dim=0)
    out = {}
    for t in task_names:
        vec = [ignore_index if d.get(t, None) is None else int(d[t]) for d in lbls]
        out[t] = torch.tensor(vec, dtype=torch.long)
    return imgs, out


def create_dataloader(dataset: Dataset,
                      task_names: List[str],
                      batch_size: int,
                      num_workers: int = 4,
                      shuffle: bool = False,
                      ignore_index: int = IGNORE_INDEX) -> DataLoader:
    """
    Crée un DataLoader avec le collate multi-tâches.
    """
    def _collate_fn(batch):
        return multitask_collate(batch, task_names, ignore_index)

    return DataLoader(dataset,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      num_workers=num_workers,
                      collate_fn=_collate_fn)


def build_default_transform(img_size: int = 224) -> transforms.Compose:
    """
    Transform standard ImageNet (Resize 256, CenterCrop img_size, Normalisation).
    """
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])


def subsample_dataset(dataset: Dataset,
                      num_samples: int | None) -> Dataset:
    """
    Retourne un Subset si num_samples est précisé, sinon dataset.
    """
    if num_samples is None or num_samples <= 0:
        return dataset
    idxs = list(range(len(dataset)))
    random.shuffle(idxs)
    idxs = idxs[:num_samples]
    return Subset(dataset, idxs)
