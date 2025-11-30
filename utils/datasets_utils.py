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

def _get_loader_paths(loader):
    """
    Renvoie la liste des chemins d’images selon l’ordre d’itération du DataLoader.
    Compatible avec:
      - Dataset ayant .paths_list (ancienne version)
      - Dataset ayant .samples (liste de tuples (path, labels) ou dicts)
      - Subset(Dataset, indices)
    """
    ds = loader.dataset
    # Récupération du dataset de base et des indices
    if isinstance(ds, Subset):
        base = ds.dataset
        idxs = [int(i) for i in ds.indices]
    else:
        base = ds
        idxs = list(range(len(base)))

    # 1) Ancienne variante: attribut paths_list
    if hasattr(base, 'paths_list'):
        return [base.paths_list[i] for i in idxs]

    # 2) Variante actuelle: attribut samples
    if hasattr(base, 'samples'):
        # samples peut être: [(path, labels), ...]  OU  [ {'image_path':..., ...}, ... ]
        if len(base.samples) == 0:
            return []
        s0 = base.samples[0]
        if isinstance(s0, (list, tuple)):
            # (path, labels)
            return [base.samples[i][0] for i in idxs]
        elif isinstance(s0, dict):
            # {'image_path': ...} ou {'path': ...}
            paths = []
            for i in idxs:
                rec = base.samples[i]
                p = rec.get('image_path', rec.get('path'))
                if p is None:
                    raise KeyError("Impossible de trouver 'image_path' ou 'path' dans sample dict.")
                paths.append(p)
            return paths

    # 3) Fallback si le dataset expose une méthode
    if hasattr(base, 'get_path') and callable(getattr(base, 'get_path')):
        return [base.get_path(i) for i in idxs]

    raise AttributeError("Dataset inconnu: ni 'paths_list', ni 'samples', ni 'get_path(i)'.")

def map_folder_to_class(folder_name, class_list):
    """
    Essaie de faire correspondre le nom du dossier (ground truth)
    à l'une des classes en vérifiant si le nom du dossier est contenu
    dans le nom de la classe (sans tenir compte de la casse).
    """
    folder_lower = folder_name.lower()
    for cls in class_list:
        if folder_lower in cls.lower():
            return cls
    return None



VALID_EXTS = {'.jpg', '.jpeg', '.png', '.bmp'}
def collect_image_paths(folder):
    """
    Retourne la liste de tous les fichiers images dans `folder` et ses sous-dossiers.
    """
    paths = []
    for root, _, files in os.walk(folder):
        for fn in files:
            ext = os.path.splitext(fn)[1].lower()
            if ext in VALID_EXTS:
                paths.append(os.path.join(root, fn))
    return paths

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
