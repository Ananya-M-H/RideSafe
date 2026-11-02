"""Dataset loading and stratified train/val/test split for pothole images."""
from pathlib import Path
from typing import List, Tuple

import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import transforms

CLASSES = ["no_pothole", "pothole"]
IMAGE_SIZE = 128
SEED = 42

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

train_transform = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
)

eval_transform = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
)


def list_samples(data_dir: Path) -> Tuple[List[Path], List[int]]:
    paths, labels = [], []
    for label, cls in enumerate(CLASSES):
        for path in sorted((data_dir / cls).glob("*.jpg")):
            paths.append(path)
            labels.append(label)
    return paths, labels


class PotholeDataset(Dataset):
    def __init__(self, paths: List[Path], labels: List[int], transform):
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        image = Image.open(self.paths[idx]).convert("RGB")
        image = self.transform(image)
        label = self.labels[idx]
        return image, label


def make_splits(data_dir: Path, val_size: float = 0.15, test_size: float = 0.15):
    paths, labels = list_samples(data_dir)

    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        paths,
        labels,
        test_size=val_size + test_size,
        stratify=labels,
        random_state=SEED,
    )
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths,
        temp_labels,
        test_size=test_size / (val_size + test_size),
        stratify=temp_labels,
        random_state=SEED,
    )

    train_ds = PotholeDataset(train_paths, train_labels, train_transform)
    val_ds = PotholeDataset(val_paths, val_labels, eval_transform)
    test_ds = PotholeDataset(test_paths, test_labels, eval_transform)
    return train_ds, val_ds, test_ds


def class_weights(labels: List[int]) -> torch.Tensor:
    counts = torch.bincount(torch.tensor(labels), minlength=len(CLASSES)).float()
    weights = counts.sum() / (len(CLASSES) * counts)
    return weights