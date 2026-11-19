"""Preserved and fixed data preparation script matching training normalization."""

from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "Dataset"

IMAGE_SIZE = 224
BATCH_SIZE = 32

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

train_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

eval_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def load_datasets():
    train_dataset = datasets.ImageFolder(DATASET_PATH / "train", transform=train_transforms)
    valid_dataset = datasets.ImageFolder(DATASET_PATH / "valid", transform=eval_transforms)
    test_dataset = datasets.ImageFolder(DATASET_PATH / "test", transform=eval_transforms)

    for name, dataset in [
        ("train", train_dataset),
        ("valid", valid_dataset),
        ("test", test_dataset),
    ]:
        if dataset.class_to_idx != train_dataset.class_to_idx:
            raise ValueError(f"Class mapping mismatch in {name} dataset.")

    return train_dataset, valid_dataset, test_dataset


def main():
    print("Loading datasets...")
    train_dataset, valid_dataset, test_dataset = load_datasets()

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print("\nClass mapping:", train_dataset.class_to_idx)
    print("Training images:", len(train_dataset))
    print("Validation images:", len(valid_dataset))
    print("Test images:", len(test_dataset))
    print("Training batches:", len(train_loader))
    print("Validation batches:", len(valid_loader))
    print("Test batches:", len(test_loader))


if __name__ == "__main__":
    main()
