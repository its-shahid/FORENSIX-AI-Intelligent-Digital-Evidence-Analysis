
from pathlib import Path
import copy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms
from torchvision.models import EfficientNet_B0_Weights


# Project paths
ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = ROOT / "Dataset"
MODEL_DIR = ROOT / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Training settings
IMAGE_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 0.0001

TRAIN_LIMIT = None
VALID_LIMIT = None

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Image preprocessing for training
train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

# Image preprocessing for validation
valid_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


import random

def balanced_indices(dataset, samples_per_class):
    """Select random, balanced images from each class."""
    selected = []

    for class_id in range(len(dataset.classes)):
        class_indices = [
            index
            for index, (_, label) in enumerate(dataset.samples)
            if label == class_id
        ]

        if len(class_indices) < samples_per_class:
            raise ValueError(
                f"Class {dataset.classes[class_id]} has only "
                f"{len(class_indices)} images."
            )

        random.shuffle(class_indices)
        selected.extend(random.sample(class_indices, samples_per_class))

    random.shuffle(selected)
    return selected


def load_data():
    """Load balanced training and validation subsets."""
    train_path = DATASET_PATH / "train"
    valid_path = DATASET_PATH / "valid"

    if not train_path.exists() or not valid_path.exists():
        raise FileNotFoundError(
            f"Could not find train or valid folders in {DATASET_PATH}"
        )

    train_data = datasets.ImageFolder(
        train_path,
        transform=train_transform
    )

    valid_data = datasets.ImageFolder(
        valid_path,
        transform=valid_transform
    )

    if train_data.class_to_idx != valid_data.class_to_idx:
        raise ValueError(
            "Training and validation class mappings differ."
        )

    if len(train_data.classes) != 2:
        raise ValueError(
            f"Expected two classes, found {train_data.classes}"
        )

    # Select images from both classes
    if TRAIN_LIMIT is None:
        train_indices = list(range(len(train_data)))
    else:
         train_indices = balanced_indices(
        train_data,
        samples_per_class=TRAIN_LIMIT // 2
    )

    if VALID_LIMIT is None:
        valid_indices = list(range(len(valid_data)))
    else:
        valid_indices = balanced_indices(
        valid_data,
        samples_per_class=VALID_LIMIT // 2
    )

    train_subset = Subset(train_data, train_indices)
    valid_subset = Subset(valid_data, valid_indices)

    train_loader = DataLoader(
        train_subset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    valid_loader = DataLoader(
        valid_subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    print("Class mapping:", train_data.class_to_idx)
    print("Training samples per class:", TRAIN_LIMIT // 2)
    print("Validation samples per class:", VALID_LIMIT // 2)

    return train_loader, valid_loader, train_data.class_to_idx


def build_model():
    """Load EfficientNet-B0 and replace its classifier."""
    print("Loading EfficientNet-B0...")

    model = models.efficientnet_b0(
        weights=EfficientNet_B0_Weights.DEFAULT
    )

    # Freeze pretrained feature-extraction layers
    for parameter in model.features.parameters():
        parameter.requires_grad = True

    # Create a new classifier for fake and real images
    input_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(input_features, 2)

    return model.to(DEVICE)


def evaluate(model, loader, criterion):
    """Calculate validation loss and accuracy."""
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * labels.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

    if total == 0:
        raise ValueError("Validation dataset is empty.")

    return total_loss / total, correct / total


def main():
    print("Device:", DEVICE)
    print("Preparing datasets...")

    train_loader, valid_loader, class_to_idx = load_data()

    print("Training images:", len(train_loader.dataset))
    print("Validation images:", len(valid_loader.dataset))

    model = build_model()

    criterion = nn.CrossEntropyLoss()

    # Only the new classifier is trainable
    optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LEARNING_RATE
     )

    best_accuracy = -1.0

    for epoch in range(EPOCHS):
        model.train()

        total_loss = 0.0
        correct = 0
        total = 0

        print(f"\nEpoch {epoch + 1}/{EPOCHS}")

        for images, labels in train_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

        train_loss = total_loss / total
        train_accuracy = correct / total

        valid_loss, valid_accuracy = evaluate(
            model,
            valid_loader,
            criterion
        )

        print(f"Training loss: {train_loss:.4f}")
        print(f"Training accuracy: {train_accuracy:.2%}")
        print(f"Validation loss: {valid_loss:.4f}")
        print(f"Validation accuracy: {valid_accuracy:.2%}")

        if valid_accuracy > best_accuracy:
            best_accuracy = valid_accuracy

            # Save the best model and its class mapping
            torch.save(
                {
                    "model_state_dict": copy.deepcopy(
                        model.state_dict()
                    ),
                    "class_to_idx": class_to_idx,
                    "image_size": IMAGE_SIZE,
                },
                MODEL_DIR / "best_model.pth"
            )

            print("Best model saved!")

    print("\nTraining test completed!")
    print(f"Best validation accuracy: {best_accuracy:.2%}")
    print("Model location:", MODEL_DIR / "best_model.pth")


if __name__ == "__main__":
    main()

