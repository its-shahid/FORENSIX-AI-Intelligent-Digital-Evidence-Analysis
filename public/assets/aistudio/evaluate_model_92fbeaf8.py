

from pathlib import Path
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

import matplotlib.pyplot as plt
import seaborn as sns


# Project paths
ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = ROOT / "Dataset"
MODEL_PATH = ROOT / "models" / "best_model.pth"
RESULTS_DIR = ROOT / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Settings
IMAGE_SIZE = 224
BATCH_SIZE = 16
SAMPLE_SIZE = 1000
DEVICE = torch.device("cpu")

CLASS_NAMES = ["fake", "real"]


def main():
    print("Loading test dataset...")

    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    # Load the complete test dataset
    full_test_data = datasets.ImageFolder(
        DATASET_PATH / "test",
        transform=transform,
    )

    if full_test_data.class_to_idx != {"fake": 0, "real": 1}:
        raise ValueError(
            f"Unexpected class mapping: {full_test_data.class_to_idx}"
        )

    # Select up to 500 images from each class
    # Select random, balanced images from each class
fake_indices = [
    i for i, (_, label) in enumerate(full_test_data.samples)
    if label == 0
]

real_indices = [
    i for i, (_, label) in enumerate(full_test_data.samples)
    if label == 1
]

per_class = SAMPLE_SIZE // 2

random.seed(42)

if len(fake_indices) < per_class or len(real_indices) < per_class:
    raise ValueError("Not enough images in one of the classes.")

selected_fake = random.sample(fake_indices, per_class)
selected_real = random.sample(real_indices, per_class)

selected_indices = selected_fake + selected_real
random.shuffle(selected_indices)

if not selected_indices:
    raise ValueError("No test images were found.")

    test_data = Subset(full_test_data, selected_indices)

    test_loader = DataLoader(
        test_data,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    print("Total test images available:", len(full_test_data))
    print("Images used for this evaluation:", len(test_data))
    print("Fake images selected:", min(per_class, len(fake_indices)))
    print("Real images selected:", min(per_class, len(real_indices)))
    print("Class mapping:", full_test_data.class_to_idx)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}"
        )

    print("\nLoading trained model...")

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True,
    )

    if checkpoint["class_to_idx"] != full_test_data.class_to_idx:
        raise ValueError("Model and test dataset class mappings differ.")

    model = models.efficientnet_b0(weights=None)

    input_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(input_features, 2)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    all_labels = []
    all_predictions = []

    print("\nEvaluating test images...")

    with torch.no_grad():
        for batch_number, (images, labels) in enumerate(test_loader, start=1):
            images = images.to(DEVICE)

            outputs = model(images)
            predictions = outputs.argmax(dim=1)

            all_labels.extend(labels.tolist())
            all_predictions.extend(predictions.cpu().tolist())

            if batch_number % 10 == 0:
                print(
                    f"Processed {len(all_labels)}/{len(test_data)} images"
                )

    # Calculate evaluation metrics
    accuracy = accuracy_score(all_labels, all_predictions)

    print("\nTEST RESULTS")
    print("=" * 40)
    print(f"Accuracy: {accuracy:.2%}")

    print("\nClassification Report:")
    print(
        classification_report(
            all_labels,
            all_predictions,
            labels=[0, 1],
            target_names=CLASS_NAMES,
            zero_division=0,
        )
    )

    matrix = confusion_matrix(
        all_labels,
        all_predictions,
        labels=[0, 1],
    )

    print("Confusion Matrix:")
    print(matrix)
    print("\nRows = actual classes; columns = predicted classes.")
    print("Class order: fake, real")

    # Save confusion matrix image
    plt.figure(figsize=(7, 5))

    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
    )

    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.title("Deepfake Detection Confusion Matrix")
    plt.tight_layout()

    output_path = RESULTS_DIR / "confusion_matrix.png"
    plt.savefig(output_path)
    plt.close()

    print("\nConfusion matrix saved to:", output_path)
    print("Evaluation completed!")


if __name__ == "__main__":
    main()
