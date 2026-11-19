from pathlib import Path

import torch
import torch.nn as nn
from torchvision import datasets, models, transforms


# Project paths
ROOT = Path(__file__).resolve().parent.parent
TEST_PATH = ROOT / "Dataset" / "test"
MODEL_PATH = ROOT / "models" / "best_model.pth"

IMAGE_SIZE = 224
BATCH_SIZE = 16
DEVICE = torch.device("cpu")


def main():
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    print("Loading test dataset...")

    dataset = datasets.ImageFolder(
        TEST_PATH,
        transform=transform
    )

    print("Class mapping:", dataset.class_to_idx)
    print("Total images:", len(dataset))

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    print("\nLoading saved model...")

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True
    )

    model = models.efficientnet_b0(weights=None)

    input_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(input_features, 2)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    actual_counts = [0, 0]
    predicted_counts = [0, 0]
    correct_counts = [0, 0]

    with torch.no_grad():
        for batch_number, (images, labels) in enumerate(loader, start=1):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            predictions = outputs.argmax(dim=1)

            for class_id in range(2):
                actual_counts[class_id] += (
                    labels == class_id
                ).sum().item()

                predicted_counts[class_id] += (
                    predictions == class_id
                ).sum().item()

                correct_counts[class_id] += (
                    (predictions == class_id) &
                    (labels == class_id)
                ).sum().item()

            if batch_number % 100 == 0:
                print(f"Processed {batch_number} batches...")

    print("\nPrediction Summary")
    print("=" * 35)

    for class_name, class_id in dataset.class_to_idx.items():
        print(f"\nClass: {class_name}")
        print("Actual images:", actual_counts[class_id])
        print("Predicted as this class:", predicted_counts[class_id])
        print("Correct predictions:", correct_counts[class_id])

    total_correct = sum(correct_counts)
    total_images = sum(actual_counts)

    print(f"\nOverall accuracy: {total_correct / total_images:.2%}")


if __name__ == "__main__":
    main()