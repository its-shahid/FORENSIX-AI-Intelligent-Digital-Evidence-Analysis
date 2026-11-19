from pathlib import Path
from PIL import Image

DATASET_PATH = Path(
    r"C:\Users\shahid raza\Downloads\Photo\OneDrive\Desktop\deep_image_detection\Dataset"
)

SPLITS = ("train", "valid", "test")
CLASSES = ("real", "fake")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
SAMPLE_SIZE = 100


def main():
    print("FAST DATASET VALIDATION")
    print("=" * 45)

    for split in SPLITS:
        for label in CLASSES:
            folder = DATASET_PATH / split / label

            print(f"\nChecking {split}/{label}...")

            if not folder.exists():
                print("ERROR: Folder not found")
                continue

            images = sorted(
                path for path in folder.iterdir()
                if path.is_file()
                and path.suffix.lower() in IMAGE_EXTENSIONS
            )

            print(f"Total images: {len(images)}")

            sample = images[:SAMPLE_SIZE]
            corrupted = []

            for image_path in sample:
                try:
                    with Image.open(image_path) as image:
                        image.verify()
                except Exception:
                    corrupted.append(image_path.name)

            print(f"Images checked: {len(sample)}")
            print(f"Corrupted in sample: {len(corrupted)}")

            if corrupted:
                print("Examples:", corrupted[:5])

    print("\nSample validation complete.")


if __name__ == "__main__":
    main()