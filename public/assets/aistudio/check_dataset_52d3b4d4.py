from pathlib import Path

print("Dataset checker is running!")

DATASET_PATH = Path(
    r"C:\Users\shahid raza\Downloads\Photo\OneDrive\Desktop\deep_image_detection\Dataset"
)

def inspect_dataset():
    if not DATASET_PATH.exists():
        print(f"ERROR: Dataset folder not found:\n{DATASET_PATH}")
        return

    print(f"\nDataset found: {DATASET_PATH}")

    for split in ("train", "valid", "test"):
        print(f"\n--- {split.upper()} ---")

        for label in ("real", "fake"):
            folder = DATASET_PATH / split / label

            if not folder.exists():
                print(f"{label}: FOLDER NOT FOUND")
                continue

            extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
            count = sum(
                1 for file in folder.rglob("*")
                if file.is_file() and file.suffix.lower() in extensions
            )

            print(f"{label}: {count} images")


if __name__ == "__main__":
    inspect_dataset()