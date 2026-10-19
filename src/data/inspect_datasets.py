
from datasets import load_dataset

DATASETS = [
    "arbml/arabic_100k_reviews",
    "fr3on/arabic-feedback-corpus",
]

for dataset_name in DATASETS:
    print(f"\n{'=' * 60}")
    print(f"Dataset: {dataset_name}")

    try:
        dataset = load_dataset(dataset_name)
        print(dataset)

        for split_name, split in dataset.items():
            print(f"\nSplit: {split_name}")
            print(f"Rows: {len(split)}")
            print(f"Columns: {split.column_names}")
            print("Examples:")

            for example in split.select(range(min(3, len(split)))):
                print(example)

    except Exception as exc:
        print(f"Could not load dataset: {exc}")
