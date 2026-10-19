
from datasets import load_dataset

dataset = load_dataset("arbml/arabic_100k_reviews")["train"]

print("Dataset features:")
print(dataset.features)

print("\nLabel names from dataset metadata:")
label_feature = dataset.features["label"]

if hasattr(label_feature, "names") and label_feature.names:
    print(label_feature.names)
else:
    print("No label names in metadata.")

print("\nReview samples by label:")
for label in sorted(set(dataset["label"])):
    print(f"\nLabel {label}:")
    shown = 0

    for row in dataset:
        if row["label"] == label:
            print(row["text"][:250])
            shown += 1
            if shown == 5:
                break
