
from collections import Counter

from datasets import load_dataset


def audit_general_reviews():
    print("\n" + "=" * 65)
    print("GENERAL ARABIC REVIEWS")

    dataset = load_dataset("arbml/arabic_100k_reviews")["train"]

    print("Rows:", len(dataset))
    print("Label counts:", Counter(dataset["label"]))

    # Inspect examples to help infer the sentiment mapping.
    for label in sorted(set(dataset["label"])):
        print(f"\nLabel {label} examples:")
        shown = 0
        for row in dataset:
            if row["label"] == label:
                print("-", row["text"][:300])
                shown += 1
                if shown == 10:
                    break


def audit_delivery_feedback():
    print("\n" + "=" * 65)
    print("DELIVERY FEEDBACK")

    dataset = load_dataset("fr3on/arabic-feedback-corpus")["train"]

    print("Rows:", len(dataset))
    print("Label counts:", Counter(dataset["label"]))
    print("Conflict counts:", Counter(dataset["is_conflict"]))
    print("Score counts:", Counter(dataset["score"]))
    print("Domain counts:", Counter(dataset["domain"]))

    conflicts = dataset.filter(lambda row: row["is_conflict"])
    print("\nNumber of flagged conflicts:", len(conflicts))

    for row in conflicts.select(range(min(10, len(conflicts)))):
        print({
            "text": row["text"],
            "label": row["label"],
            "score": row["score"],
            "is_conflict": row["is_conflict"],
        })

    print("\nExamples labeled positive:")
    positive = dataset.filter(lambda row: row["label"] == "positive")
    for row in positive.select(range(min(15, len(positive)))):
        print({
            "text": row["text"],
            "label": row["label"],
            "score": row["score"],
            "is_conflict": row["is_conflict"],
        })


if __name__ == "__main__":
    audit_general_reviews()
    audit_delivery_feedback()
