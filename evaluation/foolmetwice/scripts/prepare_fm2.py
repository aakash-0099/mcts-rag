import json
from pathlib import Path


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

# prepare_fm2.py is located at:
# D:\MCTS RAG\evaluation\foolmetwice\scripts\prepare_fm2.py
#
# parents[3] = D:\MCTS RAG
#
PROJECT_ROOT = Path(__file__).resolve().parents[3]

FM2_DATASET = (
    PROJECT_ROOT
    / "fool-me-twice-main"
    / "dataset"
    / "dev.jsonl"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "evaluation"
    / "foolmetwice"
    / "processed"
)
OUTPUT_DATASET = OUTPUT_DIR / "dev_100.jsonl"
OUTPUT_PAGES = OUTPUT_DIR / "dev_100_pages.txt"

NUM_EXAMPLES = 100


# ---------------------------------------------------------
# Load FM2 examples
# ---------------------------------------------------------

def load_examples(path: Path):
    examples = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            examples.append(json.loads(line))

    return examples


# ---------------------------------------------------------
# Process examples
# ---------------------------------------------------------

def process_example(example: dict) -> dict:
    return {
        "id": example["id"],
        "claim": example["text"],
        "label": example["label"],
        "category": example["category"],
        "wikipedia_page": example["wikipedia_page"],
        "gold_evidence": example["gold_evidence"],
    }


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    if not FM2_DATASET.exists():
        raise FileNotFoundError(
            f"FM2 dataset not found:\n{FM2_DATASET}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    examples = load_examples(FM2_DATASET)

    print(f"Loaded {len(examples)} FM2 dev examples.")

    selected = examples[:NUM_EXAMPLES]

    processed = [
        process_example(example)
        for example in selected
    ]

    # -----------------------------------------------------
    # Save processed JSONL
    # -----------------------------------------------------

    with OUTPUT_DATASET.open("w", encoding="utf-8") as f:
        for example in processed:
            f.write(
                json.dumps(
                    example,
                    ensure_ascii=False
                )
                + "\n"
            )

    # -----------------------------------------------------
    # Extract unique Wikipedia pages
    # -----------------------------------------------------

    pages = sorted(
        {
            example["wikipedia_page"]
            for example in processed
        }
    )

    with OUTPUT_PAGES.open("w", encoding="utf-8") as f:
        for page in pages:
            f.write(page + "\n")

    # -----------------------------------------------------
    # Summary
    # -----------------------------------------------------

    print()
    print("FM2 preparation complete.")
    print("--------------------------------")
    print(f"Examples selected : {len(processed)}")
    print(f"Unique pages      : {len(pages)}")
    print()
    print(f"Dataset           : {OUTPUT_DATASET}")
    print(f"Pages             : {OUTPUT_PAGES}")
    print()
    print("Example pages:")
    for page in pages[:10]:
        print(f"  - {page}")


if __name__ == "__main__":
    main()