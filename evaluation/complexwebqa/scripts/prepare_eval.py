import json
from pathlib import Path


INPUT_FILE = Path(
    "evaluation/complexwebqa/data/ComplexWebQuestions_dev.json"
)

OUTPUT_FILE = Path(
    "evaluation/complexwebqa/data/dev_100.jsonl"
)


def normalize_answer(answer):
    if isinstance(answer, dict):
        return answer.get("answer", "")

    return str(answer)


def main():
    with INPUT_FILE.open("r", encoding="utf-8") as f:
        data = json.load(f)

    selected = data[:100]

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        for item in selected:

            answers = [
                normalize_answer(a)
                for a in item.get("answers", [])
            ]

            record = {
                "id": item["ID"],
                "question": item["question"],
                "answers": answers,
                "compositionality_type": item.get(
                    "compositionality_type"
                ),
                "sparql": item.get("sparql"),
            }

            f.write(
                json.dumps(record, ensure_ascii=False)
                + "\n"
            )

    print(f"Created {OUTPUT_FILE}")
    print(f"Examples: {len(selected)}")


if __name__ == "__main__":
    main()