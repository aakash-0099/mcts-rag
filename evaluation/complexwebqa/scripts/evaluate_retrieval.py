import sys
import json
from pathlib import Path

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1])
)

from cwq_retriever import CWQRetriever


DATA_FILE = Path(
    "evaluation/complexwebqa/data/test_100.jsonl"
)

CORPUS_FILE = Path(
    "evaluation/complexwebqa/data/corpus.jsonl"
)


def load_questions():
    questions = []

    with DATA_FILE.open(
        "r",
        encoding="utf-8",
    ) as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))

    return questions


def evaluate(questions, retriever, k):
    hits = 0
    total = len(questions)

    details = []

    for i, item in enumerate(
        questions,
        start=1,
    ):
        question_id = item["id"]
        question = item["question"]

        results = retriever.retrieve(
            question
        )

        top_results = results[:k]

        retrieved_ids = [
            result["document"]["question_id"]
            for result in top_results
        ]

        hit = question_id in retrieved_ids

        if hit:
            hits += 1

        details.append(
            {
                "id": question_id,
                "question": question,
                "hit": hit,
                "retrieved_question_ids": retrieved_ids,
            }
        )

        if i % 10 == 0:
            print(
                f"Processed {i}/{total}"
            )

    recall = (
        hits / total
        if total
        else 0.0
    )

    return recall, details


def main():

    print("Loading questions...")

    questions = load_questions()

    print(
        f"Loaded {len(questions)} questions."
    )

    retriever = CWQRetriever(
        corpus_path=CORPUS_FILE,
        top_k=10,
    )

    print()
    print("=" * 60)
    print("CWQ LEXICAL RETRIEVAL BASELINE")
    print("=" * 60)

    all_results = {}

    for k in [1, 5, 10]:

        print()
        print(f"Evaluating Recall@{k}...")

        recall, details = evaluate(
            questions,
            retriever,
            k,
        )

        all_results[f"recall@{k}"] = recall

        print(
            f"Recall@{k}: "
            f"{recall:.4f}"
        )

    output_file = Path(
        "evaluation/complexwebqa/data/"
        "retrieval_baseline.json"
    )

    with output_file.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            all_results,
            f,
            indent=2,
        )

    print()
    print("=" * 60)
    print("BASELINE COMPLETE")
    print("=" * 60)

    for metric, value in all_results.items():
        print(
            f"{metric}: {value:.4f}"
        )

    print()
    print(
        f"Saved: {output_file}"
    )


if __name__ == "__main__":
    main()