import sys
from pathlib import Path

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1])
)
from cwq_retriever import CWQRetriever

CORPUS = (
    "evaluation/complexwebqa/data/corpus.jsonl"
)


def main():

    retriever = CWQRetriever(
        corpus_path=CORPUS,
        top_k=5,
    )

    query = (
        "Anne Frank school"
    )

    results = retriever.retrieve(query)

    print()
    print("=" * 60)
    print("RETRIEVAL RESULTS")
    print("=" * 60)

    for i, result in enumerate(
        results,
        start=1,
    ):

        document = result["document"]

        print()
        print(f"Result {i}")
        print(f"Score: {result['score']}")
        print(f"Title: {document['title']}")
        print(
            f"Question ID: "
            f"{document['question_id']}"
        )
        print(
            f"Text: "
            f"{document['text'][:300]}"
        )


if __name__ == "__main__":
    main()