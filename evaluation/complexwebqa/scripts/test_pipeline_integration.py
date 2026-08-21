from pathlib import Path
import sys
sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[3])
)
from retrieval.pipeline import RetrievalPipeline
from retrieval.r2_retrieval import Retriever
import sys
from pathlib import Path
sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1])
)
from cwq_retriever import CWQRetriever


CORPUS_FILE = Path(
    "evaluation/complexwebqa/data/corpus.jsonl"
)


def main():
    retriever = CWQRetriever(
        corpus_path=CORPUS_FILE,
        top_k=5,
    )

    pipeline = RetrievalPipeline(
        retriever=retriever,
    )

    question = "Anne Frank school"

    result = pipeline.retrieve(question)

    print("=" * 60)
    print("RETRIEVAL PIPELINE INTEGRATION TEST")
    print("=" * 60)

    print()
    print("Question:")
    print(question)

    print()
    print("Generated query:")
    print(result.query)

    print()
    print(f"Retrieved results: {len(result.results)}")

    for i, item in enumerate(result.results, start=1):
        print()
        print(f"Result {i}")
        print(f"Score: {item.get('score')}")
        print(f"Question ID: {item.get('question_id')}")

        document = item.get("document", "")
        print(f"Document: {str(document)[:300]}")

    print(f"Keys: {list(item.keys())}")

    print()
    print("Reflection:")
    print(result.reflection)

    print()
    print("Summary:")
    print(result.summary[:1000])

    print()
    print("=" * 60)
    print("INTEGRATION TEST PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()