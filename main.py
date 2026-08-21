from pathlib import Path

from actions.llm import LLM
from mcts_rag.pipeline import MCTSRAG
from retrieval import RetrievalPipeline
from retrieval.r2_retrieval import DefaultRetriever


def load_documents(
    path: str,
    chunk_size: int = 500,
) -> list[str]:
    text = Path(path).read_text(
        encoding="utf-8"
    )

    return [
        text[i:i + chunk_size]
        for i in range(
            0,
            len(text),
            chunk_size,
        )
    ]


def main():

    documents = load_documents(
        "documents/doc.txt"
    )

    retrieval_strategy = RetrievalPipeline(
        retriever=DefaultRetriever(
            documents=documents
        ),
    )

    llm = LLM()

    mcts_rag = MCTSRAG(
        num_simulations=200,
        retrieval_strategy=retrieval_strategy,
        llm=llm,
    )

    question = input(
        "Question: "
    )

    result = mcts_rag.run(question)

    print("\n" + "=" * 60)
    print("SEARCH TREE")
    print("=" * 60)

    print(result.tree)

    print("\n" + "=" * 60)
    print("CANDIDATE ANSWERS")
    print("=" * 60)

    for i, candidate in enumerate(
        result.candidates
    ):
        print(
            f"\n[{i + 1}] "
            f"reward={candidate.reward:.4f}"
        )

        print(candidate.answer)

    print("\n" + "=" * 60)
    print("FINAL ANSWER")
    print("=" * 60)

    print(result.final_answer)


if __name__ == "__main__":
    main()  