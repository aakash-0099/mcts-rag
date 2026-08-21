from pathlib import Path
import sys


# Make project root importable when this file is run directly.
PROJECT_ROOT = Path(__file__).resolve().parents[3]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from retrieval.pipeline import RetrievalPipeline
from mcts_rag.pipeline import MCTSRAG
sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1])
)
from cwq_retriever import CWQRetriever


CORPUS_FILE = (
    PROJECT_ROOT
    / "evaluation"
    / "complexwebqa"
    / "data"
    / "corpus.jsonl"
)


def main():

    print("=" * 70)
    print("MCTS-RAG + CWQ END-TO-END INTEGRATION TEST")
    print("=" * 70)

    print()
    print(f"Corpus: {CORPUS_FILE}")

    if not CORPUS_FILE.exists():
        raise FileNotFoundError(
            f"CWQ corpus not found: {CORPUS_FILE}"
        )

    # ---------------------------------------------------------
    # R2 — Real CWQ retriever
    # ---------------------------------------------------------

    cwq_retriever = CWQRetriever(
        corpus_path=CORPUS_FILE,
        top_k=5,
    )

    # ---------------------------------------------------------
    # R1 → R2 → R3 → R4
    # ---------------------------------------------------------

    retrieval_pipeline = RetrievalPipeline(
        retriever=cwq_retriever,
    )

    # ---------------------------------------------------------
    # MCTS-RAG
    # ---------------------------------------------------------

    rag = MCTSRAG(
        num_simulations=10,
        retrieval_strategy=retrieval_pipeline,
    )

    question = "Anne Frank school"

    print()
    print("Question:")
    print(question)

    print()
    print("Running MCTS-RAG...")

    result = rag.run(question)

    # ---------------------------------------------------------
    # Results
    # ---------------------------------------------------------

    print()
    print("=" * 70)
    print("RESULT")
    print("=" * 70)

    print()
    print("Final answer:")
    print(result.final_answer)

    print()
    print(f"Candidates: {len(result.candidates)}")

    print()
    print("Best candidate:")

    if result.best_candidate is not None:
        print(result.best_candidate)

    else:
        print("None")

    print()
    print("Best trajectory:")

    for index, item in enumerate(
        result.best_trajectory,
        start=1,
    ):
        action, state = item

        action_name = (
            "ROOT"
            if action is None
            else action.value
        )

        print()
        print(f"Step {index}")
        print(f"Action: {action_name}")
        print(f"Content: {str(state.content)[:500]}")

    print()
    print("MCTS tree:")
    print(result.tree)

    print()
    print("=" * 70)
    print("MCTS-RAG + CWQ TEST PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()