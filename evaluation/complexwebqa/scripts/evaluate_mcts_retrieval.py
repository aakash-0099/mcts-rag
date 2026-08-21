import sys
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from retrieval.pipeline import RetrievalPipeline
from mcts_rag.pipeline import MCTSRAG

sys.path.insert(
    0,
    str(PROJECT_ROOT / "evaluation" / "complexwebqa" / "scripts")
)

sys.path.insert( 0, str(Path(__file__).resolve().parents[1]) ) 
from cwq_retriever import CWQRetriever


DATA_FILE = (
    PROJECT_ROOT
    / "evaluation"
    / "complexwebqa"
    / "data"
    / "test_100.jsonl"
)

CORPUS_FILE = (
    PROJECT_ROOT
    / "evaluation"
    / "complexwebqa"
    / "data"
    / "corpus.jsonl"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "evaluation"
    / "complexwebqa"
    / "data"
)


BATCH_SIZE = 10
NUM_SIMULATIONS = 10


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


def collect_retrieval_metadata(root):
    """
    Walk the complete MCTS tree and collect every
    question_id retrieved by A4/A5.
    """

    question_ids = []
    doc_ids = []
    retrieval_actions = []

    def visit(node):
        metadata = node.state.metadata

        if metadata.get("retrieval"):
            retrieval_actions.append(
                metadata.get("retrieval_action")
            )

            question_ids.extend(
                metadata.get(
                    "retrieved_question_ids",
                    [],
                )
            )

            doc_ids.extend(
                metadata.get(
                    "retrieved_doc_ids",
                    [],
                )
            )

        for child in node.children:
            visit(child)

    visit(root)

    return {
        "question_ids": list(
            dict.fromkeys(question_ids)
        ),
        "doc_ids": list(
            dict.fromkeys(doc_ids)
        ),
        "retrieval_actions": retrieval_actions,
    }


def evaluate_question(item, rag):

    question_id = item["id"]
    question = item["question"]

    if not isinstance(question, str):
        raise TypeError(
            f"Expected question to be str, "
            f"got {type(question).__name__}: {question!r}"
        )

    result = rag.run(question)

    retrieval = collect_retrieval_metadata(
        result.root
    )

    retrieved_question_ids = retrieval[
        "question_ids"
    ]

    hit = question_id in retrieved_question_ids

    return {
        "id": question_id,
        "question": question,
        "hit": hit,
        "retrieved_question_ids": retrieved_question_ids,
        "retrieved_doc_ids": retrieval["doc_ids"],
        "retrieval_actions": retrieval[
            "retrieval_actions"
        ],
        "final_answer": result.final_answer,
        "candidate_count": len(
            result.candidates
        ),
    }


def run_batch(
    questions,
    batch_number,
    rag,
):
    """
    Run exactly one batch of questions.
    """

    start = (
        batch_number - 1
    ) * BATCH_SIZE

    end = min(
        start + BATCH_SIZE,
        len(questions),
    )

    batch = questions[start:end]

    if not batch:
        raise ValueError(
            f"Batch {batch_number} does not exist."
        )

    print()
    print("=" * 70)
    print(
        f"BATCH {batch_number}"
        f" | QUESTIONS {start + 1}-{end}"
    )
    print("=" * 70)

    details = []
    hits = 0

    for local_index, item in enumerate(
        batch,
        start=1,
    ):
        global_index = start + local_index

        print()
        print(
            f"[{global_index}/{len(questions)}] "
            f"{item['question']}"
        )

        result = evaluate_question(
            item,
            rag,
        )

        details.append(result)

        if result["hit"]:
            hits += 1

        print(
            f"Retrieved correct question pool: "
            f"{'YES' if result['hit'] else 'NO'}"
        )

        print(
            f"Candidates: "
            f"{result['candidate_count']}"
        )

        print(
            f"Final answer: "
            f"{result['final_answer']}"
        )

    recall = (
        hits / len(batch)
        if batch
        else 0.0
    )

    output = {
        "batch": batch_number,
        "start_question": start + 1,
        "end_question": end,
        "num_questions": len(batch),
        "num_simulations": NUM_SIMULATIONS,
        "hits": hits,
        "retrieval_coverage": recall,
        "details": details,
    }

    output_file = (
        OUTPUT_DIR
        / f"mcts_retrieval_batch_{batch_number}.json"
    )

    with output_file.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            output,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print("=" * 70)
    print(
        f"BATCH {batch_number} COMPLETE"
    )
    print("=" * 70)

    print(
        f"Questions: {len(batch)}"
    )

    print(
        f"Hits: {hits}/{len(batch)}"
    )

    print(
        f"Retrieval coverage: {recall:.4f}"
    )

    print(
        f"Saved: {output_file}"
    )

    return output


def main():

    print("=" * 70)
    print("MCTS-RAG CWQ BATCHED RETRIEVAL EVALUATION")
    print("=" * 70)

    questions = load_questions()

    print(
        f"Loaded {len(questions)} questions."
    )

    print(
        f"Batch size: {BATCH_SIZE}"
    )

    print(
        f"MCTS simulations: {NUM_SIMULATIONS}"
    )

    # Ask which batch to run.
    print()
    print(
        "Available batches:"
    )

    total_batches = (
        len(questions) + BATCH_SIZE - 1
    ) // BATCH_SIZE

    for batch_number in range(
        1,
        total_batches + 1,
    ):
        start = (
            (batch_number - 1)
            * BATCH_SIZE
            + 1
        )

        end = min(
            batch_number * BATCH_SIZE,
            len(questions),
        )

        print(
            f"  Batch {batch_number}: "
            f"questions {start}-{end}"
        )

    batch_number = int(
        input(
            "\nEnter batch number to run: "
        )
    )

    if not 1 <= batch_number <= total_batches:
        raise ValueError(
            f"Batch must be between "
            f"1 and {total_batches}."
        )

    print()
    print("Loading CWQ retriever...")

    cwq_retriever = CWQRetriever(
        corpus_path=CORPUS_FILE,
        top_k=10,
    )

    retrieval_pipeline = RetrievalPipeline(
        retriever=cwq_retriever,
    )

    rag = MCTSRAG(
        num_simulations=NUM_SIMULATIONS,
        retrieval_strategy=retrieval_pipeline,
    )

    run_batch(
        questions,
        batch_number,
        rag,
    )


if __name__ == "__main__":
    main()