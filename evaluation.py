from typing import List, Dict


def recall_at_k(
    retrieved_documents: List[Dict],
    relevant_documents: List[int],
) -> float:

    retrieved_indices = {
        doc["index"]
        for doc in retrieved_documents
    }

    relevant_indices = set(
        relevant_documents
    )

    if not relevant_indices:
        return 0.0

    found = (
        retrieved_indices
        & relevant_indices
    )

    return len(found) / len(relevant_indices)


def precision_at_k(
    retrieved_documents: List[Dict],
    relevant_documents: List[int],
) -> float:

    if not retrieved_documents:
        return 0.0

    retrieved_indices = {
        doc["index"]
        for doc in retrieved_documents
    }

    relevant_indices = set(
        relevant_documents
    )

    found = (
        retrieved_indices
        & relevant_indices
    )

    return len(found) / len(retrieved_documents)


def evaluate_retrieval(
    retrieved_documents: List[Dict],
    relevant_documents: List[int],
) -> Dict:

    return {
        "precision_at_k": precision_at_k(
            retrieved_documents,
            relevant_documents,
        ),
        "recall_at_k": recall_at_k(
            retrieved_documents,
            relevant_documents,
        ),
    }