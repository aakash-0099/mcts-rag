from .state import ActionState
from .decomposition import (
    DecompositionStrategy,
    DefaultDecomposition,
)
from .llm import LLM

from retrieval import (
    RetrievalStrategy,
    RetrievalPipeline,
)


def _context_block(state: ActionState) -> str:
    """
    Return any reasoning/retrieval content already accumulated
    on this state, formatted for inclusion in a prompt. Empty
    string if nothing has been accumulated yet (e.g. state is
    the root).
    """

    if state.content:
        return f"\n\nPrior reasoning / retrieved knowledge:\n{state.content}\n"

    return ""


def execute_a1(
    state: ActionState,
    llm: LLM | None = None,
) -> ActionState:
    """
    A1 — Direct Answer.

    Uses any context already accumulated on this state (e.g. if
    A1 is expanded as a child of A4/A5, it must use what was
    retrieved, not answer cold).
    """

    answer = None

    if llm is not None:
        prompt = (
            f"Answer the following question directly and concisely, "
            f"using the context below if it is provided and relevant. "
            f"If the context contradicts your own knowledge, trust the "
            f"context.\n"
            f"{_context_block(state)}\n"
            f"Question: {state.question}\n\nAnswer:"
        )
        answer = llm.complete(prompt)

        if answer is not None:
            answer = answer.strip()

    if not answer:
        answer = f"Direct answer for: {state.question}"

    return state.child(
        content=answer,
        metadata={
            "terminal": True,
            "candidate_answer": answer,
        },
    )


def execute_a2(
    state: ActionState,
    llm: LLM | None = None,
) -> ActionState:
    """
    A2 — Quick Reasoning.
    """

    if llm is not None:
        prompt = (
            f"Think step by step, briefly, about the following question, "
            f"using the context below if it is provided and relevant. "
            f"Do not give a final answer yet, just the reasoning.\n"
            f"{_context_block(state)}\n"
            f"Question: {state.question}\n\nReasoning:"
        )
        reasoning = llm.complete(prompt)
    else:
        reasoning = f"Quick reasoning for: {state.question}"

    return state.child(content=reasoning)


def execute_a3(
    state: ActionState,
    llm: LLM | None = None,
) -> ActionState:
    """
    A3 — Decompose Question.
    """

    decomposition_strategy = DefaultDecomposition()

    subquestions = decomposition_strategy.decompose(
        state.question
    )

    return state.child(
        content=(
            f"Decomposed question: {state.question}\n"
            f"Subquestions: {subquestions}"
        ),
    )


def execute_a4(
    state: ActionState,
    retrieval_strategy: RetrievalStrategy | None = None,
    llm: LLM | None = None,
) -> ActionState:
    """
    A4 — Retrieval Reasoning.
    """
    print(
        f"[A4] Retrieving for: {state.question}"
    )
    if retrieval_strategy is None:
        retrieval_strategy = RetrievalPipeline()

    result = retrieval_strategy.retrieve(
        state.question
    )
    print(
        f"[A4] Query: {result.query}"
    )

    print(
        f"[A4] Retrieved {len(result.results)} results"
    )
    if llm is not None:
        prompt = (
            f"Using the knowledge below, reason about the question. "
            f"Do not give a final answer yet, just the reasoning.\n\n"
            f"Knowledge:\n{result.summary}\n\n"
            f"Question: {state.question}\n\nReasoning:"
        )
        reasoning = llm.complete(prompt)
    else:
        reasoning = (
            f"Retrieval reasoning for: {state.question}\n"
            f"Query: {result.query}\n"
            f"Knowledge: {result.summary}"
        )

    # return state.child(content=reasoning)
    return state.child(
        content=reasoning,
        metadata={
            "retrieval": True,
            "retrieval_action": "A4",
            "query": result.query,
            "retrieved_question_ids": [
                item["document"]["question_id"]
                for item in result.results
                if isinstance(item.get("document"), dict)
                and item["document"].get("question_id") is not None
            ],
            "retrieved_doc_ids": [
                item["document"]["doc_id"]
                for item in result.results
                if isinstance(item.get("document"), dict)
                and item["document"].get("doc_id") is not None
            ],
        },
    )


def execute_a5(
    state: ActionState,
    retrieval_strategy: RetrievalStrategy | None = None,
    decomposition_strategy: DecompositionStrategy | None = None,
    llm: LLM | None = None,
) -> ActionState:
    """
    A5 — Retrieval + Decompose.
    """

    if retrieval_strategy is None:
        retrieval_strategy = RetrievalPipeline()

    if decomposition_strategy is None:
        decomposition_strategy = DefaultDecomposition()

    subquestions = decomposition_strategy.decompose(
        state.question
    )
    print(
        f"[A5] Decomposed into: {subquestions}"
    )
    retrieval_results = []

    for subquestion in subquestions:
        print(
            f"[A5] Retrieving subquestion: {subquestion}"
        )
        result = retrieval_strategy.retrieve(
            subquestion
        )

        retrieval_results.append(result)

    summaries = [
        result.summary
        for result in retrieval_results
        if result.summary
    ]

    if llm is not None:
        knowledge_block = "\n\n".join(summaries)
        prompt = (
            f"Using the knowledge below, reason about the question. "
            f"Do not give a final answer yet, just the reasoning.\n\n"
            f"Subquestions: {subquestions}\n\n"
            f"Knowledge:\n{knowledge_block}\n\n"
            f"Question: {state.question}\n\nReasoning:"
        )
        reasoning = llm.complete(prompt)
    else:
        reasoning = (
            f"Retrieval + decomposition for: "
            f"{state.question}\n"
            f"Subquestions: {subquestions}\n"
            f"Knowledge: {summaries}"
        )

    # return state.child(content=reasoning)
    return state.child(
        content=reasoning,
        metadata={
            "retrieval": True,
            "retrieval_action": "A5",
            "queries": [
                result.query
                for result in retrieval_results
            ],
            "retrieved_question_ids": list(
                dict.fromkeys(
                    question_id
                    for result in retrieval_results
                    for item in result.results
                    if isinstance(item.get("document"), dict)
                    for question_id in [
                        item["document"].get("question_id")
                    ]
                    if question_id is not None
                )
            ),
            "retrieved_doc_ids": list(
                dict.fromkeys(
                    doc_id
                    for result in retrieval_results
                    for item in result.results
                    if isinstance(item.get("document"), dict)
                    for doc_id in [
                        item["document"].get("doc_id")
                    ]
                    if doc_id is not None
                )
            ),
        },
    )


def execute_a6(
    state: ActionState,
    llm: LLM | None = None,
) -> ActionState:
    """
    A6 — Summarized Answer.

    Synthesizes a final answer from whatever has been
    accumulated on this state so far.
    """

    answer = None

    if llm is not None:
        prompt = (
            f"Based on the reasoning and/or retrieved knowledge below, "
            f"give a final, concise answer to the question. If no "
            f"reasoning or knowledge is provided, answer from your own "
            f"knowledge but say so is uncertain.\n"
            f"{_context_block(state)}\n"
            f"Question: {state.question}\n\nFinal answer:"
        )
        answer = llm.complete(prompt)

        if answer is not None:
            answer = answer.strip()

    if not answer:
        answer = f"Summarized answer for: {state.question}"

    return state.child(
        content=answer,
        metadata={
            "terminal": True,
            "candidate_answer": answer,
        },
    )