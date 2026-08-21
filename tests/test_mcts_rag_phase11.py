from actions.action import Action
from actions.state import ActionState
from mcts_rag import MCTSRAG
from retrieval import RetrievalResult


class TrackingRetrievalStrategy:

    def __init__(self):
        self.calls = []

    def retrieve(self, question):

        self.calls.append(question)

        return RetrievalResult(
            query=question,
            results=[
                {
                    "document": "MCTS is a tree search method.",
                    "score": 1,
                }
            ],
            reflection={
                "useful": True,
                "reason": "Relevant knowledge",
                "results": [
                    {
                        "document": (
                            "MCTS is a tree search method."
                        ),
                        "score": 1,
                    }
                ],
            },
            summary=(
                "MCTS is a tree search method."
            ),
        )


def test_end_to_end_single_question():

    pipeline = MCTSRAG(
        num_simulations=20,
    )

    result = pipeline.run(
        "What is MCTS?"
    )

    assert result.question == (
        "What is MCTS?"
    )

    assert result.root is not None

    assert result.tree


def test_multiple_rollouts_create_multiple_candidates():

    pipeline = MCTSRAG(
        num_simulations=20,
    )

    result = pipeline.run(
        "What is MCTS?"
    )

    assert len(result.candidates) > 1


def test_multiple_branches_are_explored():

    pipeline = MCTSRAG(
        num_simulations=20,
    )

    result = pipeline.run(
        "What is MCTS?"
    )

    assert len(
        result.root.children
    ) > 1


def test_retrieval_only_occurs_through_a4_a5():

    retrieval = TrackingRetrievalStrategy()

    pipeline = MCTSRAG(
        num_simulations=20,
        retrieval_strategy=retrieval,
    )

    result = pipeline.run(
        "What is MCTS?"
    )

    assert result.root is not None

    retrieval_actions = []

    def visit(node):

        if node.action in {
            Action.A4_RETRIEVAL_REASONING,
            Action.A5_RETRIEVAL_DECOMPOSE,
        }:
            retrieval_actions.append(
                node.action
            )

        for child in node.children:
            visit(child)

    visit(result.root)

    assert len(retrieval.calls) == (
        sum(
            1
            for action in retrieval_actions
            if action == Action.A4_RETRIEVAL_REASONING
        )
        +
        sum(
            1
            for action in retrieval_actions
            if action == Action.A5_RETRIEVAL_DECOMPOSE
        )
    )


def test_final_answer_comes_from_candidate():

    pipeline = MCTSRAG(
        num_simulations=20,
    )

    result = pipeline.run(
        "What is MCTS?"
    )

    assert result.final_answer is not None

    assert result.best_candidate is not None

    assert (
        result.final_answer
        == result.best_candidate.answer
    )


def test_best_candidate_has_trajectory():

    pipeline = MCTSRAG(
        num_simulations=20,
    )

    result = pipeline.run(
        "What is MCTS?"
    )

    assert result.best_candidate is not None

    assert (
        len(
            result.best_candidate.actions
        )
        > 0
    )

    assert (
        len(
            result.best_candidate.trajectory
        )
        > 0
    )


def test_tree_visualization_contains_root():

    pipeline = MCTSRAG(
        num_simulations=10,
    )

    result = pipeline.run(
        "What is MCTS?"
    )

    assert "ROOT" in result.tree


def test_tree_visualization_contains_actions():

    pipeline = MCTSRAG(
        num_simulations=20,
    )

    result = pipeline.run(
        "What is MCTS?"
    )

    assert "A1" in result.tree
    assert "A2" in result.tree


def test_empty_question_is_rejected():

    pipeline = MCTSRAG()

    try:
        pipeline.run("")
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Empty question should raise ValueError."
        )


def test_non_retrieval_actions_do_not_call_retrieval():

    retrieval = TrackingRetrievalStrategy()

    pipeline = MCTSRAG(
        num_simulations=6,
        retrieval_strategy=retrieval,
    )

    result = pipeline.run(
        "What is MCTS?"
    )

    for child in result.root.children:

        if child.action in {
            Action.A1_DIRECT_ANSWER,
            Action.A2_QUICK_REASONING,
            Action.A3_DECOMPOSE_QUESTION,
            Action.A6_SUMMARIZED_ANSWER,
        }:
            assert child.action not in {
                Action.A4_RETRIEVAL_REASONING,
                Action.A5_RETRIEVAL_DECOMPOSE,
            }

def test_evaluator_is_used_by_end_to_end_pipeline():

    received = []

    def evaluator(trajectory):

        received.append(
            trajectory
        )

        return 1.0

    pipeline = MCTSRAG(
        num_simulations=10,
        evaluator=evaluator,
    )

    result = pipeline.run(
        "What is MCTS?"
    )

    assert len(received) > 0

    assert result.root.visits > 0