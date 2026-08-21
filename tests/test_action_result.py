from retrieval import RetrievalResult

from actions.executors import (
    execute_a1,
    execute_a2,
    execute_a3,
    execute_a4,
    execute_a5,
    execute_a6,
)

from actions.state import ActionState


class TrackingRetrievalStrategy:

    def __init__(self):
        self.calls = []

    def retrieve(self, question):
        self.calls.append(question)

        return RetrievalResult(
            query=question,
            results=[
                {
                    "document": "Test knowledge",
                    "score": 1,
                }
            ],
            reflection={
                "useful": True,
                "reason": "Test retrieval",
                "results": [
                    {
                        "document": "Test knowledge",
                        "score": 1,
                    }
                ],
            },
            summary="Test knowledge",
        )


def make_state():
    return ActionState(
        question="What is MCTS?",
    )


def test_a4_invokes_retrieval():
    retrieval = TrackingRetrievalStrategy()

    result = execute_a4(
        make_state(),
        retrieval,
    )

    assert retrieval.calls == [
        "What is MCTS?"
    ]

    assert "Retrieval reasoning" in result.content
    assert "Test knowledge" in result.content


def test_a5_invokes_retrieval():
    retrieval = TrackingRetrievalStrategy()

    result = execute_a5(
        make_state(),
        retrieval,
    )

    assert retrieval.calls == [
        "What is MCTS?"
    ]

    assert "Retrieval + decomposition" in result.content
    assert "Test knowledge" in result.content


def test_a1_does_not_invoke_retrieval():
    retrieval = TrackingRetrievalStrategy()

    execute_a1(make_state())

    assert retrieval.calls == []


def test_a2_does_not_invoke_retrieval():
    retrieval = TrackingRetrievalStrategy()

    execute_a2(make_state())

    assert retrieval.calls == []


def test_a3_does_not_invoke_retrieval():
    retrieval = TrackingRetrievalStrategy()

    execute_a3(make_state())

    assert retrieval.calls == []


def test_a6_does_not_invoke_retrieval():
    retrieval = TrackingRetrievalStrategy()

    execute_a6(make_state())

    assert retrieval.calls == []

def test_non_retrieval_actions_do_not_require_retrieval():
    state = make_state()

    result_a1 = execute_a1(state)
    result_a2 = execute_a2(state)
    result_a3 = execute_a3(state)
    result_a6 = execute_a6(state)

    assert result_a1 is not None
    assert result_a2 is not None
    assert result_a3 is not None
    assert result_a6 is not None

from retrieval import RetrievalResult

from actions.executors import execute_a5
from actions.state import ActionState


class TrackingRetrievalStrategy:

    def __init__(self):
        self.calls = []

    def retrieve(self, question):
        self.calls.append(question)

        return RetrievalResult(
            query=question,
            results=[
                {
                    "document": "Test knowledge",
                    "score": 1,
                }
            ],
            reflection={
                "useful": True,
                "reason": "Test retrieval",
                "results": [
                    {
                        "document": "Test knowledge",
                        "score": 1,
                    }
                ],
            },
            summary="Test knowledge",
        )
class TrackingDecompositionStrategy:

    def __init__(self):
        self.calls = []

    def decompose(self, question):
        self.calls.append(question)

        return [
            "First subquestion",
            "Second subquestion",
        ]


def test_a5_uses_decomposition_and_retrieval():

    retrieval = TrackingRetrievalStrategy()
    decomposition = TrackingDecompositionStrategy()

    state = ActionState(
        question="What is MCTS?"
    )

    result = execute_a5(
        state,
        retrieval_strategy=retrieval,
        decomposition_strategy=decomposition,
    )

    # Decomposition was invoked.
    assert decomposition.calls == [
        "What is MCTS?"
    ]

    # Retrieval was invoked once per subquestion.
    assert retrieval.calls == [
        "First subquestion",
        "Second subquestion",
    ]

    assert "First subquestion" in result.content
    assert "Second subquestion" in result.content