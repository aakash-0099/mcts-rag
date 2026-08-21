from actions.action import Action
from actions.state import ActionState
from actions.terminal import (
    is_terminal_action,
    is_terminal_state,
)


def test_a1_is_answer_action():
    assert is_terminal_action(
        Action.A1_DIRECT_ANSWER
    )


def test_a6_is_answer_action():
    assert is_terminal_action(
        Action.A6_SUMMARIZED_ANSWER
    )


def test_a2_is_not_answer_action():
    assert not is_terminal_action(
        Action.A2_QUICK_REASONING
    )


def test_a3_is_not_answer_action():
    assert not is_terminal_action(
        Action.A3_DECOMPOSE_QUESTION
    )


def test_a4_is_not_answer_action():
    assert not is_terminal_action(
        Action.A4_RETRIEVAL_REASONING
    )


def test_a5_is_not_answer_action():
    assert not is_terminal_action(
        Action.A5_RETRIEVAL_DECOMPOSE
    )


def test_state_is_not_terminal_by_default():
    state = ActionState(
        question="What is MCTS?"
    )

    assert not is_terminal_state(state)


def test_terminal_state():
    state = ActionState(
        question="What is MCTS?",
        content="MCTS is a search algorithm.",
        metadata={
            "terminal": True,
            "candidate_answer": (
                "MCTS is a search algorithm."
            ),
        },
    )

    assert is_terminal_state(state)


def test_candidate_answer_property():
    state = ActionState(
        question="What is MCTS?",
        metadata={
            "terminal": True,
            "candidate_answer": "MCTS is a search algorithm.",
        },
    )

    assert state.candidate_answer == (
        "MCTS is a search algorithm."
    )