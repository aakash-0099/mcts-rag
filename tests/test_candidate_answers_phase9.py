from actions.action import Action
from actions.candidates import (
    CandidateAnswerCollector,
)
from actions.state import ActionState
from actions.trajectory import reconstruct_trajectory
from mcts.node import Node


def make_root() -> Node:
    return Node(
        state=ActionState(
            question="What is MCTS?"
        )
    )


def make_terminal_child(
    root: Node,
    action: Action,
    answer: str,
) -> Node:

    state = root.state.child(
        content=answer,
        metadata={
            "terminal": True,
            "candidate_answer": answer,
        },
    )

    child = Node(
        state=state,
        parent=root,
        action=action,
    )

    root.add_child(child)

    return child


def test_a1_produces_candidate_answer():
    root = make_root()

    child = make_terminal_child(
        root,
        Action.A1_DIRECT_ANSWER,
        "Answer from A1",
    )

    assert child.is_terminal
    assert child.state.candidate_answer == (
        "Answer from A1"
    )


def test_a6_produces_candidate_answer():
    root = make_root()

    child = make_terminal_child(
        root,
        Action.A6_SUMMARIZED_ANSWER,
        "Answer from A6",
    )

    assert child.is_terminal
    assert child.state.candidate_answer == (
        "Answer from A6"
    )


def test_candidate_collector_collects_answer():
    root = make_root()

    child = make_terminal_child(
        root,
        Action.A1_DIRECT_ANSWER,
        "Answer 1",
    )

    collector = CandidateAnswerCollector()

    candidate = collector.add(
        child,
        reward=0.8,
    )

    assert candidate is not None
    assert candidate.answer == "Answer 1"
    assert candidate.reward == 0.8
    assert candidate.actions == [
        Action.A1_DIRECT_ANSWER
    ]


def test_multiple_candidates_are_collected():
    root = make_root()

    child1 = make_terminal_child(
        root,
        Action.A1_DIRECT_ANSWER,
        "Answer 1",
    )

    child2 = make_terminal_child(
        root,
        Action.A6_SUMMARIZED_ANSWER,
        "Answer 2",
    )

    collector = CandidateAnswerCollector()

    collector.add(child1, 0.8)
    collector.add(child2, 0.9)

    assert len(collector) == 2

    assert collector.answers == [
        "Answer 1",
        "Answer 2",
    ]


def test_candidate_contains_full_trajectory():
    root = make_root()

    intermediate_state = root.state.child(
        content="Reasoning"
    )

    intermediate = Node(
        state=intermediate_state,
        parent=root,
        action=Action.A3_DECOMPOSE_QUESTION,
    )

    root.add_child(intermediate)

    terminal_state = intermediate.state.child(
        content="Final answer",
        metadata={
            "terminal": True,
            "candidate_answer": "Final answer",
        },
    )

    terminal = Node(
        state=terminal_state,
        parent=intermediate,
        action=Action.A6_SUMMARIZED_ANSWER,
    )

    intermediate.add_child(terminal)

    collector = CandidateAnswerCollector()

    candidate = collector.add(
        terminal,
        reward=1.0,
    )

    assert candidate.actions == [
        Action.A3_DECOMPOSE_QUESTION,
        Action.A6_SUMMARIZED_ANSWER,
    ]

    assert len(candidate.states) == 3