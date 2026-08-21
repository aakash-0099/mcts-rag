import pytest

from actions.action import Action
from actions.integration import ActionExpansion
from actions.state import ActionState
from mcts.node import Node


def test_terminal_node_has_no_untried_actions():

    state = ActionState(
        question="What is MCTS?",
        content="Answer",
        metadata={
            "terminal": True,
            "candidate_answer": "Answer",
        },
    )

    node = Node(state=state)

    expansion = ActionExpansion()

    assert expansion.get_untried_actions(node) == []


def test_terminal_node_cannot_create_child():

    state = ActionState(
        question="What is MCTS?",
        content="Answer",
        metadata={
            "terminal": True,
            "candidate_answer": "Answer",
        },
    )

    node = Node(state=state)

    expansion = ActionExpansion()

    with pytest.raises(ValueError):
        expansion.create_child(
            node,
            Action.A2_QUICK_REASONING,
        )