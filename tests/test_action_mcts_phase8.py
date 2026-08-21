from actions.integration import (
    ActionExpansion,
    ActionSimulation,
)
from actions.state import ActionState
from mcts.mcts import MCTS
from mcts.node import Node


def make_root() -> Node:
    return Node(
        state=ActionState(
            question="What is MCTS-RAG?"
        )
    )


def test_action_layer_integrates_with_mcts():
    root = make_root()

    mcts = MCTS(
        expansion=ActionExpansion(),
        simulation=ActionSimulation(
            evaluator=lambda trajectory: 1.0
        ),
        num_simulations=6,
    )

    result = mcts.search(root)

    assert result is not None

    assert len(root.children) > 0


def test_mcts_creates_multiple_action_branches():
    root = make_root()

    mcts = MCTS(
        expansion=ActionExpansion(),
        simulation=ActionSimulation(
            evaluator=lambda trajectory: 1.0
        ),
        num_simulations=6,
    )

    mcts.search(root)

    actions = {
        child.action
        for child in root.children
    }

    assert len(actions) > 1


def test_mcts_preserves_action_state_tree():
    root = make_root()

    mcts = MCTS(
        expansion=ActionExpansion(),
        simulation=ActionSimulation(
            evaluator=lambda trajectory: 1.0
        ),
        num_simulations=10,
    )

    mcts.search(root)

    for child in root.children:
        assert child.parent is root
        assert isinstance(
            child.state,
            ActionState,
        )