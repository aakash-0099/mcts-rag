from actions.action import Action
from actions.integration import ActionExpansion
from actions.state import ActionState
from actions.trajectory import (
    reconstruct_actions,
    reconstruct_states,
)
from mcts.node import Node


def make_root() -> Node:
    return Node(
        state=ActionState(
            question="What is MCTS-RAG?"
        )
    )


def test_multiple_branches_exist():
    root = make_root()

    expansion = ActionExpansion()

    branch_a = expansion.expand(root)
    branch_b = expansion.expand(root)
    branch_c = expansion.expand(root)

    assert branch_a.parent is root
    assert branch_b.parent is root
    assert branch_c.parent is root

    assert len(root.children) == 3

    assert len({
        branch_a.action,
        branch_b.action,
        branch_c.action,
    }) == 3


def test_branches_can_continue_independently():
    root = make_root()

    expansion = ActionExpansion()

    branch_a = expansion.create_child(
        root,
        Action.A2_QUICK_REASONING,
    )

    branch_b = expansion.create_child(
        root,
        Action.A3_DECOMPOSE_QUESTION,
    )

    root.add_child(branch_a)
    root.add_child(branch_b)

    branch_a_child = expansion.expand(branch_a)
    branch_b_child = expansion.expand(branch_b)

    assert branch_a_child.parent is branch_a
    assert branch_b_child.parent is branch_b

    assert branch_a_child.parent is not branch_b
    assert branch_b_child.parent is not branch_a


def test_trajectory_can_be_reconstructed():
    root = make_root()

    expansion = ActionExpansion()

    first = expansion.create_child(
        root,
        Action.A2_QUICK_REASONING,
    )

    root.add_child(first)

    second = expansion.expand(first)

    actions = reconstruct_actions(second)
    states = reconstruct_states(second)

    assert len(actions) == 2
    assert len(states) == 3

    assert actions[0] == first.action
    assert actions[1] == second.action


def test_different_branches_have_different_trajectories():
    root = make_root()

    expansion = ActionExpansion()

    branch_a = expansion.expand(root)
    branch_b = expansion.expand(root)

    actions_a = reconstruct_actions(branch_a)
    actions_b = reconstruct_actions(branch_b)

    assert actions_a != actions_b

def test_terminal_a1_cannot_be_expanded():

    root = make_root()

    expansion = ActionExpansion()

    first = expansion.expand(root)

    assert first.action == Action.A1_DIRECT_ANSWER
    assert first.is_terminal

    assert expansion.get_untried_actions(first) == []

import pytest


def test_terminal_a1_cannot_create_child():

    root = make_root()

    expansion = ActionExpansion()

    first = expansion.expand(root)

    assert first.is_terminal

    with pytest.raises(ValueError):
        expansion.create_child(
            first,
            Action.A2_QUICK_REASONING,
        )