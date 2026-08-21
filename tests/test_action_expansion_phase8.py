from actions.action import Action
from actions.integration import ActionExpansion
from actions.state import ActionState
from mcts.node import Node


def make_root() -> Node:
    return Node(
        state=ActionState(
            question="What is MCTS-RAG?"
        )
    )


def test_root_has_six_untried_actions():
    root = make_root()

    expansion = ActionExpansion()

    actions = expansion.get_untried_actions(root)

    assert actions == list(Action)


def test_expansion_creates_action_child():
    root = make_root()

    expansion = ActionExpansion()

    child = expansion.expand(root)

    assert child is not root
    assert child.parent is root
    assert child.action == Action.A1_DIRECT_ANSWER

    assert isinstance(
        child.state,
        ActionState,
    )


def test_expansion_removes_tried_action():
    root = make_root()

    expansion = ActionExpansion()

    first = expansion.expand(root)

    remaining = expansion.get_untried_actions(root)

    assert first.action not in remaining
    assert len(remaining) == 5


def test_all_six_actions_create_distinct_children():
    root = make_root()

    expansion = ActionExpansion()

    children = [
        expansion.expand(root)
        for _ in range(6)
    ]

    actions = [
        child.action
        for child in children
    ]

    assert len(children) == 6
    assert len(set(actions)) == 6
    assert set(actions) == set(Action)


def test_no_seventh_action_exists():
    root = make_root()

    expansion = ActionExpansion()

    for _ in range(6):
        expansion.expand(root)

    assert expansion.get_untried_actions(root) == []


def test_second_level_expansion_is_possible():
    root = make_root()

    expansion = ActionExpansion()

    first = expansion.create_child(
        root,
        Action.A2_QUICK_REASONING,
    )

    root.add_child(first)

    second = expansion.expand(first)

    assert second.parent is first
    assert second is not first

    assert isinstance(
        second.state,
        ActionState,
    )