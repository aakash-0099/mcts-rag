import pytest

from actions import Action, ActionState

from mcts.action_expansion import PaperActionExpansion
from mcts.node import Node


def make_root() -> Node:
    state = ActionState(
        question="What is 2 + 2?"
    )

    return Node(
        state=state,
    )


def test_paper_action_expansion_is_an_expansion_strategy():
    from mcts.expansion import ExpansionStrategy

    expansion = PaperActionExpansion()

    assert isinstance(
        expansion,
        ExpansionStrategy,
    )


def test_get_untried_actions_returns_all_six_actions_for_new_node():
    root = make_root()

    expansion = PaperActionExpansion()

    actions = expansion.get_untried_actions(root)

    assert actions == list(Action)


def test_expand_creates_one_child():
    root = make_root()

    expansion = PaperActionExpansion()

    child = expansion.expand(root)

    assert child is not root

    assert len(root.children) == 1

    assert root.children[0] is child


def test_first_expansion_creates_a1():
    root = make_root()

    expansion = PaperActionExpansion()

    child = expansion.expand(root)

    assert child.action == Action.A1_DIRECT_ANSWER


def test_child_has_correct_parent():
    root = make_root()

    expansion = PaperActionExpansion()

    child = expansion.expand(root)

    assert child.parent is root


def test_child_has_valid_state():
    root = make_root()

    expansion = PaperActionExpansion()

    child = expansion.expand(root)

    assert isinstance(
        child.state,
        ActionState,
    )

    assert child.state.question == root.state.question

    assert child.state.content != ""


def test_second_expansion_creates_a2():
    root = make_root()

    expansion = PaperActionExpansion()

    first_child = expansion.expand(root)
    second_child = expansion.expand(root)

    assert first_child.action == Action.A1_DIRECT_ANSWER
    assert second_child.action == Action.A2_QUICK_REASONING

    assert len(root.children) == 2


def test_all_six_actions_can_be_expanded():
    root = make_root()

    expansion = PaperActionExpansion()

    children = []

    for _ in range(6):
        child = expansion.expand(root)
        children.append(child)

    actions = [
        child.action
        for child in children
    ]

    assert actions == list(Action)

    assert len(root.children) == 6


def test_all_six_children_have_same_parent():
    root = make_root()

    expansion = PaperActionExpansion()

    for _ in range(6):
        expansion.expand(root)

    assert len(root.children) == 6

    for child in root.children:
        assert child.parent is root


def test_all_six_children_have_valid_states():
    root = make_root()

    expansion = PaperActionExpansion()

    for _ in range(6):
        expansion.expand(root)

    for child in root.children:
        assert isinstance(
            child.state,
            ActionState,
        )

        assert child.state.question == root.state.question

        assert child.state.content != ""


def test_actions_are_not_expanded_twice():
    root = make_root()

    expansion = PaperActionExpansion()

    for _ in range(6):
        expansion.expand(root)

    actions = [
        child.action
        for child in root.children
    ]

    assert len(actions) == len(set(actions))


def test_no_untried_actions_after_all_six_are_expanded():
    root = make_root()

    expansion = PaperActionExpansion()

    for _ in range(6):
        expansion.expand(root)

    assert expansion.get_untried_actions(root) == []


def test_expand_returns_same_node_when_fully_expanded():
    root = make_root()

    expansion = PaperActionExpansion()

    for _ in range(6):
        expansion.expand(root)

    result = expansion.expand(root)

    assert result is root

    assert len(root.children) == 6


@pytest.mark.parametrize(
    "action",
    list(Action),
)
def test_each_action_is_stored_on_its_child_node(action):
    root = make_root()

    expansion = PaperActionExpansion()

    child = expansion.create_child(
        root,
        action,
    )

    assert child.action == action

    assert child.parent is root

    assert isinstance(
        child.state,
        ActionState,
    )


def test_parent_state_is_not_modified_by_expansion():
    root = make_root()

    original_question = root.state.question
    original_content = root.state.content
    original_metadata = root.state.metadata.copy()

    expansion = PaperActionExpansion()

    expansion.expand(root)

    assert root.state.question == original_question
    assert root.state.content == original_content
    assert root.state.metadata == original_metadata


def test_a4_expansion_does_not_perform_retrieval():
    root = make_root()

    expansion = PaperActionExpansion()

    # Expand A1, A2, A3 first.
    expansion.expand(root)
    expansion.expand(root)
    expansion.expand(root)

    child = expansion.expand(root)

    assert child.action == Action.A4_RETRIEVAL_REASONING
    assert child.state.metadata == {}


def test_a5_expansion_does_not_perform_retrieval():
    root = make_root()

    expansion = PaperActionExpansion()

    # Expand A1-A4 first.
    for _ in range(4):
        expansion.expand(root)

    child = expansion.expand(root)

    assert child.action == Action.A5_RETRIEVAL_DECOMPOSE
    assert child.state.metadata == {}


def test_expansion_works_with_independent_root_nodes():
    expansion = PaperActionExpansion()

    root1 = make_root()
    root2 = make_root()

    child1 = expansion.expand(root1)
    child2 = expansion.expand(root2)

    assert child1.action == Action.A1_DIRECT_ANSWER
    assert child2.action == Action.A1_DIRECT_ANSWER

    assert len(root1.children) == 1
    assert len(root2.children) == 1