import math

from mcts.selection import UCTSelection


class FakeNode:
    """Minimal node used to test UCT selection independently."""

    def __init__(
        self,
        visits=0,
        total_reward=0.0,
        children=None,
        is_terminal=False,
        is_fully_expanded=True,
    ):
        self.visits = visits
        self.total_reward = total_reward
        self.children = children or []
        self.is_terminal = is_terminal
        self.is_fully_expanded = is_fully_expanded


def test_uct_score_matches_formula():
    selection = UCTSelection(exploration_constant=1.414)

    parent = FakeNode(visits=10)

    child = FakeNode(
        visits=5,
        total_reward=4.0,
    )

    score = selection._uct_score(child, parent)

    expected = (
        4.0 / 5
        + 1.414 * math.sqrt(math.log(10) / 5)
    )

    assert math.isclose(score, expected)


def test_unvisited_child_gets_infinite_score():
    selection = UCTSelection()

    parent = FakeNode(visits=10)
    child = FakeNode(visits=0)

    score = selection._uct_score(child, parent)

    assert score == float("inf")


def test_best_child_selects_highest_uct_score():
    selection = UCTSelection(exploration_constant=1.414)

    parent = FakeNode(visits=13)

    a = FakeNode(
        visits=10,
        total_reward=0.8,
    )

    b = FakeNode(
        visits=2,
        total_reward=0.7,
    )

    c = FakeNode(
        visits=1,
        total_reward=0.3,
    )

    parent.children = [a, b, c]

    selected = selection._best_child(parent)

    scores = [
        selection._uct_score(child, parent)
        for child in parent.children
    ]

    expected = parent.children[scores.index(max(scores))]

    assert selected is expected


def test_low_visit_child_gets_exploration_bonus():
    selection = UCTSelection(exploration_constant=1.414)

    parent = FakeNode(visits=100)

    frequently_visited = FakeNode(
        visits=50,
        total_reward=40.0,
    )

    rarely_visited = FakeNode(
        visits=2,
        total_reward=1.6,
    )

    frequent_score = selection._uct_score(
        frequently_visited,
        parent,
    )

    rare_score = selection._uct_score(
        rarely_visited,
        parent,
    )

    assert rare_score > frequent_score


def test_select_stops_at_unexpanded_node():
    selection = UCTSelection()

    node = FakeNode(
        visits=5,
        is_fully_expanded=False,
    )

    selected = selection.select(node)

    assert selected is node


def test_select_prioritizes_unvisited_child():
    selection = UCTSelection()

    parent = FakeNode(visits=10)

    visited_child = FakeNode(
        visits=10,
        total_reward=8.0,
    )

    unvisited_child = FakeNode(
        visits=0,
        total_reward=0.0,
    )

    parent.children = [
        visited_child,
        unvisited_child,
    ]

    selected = selection._best_child(parent)

    assert selected is unvisited_child