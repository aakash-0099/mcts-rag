import pytest

from mcts import MCTS, Node
from mcts.expansion import ExpansionStrategy
from mcts.simulation import SimulationStrategy


# ---------------------------------------------------------
# Test Node
# ---------------------------------------------------------

class BanditNode(Node):
    """
    A tiny problem for testing MCTS.

    The root has three possible actions:

        A -> reward 0.2
        B -> reward 0.8
        C -> reward 0.5

    MCTS should eventually identify B as the best action.
    """

    ACTIONS = {
        "A": 0.2,
        "B": 0.8,
        "C": 0.5,
    }

    def __init__(self, state, parent=None, action=None):
        super().__init__(
            state=state,
            parent=parent,
            action=action,
        )

        self.untried_actions = list(self.ACTIONS.keys())

    @property
    def is_terminal(self) -> bool:
        return self.action is not None

    @property
    def is_fully_expanded(self) -> bool:
        return len(self.untried_actions) == 0


# ---------------------------------------------------------
# Expansion
# ---------------------------------------------------------

class BanditExpansion(ExpansionStrategy):

    def expand(self, node: BanditNode) -> BanditNode:

        if not node.untried_actions:
            return node

        action = node.untried_actions.pop(0)

        child = BanditNode(
            state=action,
            parent=node,
            action=action,
        )

        node.add_child(child)

        return child


# ---------------------------------------------------------
# Simulation
# ---------------------------------------------------------

class BanditSimulation(SimulationStrategy):

    def simulate(self, node: BanditNode) -> float:
        """
        Return the reward associated with the selected action.
        """

        return BanditNode.ACTIONS[node.action]


# ---------------------------------------------------------
# Tests
# ---------------------------------------------------------

def test_mcts_finds_best_action():
    """
    MCTS should identify action B because it has the
    highest reward.
    """

    root = BanditNode(state="root")

    mcts = MCTS(
        expansion=BanditExpansion(),
        simulation=BanditSimulation(),
        num_simulations=100,
    )

    best = mcts.search(root)

    assert best.action == "B"


def test_root_is_updated_during_search():
    """
    Every simulation should update the root's visit count.
    """

    root = BanditNode(state="root")

    mcts = MCTS(
        expansion=BanditExpansion(),
        simulation=BanditSimulation(),
        num_simulations=100,
    )

    mcts.search(root)

    assert root.visits == 100


def test_all_actions_are_expanded():
    """
    The root should eventually contain all possible actions.
    """

    root = BanditNode(state="root")

    mcts = MCTS(
        expansion=BanditExpansion(),
        simulation=BanditSimulation(),
        num_simulations=100,
    )

    mcts.search(root)

    actions = {
        child.action
        for child in root.children
    }

    assert actions == {"A", "B", "C"}


def test_best_action_has_highest_value():
    """
    The best action should have the highest average reward.
    """

    root = BanditNode(state="root")

    mcts = MCTS(
        expansion=BanditExpansion(),
        simulation=BanditSimulation(),
        num_simulations=100,
    )

    mcts.search(root)

    best_child = max(
        root.children,
        key=lambda child: child.value,
    )

    assert best_child.action == "B"
    assert best_child.value > 0.7


def test_children_have_visits():
    """
    Each expanded action should have been visited.
    """

    root = BanditNode(state="root")

    mcts = MCTS(
        expansion=BanditExpansion(),
        simulation=BanditSimulation(),
        num_simulations=100,
    )

    mcts.search(root)

    for child in root.children:
        assert child.visits > 0


def test_mcts_returns_root_if_no_children():
    """
    MCTS should safely handle a node that cannot be expanded.
    """

    root = BanditNode(state="root")

    # Prevent expansion.
    root.untried_actions = []

    mcts = MCTS(
        expansion=BanditExpansion(),
        simulation=BanditSimulation(),
        num_simulations=10,
    )

    result = mcts.search(root)

    assert result == root