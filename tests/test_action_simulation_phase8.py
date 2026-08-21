from actions.action import Action
from actions.integration import (
    ActionSimulation,
    ActionTrajectory,
)
from actions.state import ActionState
from mcts.node import Node


def make_root() -> Node:
    return Node(
        state=ActionState(
            question="What is MCTS-RAG?"
        )
    )


def test_simulation_implements_generic_interface():
    from mcts.simulation import SimulationStrategy

    simulation = ActionSimulation()

    assert isinstance(
        simulation,
        SimulationStrategy,
    )


def test_trajectory_contains_root_and_child():
    root = make_root()

    child_state = root.state.child(
        content="Intermediate result"
    )

    child = Node(
        state=child_state,
        parent=root,
        action=Action.A2_QUICK_REASONING,
    )

    simulation = ActionSimulation()

    trajectory = simulation.build_trajectory(
        child
    )

    assert isinstance(
        trajectory,
        ActionTrajectory,
    )

    assert len(trajectory.states) == 2
    assert len(trajectory.actions) == 1

    assert trajectory.actions[0] == (
        Action.A2_QUICK_REASONING
    )


def test_trajectory_depth():
    root = make_root()

    child_state = root.state.child(
        content="Result"
    )

    child = Node(
        state=child_state,
        parent=root,
        action=Action.A2_QUICK_REASONING,
    )

    simulation = ActionSimulation()

    trajectory = simulation.build_trajectory(
        child
    )

    assert trajectory.depth == 1


def test_evaluator_receives_trajectory():
    root = make_root()

    child_state = root.state.child(
        content="Result"
    )

    child = Node(
        state=child_state,
        parent=root,
        action=Action.A1_DIRECT_ANSWER,
    )

    received = []

    def evaluator(trajectory):
        received.append(trajectory)
        return 0.75

    simulation = ActionSimulation(
        evaluator=evaluator
    )

    reward = simulation.simulate(child)

    assert reward == 0.75
    assert len(received) == 1

    assert received[0].actions == [
        Action.A1_DIRECT_ANSWER
    ]


def test_default_simulation_is_neutral():
    root = make_root()

    simulation = ActionSimulation()

    assert simulation.simulate(root) == 0.0