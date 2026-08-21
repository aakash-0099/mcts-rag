from __future__ import annotations

from mcts.node import Node

from .action import Action
from .state import ActionState


def reconstruct_actions(
    node: Node,
) -> list[Action]:
    """
    Reconstruct actions from root to the supplied node.
    """

    actions: list[Action] = []

    current = node

    while current is not None:
        if current.action is not None:
            actions.append(current.action)

        current = current.parent

    actions.reverse()

    return actions


def reconstruct_states(
    node: Node,
) -> list[ActionState]:
    """
    Reconstruct states from root to the supplied node.
    """

    states: list[ActionState] = []

    current = node

    while current is not None:
        if current.state is not None:
            states.append(current.state)

        current = current.parent

    states.reverse()

    return states


def reconstruct_trajectory(
    node: Node,
) -> list[tuple[Action | None, ActionState]]:
    """
    Reconstruct the complete root-to-node trajectory.

    Each entry is:

        (action_that_created_node, resulting_state)
    """

    trajectory: list[
        tuple[Action | None, ActionState]
    ] = []

    current = node

    while current is not None:
        trajectory.append(
            (
                current.action,
                current.state,
            )
        )

        current = current.parent

    trajectory.reverse()

    return trajectory