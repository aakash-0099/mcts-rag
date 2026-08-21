from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from mcts.expansion import DefaultExpansion
from mcts.node import Node
from mcts.simulation import SimulationStrategy

from retrieval import (
    RetrievalPipeline,
    RetrievalStrategy,
)

from .action import Action
from .dispatcher import (
    execute_action,
    get_available_actions,
)
from .executors import (
    execute_a4,
    execute_a5,
)
from .state import ActionState


class ActionExpansion(DefaultExpansion):
    """
    Paper-specific expansion strategy.

    MCTS remains generic.

    Retrieval is injected into the action layer and is only
    used by A4 and A5.
    """

    def __init__(
        self,
        retrieval_strategy: RetrievalStrategy | None = None,
        llm=None,
    ):
        self.retrieval_strategy = (
            retrieval_strategy
            or RetrievalPipeline()
        )
        self.llm = llm


    def get_untried_actions(
        self,
        node: Node,
    ) -> list[Action]:

        if not isinstance(
            node.state,
            ActionState,
        ):
            raise TypeError(
                "ActionExpansion requires Node.state "
                "to be an ActionState."
            )

        if node.is_terminal:
            return []

        available_actions = get_available_actions(
            node.state
        )

        tried_actions = {
            child.action
            for child in node.children
            if child.action is not None
        }

        return [
            action
            for action in available_actions
            if action not in tried_actions
        ]

    def create_child(
        self,
        node: Node,
        action: Action,
    ) -> Node:
        print(
            f"[EXPANSION] Executing action: {action.value}"
        )
        if not isinstance(node.state, ActionState):
            raise TypeError(
                "ActionExpansion requires Node.state "
                "to be an ActionState."
            )

        if node.is_terminal:
            raise ValueError(
                "Cannot expand a terminal node."
            )

        if action == Action.A4_RETRIEVAL_REASONING:
            child_state = execute_a4(
                node.state,
                self.retrieval_strategy,
                self.llm,
            )

        elif action == Action.A5_RETRIEVAL_DECOMPOSE:
            child_state = execute_a5(
                node.state,
                self.retrieval_strategy,
                llm=self.llm,
            )

        else:
            child_state = execute_action(
                action,
                node.state,
                llm=self.llm,
            )

        child = Node(
            state=child_state,
            parent=node,
            action=action,
        )
        print(
            f"[EXPANSION] Created child: "
            f"action={action.value}, "
            f"terminal={child_state.is_terminal}"
        )
        # Mark this node fully expanded once its last untried
        # action has been consumed, so selection stops re-expanding
        # it and instead descends via UCT into its children.
        remaining = [
            a for a in self.get_untried_actions(node)
            if a != action
        ]

        if not remaining:
            node.mark_fully_expanded()

        return child


@dataclass
class ActionTrajectory:
    """
    One root-to-node MCTS trajectory.
    """

    states: list[ActionState]
    actions: list[Action]

    @property
    def current_state(self) -> ActionState:
        return self.states[-1]

    @property
    def depth(self) -> int:
        return len(self.actions)


class ActionSimulation(SimulationStrategy):
    """
    Simulation strategy for action-driven MCTS.

    The evaluator is injected.

    Retrieval logic does NOT belong here.
    """

    def __init__(
        self,
        evaluator: Callable[
            [ActionTrajectory],
            float,
        ] | None = None,
    ):
        self.evaluator = evaluator

    def simulate(
        self,
        node: Node,
    ) -> float:

        trajectory = self.build_trajectory(
            node
        )

        if self.evaluator is None:
            return 0.0

        return float(
            self.evaluator(
                trajectory
            )
        )

    def build_trajectory(
        self,
        node: Node,
    ) -> ActionTrajectory:

        nodes: list[Node] = []

        current = node

        while current is not None:
            nodes.append(current)
            current = current.parent

        nodes.reverse()

        states: list[ActionState] = []
        actions: list[Action] = []

        for current in nodes:

            if current.state is not None:
                states.append(
                    current.state
                )

            if current.action is not None:
                actions.append(
                    current.action
                )

        return ActionTrajectory(
            states=states,
            actions=actions,
        )