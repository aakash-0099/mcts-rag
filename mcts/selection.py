import math

from .node import Node


class UCTSelection:

    def __init__(self, exploration_constant: float = 1.414):
        self.exploration_constant = exploration_constant

    def select(self, node: Node) -> Node:
        current = node

        while True:

            # Terminal node
            if current.is_terminal:
                return current

            # Node still has actions that can be expanded
            if not current.is_fully_expanded:
                return current

            # Fully expanded node with no children
            if not current.children:
                return current

            current = self._best_child(current)

    def _best_child(self, node: Node) -> Node:

        best_child = None
        best_score = float("-inf")

        for child in node.children:

            score = self._uct_score(
                child,
                node,
            )

            if score > best_score:
                best_score = score
                best_child = child

        return best_child

    def _uct_score(
        self,
        child: Node,
        parent: Node,
    ) -> float:

        if child.visits == 0:
            return float("inf")

        exploitation = (
            child.total_reward / child.visits
        )

        exploration = (
            self.exploration_constant
            * math.sqrt(
                math.log(parent.visits)
                / child.visits
            )
        )

        return exploitation + exploration