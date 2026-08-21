from .expansion import ExpansionStrategy
from .node import Node

from actions import (
    Action,
    ActionState,
    execute_action,
    get_available_actions,
)


class PaperActionExpansion(ExpansionStrategy):
    """
    MCTS expansion strategy for the paper's six-action space.

    Each call to expand() creates exactly one child node,
    matching the generic MCTS expansion contract.
    """

    def get_untried_actions(self, node: Node) -> list[Action]:
        """
        Return paper actions that have not yet been expanded
        from this node.
        """

        if not isinstance(node.state, ActionState):
            raise TypeError(
                "PaperActionExpansion requires node.state "
                "to be an ActionState."
            )

        available_actions = get_available_actions(node.state)

        expanded_actions = {
            child.action
            for child in node.children
            if isinstance(child.action, Action)
        }

        return [
            action
            for action in available_actions
            if action not in expanded_actions
        ]

    def create_child(
        self,
        node: Node,
        action: Action,
    ) -> Node:
        """
        Execute an action and create the corresponding MCTS child.
        """

        child_state = execute_action(
            action,
            node.state,
        )

        child = Node(
            state=child_state,
            parent=node,
            action=action,
        )

        return child

    def expand(self, node: Node) -> Node:
        """
        Expand exactly one untried paper action.

        Actions are currently expanded deterministically in enum order:
            A1 -> A2 -> A3 -> A4 -> A5 -> A6
        """

        actions = self.get_untried_actions(node)

        if not actions:
            return node

        action = actions[0]

        child = self.create_child(
            node,
            action,
        )

        node.add_child(child)

        return child