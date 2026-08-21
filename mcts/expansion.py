from abc import ABC, abstractmethod

from .node import Node


class ExpansionStrategy(ABC):

    @abstractmethod
    def expand(self, node: Node) -> Node:
        pass


class DefaultExpansion(ExpansionStrategy):

    def expand(self, node: Node) -> Node:
        actions = self.get_untried_actions(node)

        if not actions:
            return node

        action = actions[0]

        child = self.create_child(node, action)

        node.add_child(child)

        return child

    def get_untried_actions(self, node: Node):
        raise NotImplementedError

    def create_child(self, node: Node, action):
        raise NotImplementedError