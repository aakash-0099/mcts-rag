from .node import Node


class Backpropagation:

    def backpropagate(self, node: Node, reward: float) -> None:
        current = node

        while current is not None:
            current.visits += 1
            current.total_reward += reward

            current = current.parent