from .node import Node
from .selection import UCTSelection
from .expansion import ExpansionStrategy
from .simulation import SimulationStrategy
from .backpropagation import Backpropagation


class MCTS:

    def __init__(
        self,
        expansion: ExpansionStrategy,
        simulation: SimulationStrategy,
        num_simulations: int = 100,
        selection=None,
        backpropagation=None,
    ):
        self.num_simulations = num_simulations

        self.selection = selection or UCTSelection()
        self.expansion = expansion
        self.simulation = simulation
        self.backpropagation = backpropagation or Backpropagation()

    def search(self, root: Node) -> Node:

        if root.is_terminal or (
            root.is_fully_expanded and not root.children
        ):
            return root

        for simulation_number in range(self.num_simulations):

            print(
                f"\n[MCTS] Simulation "
                f"{simulation_number + 1}/{self.num_simulations}"
            )

            node = self.selection.select(root)

            print(
                f"[MCTS] Selected: "
                f"action={node.action} "
                f"visits={node.visits} "
                f"value={node.value:.3f}"
            )
            # print(
            #     "DEBUG MCTS:",
            #     "action=", getattr(node.action, "value", node.action),
            #     "state_type=", type(node.state).__name__,
            #     "state=", repr(node.state)[:300],
            # )

            if node.is_terminal:
                # Revisiting an already-known terminal leaf is
                # normal MCTS behavior — just reinforce its
                # existing value up the tree. Do NOT expand it
                # (nothing to expand) and do NOT abort the whole
                # search over it.
                self.backpropagation.backpropagate(
                    node,
                    node.value,
                )
                continue

            child = self.expansion.expand(node)

            print(
                f"[MCTS] Expanded: "
                f"action={child.action} "
                f"state_type={type(child.state).__name__}"
            )
            if child is node and not child.children:
                # node had no untried actions and no children —
                # a true dead end, not just a terminal revisit.
                # Skip this iteration rather than abandoning the
                # remaining simulation budget.
                continue

            reward = self.simulation.simulate(child)

            print(
                f"[MCTS] Reward: {reward:.3f}"
            )

            self.backpropagation.backpropagate(
                child,
                reward,
            )

        return self.best_child(root)

    def best_child(self, node: Node) -> Node:
        if not node.children:
            return node

        return max(
            node.children,
            key=lambda child: child.visits,
        )