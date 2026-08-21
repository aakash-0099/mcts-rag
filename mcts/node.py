from typing import Any, Optional


class Node:
    def __init__(
        self,
        state: Any,
        parent: Optional["Node"] = None,
        action: Any = None,
    ):
        self.state = state
        self.parent = parent
        self.action = action

        self.children: list[Node] = []

        self.visits = 0
        self.total_reward = 0.0

        self._fully_expanded = False

    @property
    def is_terminal(self) -> bool:
        return bool(
            getattr(self.state, "is_terminal", False)
        )

    @property
    def is_fully_expanded(self) -> bool:
        return self._fully_expanded

    def mark_fully_expanded(self) -> None:
        self._fully_expanded = True

    @property
    def value(self) -> float:
        if self.visits == 0:
            return 0.0

        return self.total_reward / self.visits

    def add_child(self, child: "Node") -> None:
        self.children.append(child)

    def __repr__(self) -> str:
        return (
            f"Node("
            f"state={self.state!r}, "
            f"visits={self.visits}, "
            f"value={self.value:.3f}"
            f")"
        )