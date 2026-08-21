from __future__ import annotations

from dataclasses import dataclass

from actions.candidates import (
    CandidateAnswer,
    CandidateAnswerCollector,
)

from mcts.mcts import MCTS
from mcts.node import Node


@dataclass
class ActionSearchResult:
    """
    Result of an action-driven MCTS search.

    Multiple candidate answers may be produced.
    """

    best_node: Node
    candidates: list[CandidateAnswer]

    @property
    def answers(self) -> list[str]:
        return [
            candidate.answer
            for candidate in self.candidates
        ]


class ActionMCTS:
    """
    Paper-specific wrapper around generic MCTS.

    MCTS itself remains unaware of candidate answers.
    """

    def __init__(self, mcts: MCTS):
        self.mcts = mcts

    def search(
        self,
        root: Node,
    ) -> ActionSearchResult:

        best_node = self.mcts.search(root)

        collector = CandidateAnswerCollector()

        self._collect_terminal_nodes(
            root,
            collector,
        )

        return ActionSearchResult(
            best_node=best_node,
            candidates=collector.candidates,
        )

    def _collect_terminal_nodes(
        self,
        node: Node,
        collector: CandidateAnswerCollector,
    ) -> None:

        if node.is_terminal:
            collector.add(
                node,
                node.value,
            )

        for child in node.children:
            self._collect_terminal_nodes(
                child,
                collector,
            )