from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from actions.action import Action
from actions.answer_selection import (
    AnswerSelectionResult,
    AnswerSelector,
)
from actions.candidates import (
    CandidateAnswer,
    CandidateAnswerCollector,
)
from actions.evaluator import TrajectoryEvaluator
from actions.integration import (
    ActionExpansion,
    ActionSimulation,
)
from actions.llm import LLM
from actions.state import ActionState
from mcts.mcts import MCTS
from mcts.node import Node
from retrieval import RetrievalPipeline, RetrievalStrategy


@dataclass
class MCTSRAGResult:
    question: str
    final_answer: str | None
    best_candidate: CandidateAnswer | None
    best_trajectory: list[tuple[Action | None, ActionState]]
    candidates: list[CandidateAnswer]
    selection: AnswerSelectionResult
    root: Node
    tree: str


class MCTSRAG:

    def __init__(
        self,
        num_simulations: int = 100,
        retrieval_strategy: RetrievalStrategy | None = None,
        evaluator: Callable | None = None,
        answer_selector: AnswerSelector | None = None,
        llm: LLM | None = None,
    ):
        self.num_simulations = num_simulations

        self.retrieval_strategy = (
            retrieval_strategy
            or RetrievalPipeline()
        )

        self.llm = llm or LLM()

        # Default to the grounding-based evaluator instead of
        # leaving reward at a flat 0.0 for every node.
        self.evaluator = (
            evaluator
            or TrajectoryEvaluator(llm=self.llm)
        )

        self.answer_selector = (
            answer_selector
            or AnswerSelector()
        )

    def run(
        self,
        question: str,
    ) -> MCTSRAGResult:

        if not question or not question.strip():
            raise ValueError(
                "Question must not be empty."
            )

        root_state = ActionState(question=question)
        root = Node(state=root_state)

        expansion = ActionExpansion(
            retrieval_strategy=self.retrieval_strategy,
            llm=self.llm,
        )

        simulation = ActionSimulation(
            evaluator=self.evaluator,
        )

        mcts = MCTS(
            expansion=expansion,
            simulation=simulation,
            num_simulations=self.num_simulations,
        )

        print("\n" + "=" * 60)
        print(f"MCTS START: {question}")
        print("=" * 60)

        mcts.search(root)

        print("\nMCTS SEARCH COMPLETE")
        print(self.visualize_tree(root))

        candidates = (
            CandidateAnswerCollector
            .collect_from_tree(root)
        )

        selection = self.answer_selector.select(candidates)

        best_candidate = None

        if selection.best_cluster is not None:
            best_candidate = selection.best_cluster.best_candidate

        best_trajectory = []

        if best_candidate is not None:
            best_trajectory = best_candidate.trajectory

        return MCTSRAGResult(
            question=question,
            final_answer=selection.final_answer,
            best_candidate=best_candidate,
            best_trajectory=best_trajectory,
            candidates=candidates,
            selection=selection,
            root=root,
            tree=self.visualize_tree(root),
        )

    @staticmethod
    def visualize_tree(root: Node) -> str:
        lines: list[str] = []

        def visit(node: Node, prefix: str, is_last: bool) -> None:

            if node.parent is None:
                connector = ""
            else:
                connector = "└── " if is_last else "├── "

            action = node.action
            action_text = "ROOT" if action is None else action.value
            terminal = " [terminal]" if node.is_terminal else ""

            lines.append(
                f"{prefix}{connector}"
                f"{action_text}"
                f" visits={node.visits}"
                f" value={node.value:.3f}"
                f"{terminal}"
            )

            child_prefix = prefix

            if node.parent is not None:
                child_prefix += "    " if is_last else "│   "

            for index, child in enumerate(node.children):
                visit(
                    child,
                    child_prefix,
                    index == len(node.children) - 1,
                )

        visit(root, "", True)

        return "\n".join(lines)