from __future__ import annotations

from dataclasses import dataclass, field

from mcts.node import Node

from .action import Action
from .state import ActionState
from .trajectory import reconstruct_trajectory


@dataclass
class CandidateAnswer:
    """
    One candidate answer produced by a completed trajectory.

    The trajectory is optional so CandidateAnswer remains
    convenient for answer clustering/reward tests that only
    care about the answer and reward.
    """

    answer: str
    reward: float = 0.0

    actions: list[Action] = field(
        default_factory=list
    )

    states: list[ActionState] = field(
        default_factory=list
    )

    trajectory: list[
        tuple[Action | None, ActionState]
    ] = field(
        default_factory=list
    )


class CandidateAnswerCollector:
    """
    Collect candidate answers from terminal MCTS nodes.

    Public Phase 9 API:

        collector.answers
            -> list[str]

    Rich candidate objects are available through:

        collector.candidates
            -> list[CandidateAnswer]
    """

    def __init__(self):
        self.answers: list[str] = []

        self.candidates: list[
            CandidateAnswer
        ] = []

    def add(
        self,
        node_or_candidate,
        reward: float | None = None,
    ) -> CandidateAnswer:
        """
        Add a candidate and return the CandidateAnswer.

        Supported forms:

            add(CandidateAnswer(...))

        or:

            add(node, reward=0.8)

        or:

            add(node, 0.8)
        """

        # --------------------------------------------------
        # Direct CandidateAnswer
        # --------------------------------------------------

        if isinstance(
            node_or_candidate,
            CandidateAnswer,
        ):
            candidate = node_or_candidate

            self.candidates.append(candidate)
            self.answers.append(candidate.answer)

            return candidate

        # --------------------------------------------------
        # MCTS node
        # --------------------------------------------------

        node = node_or_candidate

        if not isinstance(node, Node):
            raise TypeError(
                "add() expects either a "
                "CandidateAnswer or an MCTS Node."
            )

        if not node.is_terminal:
            raise ValueError(
                "Only terminal nodes can be "
                "added as candidate answers."
            )

        if not isinstance(
            node.state,
            ActionState,
        ):
            raise TypeError(
                "Candidate node state must be "
                "an ActionState."
            )

        answer = node.state.metadata.get(
            "candidate_answer"
        )

        if not answer:
            raise ValueError(
                "Terminal node does not contain "
                "a candidate_answer."
            )

        # --------------------------------------------------
        # Reconstruct complete trajectory
        # --------------------------------------------------

        trajectory = reconstruct_trajectory(
            node
        )

        actions = [
            action
            for action, _ in trajectory
            if action is not None
        ]

        states = [
            state
            for _, state in trajectory
        ]

        # --------------------------------------------------
        # Create rich candidate
        # --------------------------------------------------

        candidate = CandidateAnswer(
            answer=str(answer),
            reward=(
                node.value
                if reward is None
                else float(reward)
            ),
            actions=actions,
            states=states,
            trajectory=trajectory,
        )

        self.candidates.append(candidate)
        self.answers.append(candidate.answer)

        return candidate

    def collect(
        self,
    ) -> list[CandidateAnswer]:
        """
        Return rich CandidateAnswer objects.
        """

        return list(self.candidates)

    def __len__(self) -> int:
        return len(self.candidates)

    def __iter__(self):
        return iter(self.candidates)

    @classmethod
    def collect_from_tree(
        cls,
        root: Node,
    ) -> list[CandidateAnswer]:
        """
        Traverse the complete MCTS tree and collect
        every terminal candidate answer.

        Terminal nodes with no usable answer (e.g. an empty
        LLM response) are skipped rather than aborting the
        whole collection.
        """

        collector = cls()

        def visit(node: Node) -> None:

            if node.is_terminal:
                try:
                    collector.add(node)
                except ValueError:
                    pass

            for child in node.children:
                visit(child)

        visit(root)

        return collector.collect()