from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class PaperSearchState:
    """
    Immutable state representing one paper-search trajectory.

    A child state inherits all information from its parent while keeping
    its own independent trajectory.

    Evidence is cumulative along a path.
    """

    question: str

    action_history: tuple[Any, ...] = field(default_factory=tuple)
    reasoning_history: tuple[str, ...] = field(default_factory=tuple)
    sub_questions: tuple[str, ...] = field(default_factory=tuple)
    retrieved_knowledge: tuple[Any, ...] = field(default_factory=tuple)
    summaries: tuple[str, ...] = field(default_factory=tuple)

    candidate_answer: str | None = None

    def child(
        self,
        *,
        action: Any | None = None,
        reasoning: str | None = None,
        sub_questions: list[str] | tuple[str, ...] | None = None,
        retrieved_knowledge: list[Any] | tuple[Any, ...] | None = None,
        summary: str | None = None,
        candidate_answer: str | None = None,
    ) -> "PaperSearchState":
        """
        Create an independent child state from this state.

        Existing trajectory information is preserved.
        New information is appended only to the child.

        The parent state is never modified.
        """

        return PaperSearchState(
            question=self.question,

            action_history=(
                self.action_history
                + ((action,) if action is not None else ())
            ),

            reasoning_history=(
                self.reasoning_history
                + ((reasoning,) if reasoning is not None else ())
            ),

            sub_questions=(
                self.sub_questions
                + tuple(sub_questions or ())
            ),

            retrieved_knowledge=(
                self.retrieved_knowledge
                + tuple(retrieved_knowledge or ())
            ),

            summaries=(
                self.summaries
                + ((summary,) if summary is not None else ())
            ),

            candidate_answer=(
                self.candidate_answer
                if candidate_answer is None
                else candidate_answer
            ),
        )