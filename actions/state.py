from dataclasses import dataclass, field
from typing import Any


@dataclass
class ActionState:
    """
    State used by the paper-specific action layer.

    The state is immutable from the perspective of branches:
    child() creates a new ActionState.

    Phase 9 adds explicit terminal/candidate-answer information.
    """

    question: str
    content: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def child(
        self,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> "ActionState":
        """
        Create a new state without modifying the current state.

        Metadata is copied so child branches do not share
        mutable state with the parent.
        """

        child_metadata = dict(self.metadata)

        if metadata is not None:
            child_metadata.update(metadata)

        return ActionState(
            question=self.question,
            content=content,
            metadata=child_metadata,
        )

    @property
    def is_terminal(self) -> bool:
        """
        Whether this state represents a completed answer-producing state.
        """

        return bool(
            self.metadata.get("terminal", False)
        )

    @property
    def candidate_answer(self) -> str | None:
        """
        Candidate answer produced by a terminal action.
        """

        answer = self.metadata.get(
            "candidate_answer"
        )

        if answer is None:
            return None

        return str(answer)