from __future__ import annotations

import re

from .action import Action
from .integration import ActionTrajectory
from .llm import LLM


GROUNDING_PROMPT = """You are grading whether an answer is supported by the provided context.

Context (retrieved knowledge accumulated during reasoning; may be empty):
{context}

Question:
{question}

Answer to grade:
{answer}

Respond with ONLY a single number between 0 and 1:
- 1.0 if the answer is fully and specifically supported by the context above
- 0.5 if the answer is plausible but only partially supported by the context
- 0.0 if the answer contradicts the context, or if it invents specific facts (names, numbers, organizations) that do not appear anywhere in the context

If the context is empty, judge only whether the answer sounds like a specific,
confident factual claim (score it low, e.g. 0.1) versus an appropriately
hedged "I don't have enough information" answer (score it 0.5, since that is
the honest response when nothing was retrieved).

Respond with only the number, nothing else.
"""


class TrajectoryEvaluator:
    """
    Minimal Phase 14 evaluator.

    Scores a trajectory (root -> current node) by checking whether
    the accumulated content is actually grounded in retrieval
    output, rather than trusting every terminal answer equally.

    This directly fixes the problem where A1/A6 branches that
    never call retrieval can "win" against A4/A5 branches purely
    by chance, since previously every reward was a flat 0.0.
    """

    def __init__(self, llm: LLM | None = None):
        self.llm = llm or LLM()

    def __call__(self, trajectory: ActionTrajectory) -> float:

        if not trajectory.states:
            return 0.0

        final_state = trajectory.current_state

        context = self._collect_retrieved_context(trajectory)

        if not final_state.is_terminal:
            # Not an answer yet. Give a small shaped reward so
            # branches that have already retrieved something are
            # preferred for further expansion over ones that
            # haven't, even before either reaches a final answer.
            return 0.6 if context else 0.2

        answer = final_state.candidate_answer

        if not answer:
            return 0.0

        prompt = GROUNDING_PROMPT.format(
            context=context or "(no retrieval was performed on this reasoning path)",
            question=final_state.question,
            answer=answer,
        )

        raw = self.llm.complete(prompt)

        return self._parse_score(raw)

    @staticmethod
    def _collect_retrieved_context(
        trajectory: ActionTrajectory,
    ) -> str:
        """
        Concatenate content from any A4/A5 (retrieval) nodes
        along this trajectory, so the answer can be checked
        against what was actually retrieved rather than the
        model's own prior knowledge.
        """

        pieces: list[str] = []

        # states includes the root state first; actions excludes
        # the root's None action, so pad with None to keep them
        # aligned index-for-index.
        for action, state in zip(
            [None] + trajectory.actions,
            trajectory.states,
        ):
            if action in (
                Action.A4_RETRIEVAL_REASONING,
                Action.A5_RETRIEVAL_DECOMPOSE,
            ):
                if state.content:
                    pieces.append(state.content)

        return "\n\n".join(pieces)

    @staticmethod
    def _parse_score(raw: str) -> float:

        match = re.search(r"(\d+(\.\d+)?)", raw.strip())

        if not match:
            return 0.0

        score = float(match.group(1))

        return max(0.0, min(1.0, score))