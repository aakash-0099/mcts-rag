from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import re

from .candidates import CandidateAnswer


@dataclass
class AnswerCluster:
    candidates: list[CandidateAnswer] = field(default_factory=list)

    @property
    def size(self) -> int:
        return len(self.candidates)

    @property
    def answers(self) -> list[str]:
        return [c.answer for c in self.candidates]

    @property
    def best_candidate(self) -> CandidateAnswer:
        if not self.candidates:
            raise ValueError(
                "Cannot select a candidate from an empty cluster."
            )
        return max(self.candidates, key=lambda c: c.reward)


class AnswerSimilarityStrategy(ABC):
    @abstractmethod
    def are_equivalent(self, answer_a: str, answer_b: str) -> bool:
        pass


class NormalizedAnswerSimilarity(AnswerSimilarityStrategy):
    """Exact-match baseline. Kept for tests that rely on it."""

    @staticmethod
    def normalize(answer: str) -> str:
        answer = answer.lower().strip()
        answer = re.sub(r"\s+", " ", answer)
        answer = re.sub(r"[^\w\s]", "", answer)
        return answer

    def are_equivalent(self, answer_a: str, answer_b: str) -> bool:
        return self.normalize(answer_a) == self.normalize(answer_b)


class TokenOverlapSimilarity(AnswerSimilarityStrategy):
    """
    Fuzzy clustering by token (word) overlap rather than exact
    string match, so two correct answers phrased differently
    ("reports directly to CEO Marcus Whitfield" vs "reports
    directly to Marcus Whitfield") still cluster together, instead
    of each fragmenting into its own size-1 cluster and losing to
    a larger cluster of near-duplicate hedges.

    Uses Jaccard similarity over normalized word sets, ignoring
    common stopwords/hedge filler so two answers built from
    genuinely different facts don't falsely merge.
    """

    _STOPWORDS = {
        "a", "an", "the", "to", "of", "in", "is", "are", "and",
        "or", "but", "i", "dont", "do", "not", "have", "any",
        "information", "about", "please", "provide", "more",
        "context", "unable", "based", "provided", "reasoning",
        "knowledge", "final", "answer", "can", "you",
    }

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    @staticmethod
    def _tokenize(answer: str) -> set[str]:
        text = answer.lower()
        text = re.sub(r"[^\w\s]", "", text)
        tokens = set(text.split())
        return tokens - TokenOverlapSimilarity._STOPWORDS

    def are_equivalent(self, answer_a: str, answer_b: str) -> bool:

        tokens_a = self._tokenize(answer_a)
        tokens_b = self._tokenize(answer_b)

        if not tokens_a or not tokens_b:
            return tokens_a == tokens_b

        intersection = tokens_a & tokens_b
        union = tokens_a | tokens_b

        jaccard = len(intersection) / len(union)

        return jaccard >= self.threshold


class AnswerClusterer:

    def __init__(
        self,
        similarity: AnswerSimilarityStrategy | None = None,
    ):
        self.similarity = (
            similarity or TokenOverlapSimilarity()
        )

    def cluster(
        self,
        candidates: list[CandidateAnswer],
    ) -> list[AnswerCluster]:

        clusters: list[AnswerCluster] = []

        for candidate in candidates:

            matched_cluster = None

            for cluster in clusters:
                representative = cluster.candidates[0].answer

                if self.similarity.are_equivalent(
                    candidate.answer, representative
                ):
                    matched_cluster = cluster
                    break

            if matched_cluster is None:
                clusters.append(
                    AnswerCluster(candidates=[candidate])
                )
            else:
                matched_cluster.candidates.append(candidate)

        return clusters