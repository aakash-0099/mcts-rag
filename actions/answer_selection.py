from __future__ import annotations

from dataclasses import dataclass

from .candidates import CandidateAnswer
from .clustering import (
    AnswerCluster,
    AnswerClusterer,
)
from .reward import (
    AnswerClusterReward,
    ClusterScore,
)


@dataclass
class AnswerSelectionResult:
    """
    Complete result of Phase 10 answer clustering.
    """

    clusters: list[AnswerCluster]

    scores: list[ClusterScore]

    best_cluster: AnswerCluster | None

    final_answer: str | None


class AnswerSelector:
    """
    Clusters candidate answers and selects the
    highest-supported answer cluster.
    """

    def __init__(
        self,
        clusterer: AnswerClusterer | None = None,
        reward: AnswerClusterReward | None = None,
    ):
        self.clusterer = (
            clusterer
            or AnswerClusterer()
        )

        self.reward = (
            reward
            or AnswerClusterReward()
        )

    def select(
        self,
        candidates: list[CandidateAnswer],
    ) -> AnswerSelectionResult:

        if not candidates:
            return AnswerSelectionResult(
                clusters=[],
                scores=[],
                best_cluster=None,
                final_answer=None,
            )

        clusters = self.clusterer.cluster(
            candidates
        )

        scores = self.reward.score(
            clusters
        )

        best_score = max(
            scores,
            key=lambda score: score.reward,
        )

        best_cluster = best_score.cluster

        final_answer = (
            best_cluster.best_candidate.answer
        )

        return AnswerSelectionResult(
            clusters=clusters,
            scores=scores,
            best_cluster=best_cluster,
            final_answer=final_answer,
        )