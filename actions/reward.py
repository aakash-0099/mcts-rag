from __future__ import annotations

from dataclasses import dataclass

from .clustering import AnswerCluster


@dataclass(frozen=True)
class ClusterScore:
    cluster: AnswerCluster
    reward: float


class AnswerClusterReward:
    """
    Calculates the reward of an answer cluster.

    Reward is the cluster's share of total *weighted* support,
    where each candidate's vote is weighted by its own MCTS
    reward (how grounded/correct it was judged to be), not just
    counted as one vote regardless of quality.

    A cluster of one high-reward, well-grounded candidate should
    outweigh a cluster of one low-reward, ungrounded candidate,
    even though both have the same raw size.
    """

    def calculate(
        self,
        cluster: AnswerCluster,
        total_weighted_support: float,
    ) -> float:

        if total_weighted_support <= 0:
            return 0.0

        cluster_support = sum(
            candidate.reward
            for candidate in cluster.candidates
        )

        return cluster_support / total_weighted_support

    def score(
        self,
        clusters: list[AnswerCluster],
    ) -> list[ClusterScore]:

        total_weighted_support = sum(
            candidate.reward
            for cluster in clusters
            for candidate in cluster.candidates
        )

        return [
            ClusterScore(
                cluster=cluster,
                reward=self.calculate(
                    cluster,
                    total_weighted_support,
                ),
            )
            for cluster in clusters
        ]