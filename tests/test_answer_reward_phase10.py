from actions.action import Action
from actions.candidates import CandidateAnswer
from actions.clustering import AnswerCluster
from actions.reward import AnswerClusterReward


def make_candidate(answer: str) -> CandidateAnswer:

    return CandidateAnswer(
        answer=answer,
        reward=1.0,
        actions=[
            Action.A1_DIRECT_ANSWER
        ],
        states=[],
    )


def test_cluster_reward_is_relative_support():

    cluster = AnswerCluster(
        candidates=[
            make_candidate("A"),
            make_candidate("A"),
            make_candidate("A"),
        ]
    )

    reward = AnswerClusterReward()

    score = reward.calculate(
        cluster,
        total_candidates=5,
    )

    assert score == 0.6


def test_singleton_cluster_reward():

    cluster = AnswerCluster(
        candidates=[
            make_candidate("B"),
        ]
    )

    reward = AnswerClusterReward()

    score = reward.calculate(
        cluster,
        total_candidates=5,
    )

    assert score == 0.2


def test_cluster_scores_sum_to_one():

    reward = AnswerClusterReward()

    clusters = [
        AnswerCluster(
            candidates=[
                make_candidate("A"),
                make_candidate("A"),
            ]
        ),
        AnswerCluster(
            candidates=[
                make_candidate("B"),
            ]
        ),
        AnswerCluster(
            candidates=[
                make_candidate("C"),
            ]
        ),
    ]

    scores = reward.score(clusters)

    total = sum(
        score.reward
        for score in scores
    )

    assert total == 1.0


def test_best_cluster_has_highest_support():

    reward = AnswerClusterReward()

    clusters = [
        AnswerCluster(
            candidates=[
                make_candidate("A"),
                make_candidate("A"),
                make_candidate("A"),
            ]
        ),
        AnswerCluster(
            candidates=[
                make_candidate("B"),
            ]
        ),
    ]

    scores = reward.score(clusters)

    best = max(
        scores,
        key=lambda score: score.reward,
    )

    assert best.cluster.answers == [
        "A",
        "A",
        "A",
    ]

    assert best.reward == 0.75