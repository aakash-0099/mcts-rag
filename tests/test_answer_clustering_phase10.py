from actions.action import Action
from actions.candidates import CandidateAnswer
from actions.clustering import (
    AnswerClusterer,
    NormalizedAnswerSimilarity,
)


def make_candidate(
    answer: str,
    reward: float = 1.0,
) -> CandidateAnswer:

    return CandidateAnswer(
        answer=answer,
        reward=reward,
        actions=[
            Action.A1_DIRECT_ANSWER
        ],
        states=[],
    )


def test_identical_answers_are_clustered():

    candidates = [
        make_candidate("Paris is the capital of France."),
        make_candidate("Paris is the capital of France."),
        make_candidate("Paris is the capital of France."),
    ]

    clusterer = AnswerClusterer()

    clusters = clusterer.cluster(candidates)

    assert len(clusters) == 1
    assert clusters[0].size == 3


def test_normalized_equivalent_answers_are_clustered():

    candidates = [
        make_candidate("Paris is the capital of France."),
        make_candidate("  paris is the capital of france! "),
    ]

    clusterer = AnswerClusterer()

    clusters = clusterer.cluster(candidates)

    assert len(clusters) == 1
    assert clusters[0].size == 2


def test_different_answers_create_different_clusters():

    candidates = [
        make_candidate("Paris"),
        make_candidate("London"),
        make_candidate("Berlin"),
    ]

    clusterer = AnswerClusterer()

    clusters = clusterer.cluster(candidates)

    assert len(clusters) == 3

    assert [
        cluster.size
        for cluster in clusters
    ] == [1, 1, 1]


def test_multiple_answer_clusters():

    candidates = [
        make_candidate("Paris"),
        make_candidate("Paris"),
        make_candidate("London"),
        make_candidate("Paris"),
        make_candidate("London"),
    ]

    clusterer = AnswerClusterer()

    clusters = clusterer.cluster(candidates)

    assert len(clusters) == 2

    assert [
        cluster.size
        for cluster in clusters
    ] == [3, 2]


def test_similarity_is_deterministic():

    similarity = NormalizedAnswerSimilarity()

    assert similarity.are_equivalent(
        "Answer",
        "answer",
    )

    assert not similarity.are_equivalent(
        "Answer A",
        "Answer B",
    )