from actions.action import Action
from actions.answer_selection import AnswerSelector
from actions.candidates import CandidateAnswer


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


def test_best_answer_cluster_is_selected():

    candidates = [
        make_candidate("Paris"),
        make_candidate("Paris"),
        make_candidate("Paris"),
        make_candidate("London"),
    ]

    selector = AnswerSelector()

    result = selector.select(candidates)

    assert len(result.clusters) == 2

    assert result.best_cluster is not None

    assert result.best_cluster.size == 3

    assert result.final_answer == "Paris"


def test_single_candidate_is_selected():

    candidates = [
        make_candidate("MCTS is tree search."),
    ]

    selector = AnswerSelector()

    result = selector.select(candidates)

    assert len(result.clusters) == 1

    assert result.final_answer == (
        "MCTS is tree search."
    )


def test_empty_candidates_produce_no_answer():

    selector = AnswerSelector()

    result = selector.select([])

    assert result.clusters == []
    assert result.scores == []
    assert result.best_cluster is None
    assert result.final_answer is None


def test_candidate_reward_breaks_tie_inside_cluster():

    candidates = [
        make_candidate("Answer A", reward=0.5),
        make_candidate("Answer A", reward=0.9),
        make_candidate("Answer B", reward=0.2),
    ]

    selector = AnswerSelector()

    result = selector.select(candidates)

    assert result.final_answer == "Answer A"

    assert (
        result.best_cluster.best_candidate.reward
        == 0.9
    )