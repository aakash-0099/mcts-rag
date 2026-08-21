import pytest

from actions import (
    Action,
    ActionState,
    InvalidActionError,
    execute_action,
    get_available_actions,
)


def make_state() -> ActionState:
    return ActionState(
        question="What is 2 + 2?"
    )


def test_action_enum_contains_all_six_actions():
    actions = list(Action)

    assert len(actions) == 6

    assert Action.A1_DIRECT_ANSWER in actions
    assert Action.A2_QUICK_REASONING in actions
    assert Action.A3_DECOMPOSE_QUESTION in actions
    assert Action.A4_RETRIEVAL_REASONING in actions
    assert Action.A5_RETRIEVAL_DECOMPOSE in actions
    assert Action.A6_SUMMARIZED_ANSWER in actions


def test_action_values():
    assert Action.A1_DIRECT_ANSWER.value == "A1"
    assert Action.A2_QUICK_REASONING.value == "A2"
    assert Action.A3_DECOMPOSE_QUESTION.value == "A3"
    assert Action.A4_RETRIEVAL_REASONING.value == "A4"
    assert Action.A5_RETRIEVAL_DECOMPOSE.value == "A5"
    assert Action.A6_SUMMARIZED_ANSWER.value == "A6"


def test_all_six_actions_are_available():
    state = make_state()

    actions = get_available_actions(state)

    assert actions == list(Action)


@pytest.mark.parametrize(
    "action",
    list(Action),
)
def test_each_action_produces_valid_child_state(action):
    parent_state = make_state()

    child_state = execute_action(
        action,
        parent_state,
    )

    assert isinstance(child_state, ActionState)

    # The executor must create a new state.
    assert child_state is not parent_state

    # The question must propagate to the child.
    assert child_state.question == parent_state.question

    # The child must contain some action-produced content.
    assert child_state.content != ""


@pytest.mark.parametrize(
    "action",
    list(Action),
)
def test_each_action_preserves_parent_state(action):
    parent_state = make_state()

    original_question = parent_state.question
    original_content = parent_state.content
    original_metadata = parent_state.metadata.copy()

    execute_action(
        action,
        parent_state,
    )

    # Parent state must remain untouched.
    assert parent_state.question == original_question
    assert parent_state.content == original_content
    assert parent_state.metadata == original_metadata


@pytest.mark.parametrize(
    "action",
    list(Action),
)
def test_each_action_produces_independent_state(action):
    parent_state = make_state()

    child_state = execute_action(
        action,
        parent_state,
    )

    child_state.metadata["test"] = "changed"

    assert "test" not in parent_state.metadata


def test_a1_produces_direct_answer_state():
    state = make_state()

    child = execute_action(
        Action.A1_DIRECT_ANSWER,
        state,
    )

    assert "Direct answer" in child.content


def test_a2_produces_quick_reasoning_state():
    state = make_state()

    child = execute_action(
        Action.A2_QUICK_REASONING,
        state,
    )

    assert "Quick reasoning" in child.content


def test_a3_produces_decomposition_state():
    state = make_state()

    child = execute_action(
        Action.A3_DECOMPOSE_QUESTION,
        state,
    )

    assert "Decomposed question" in child.content


def test_a4_produces_retrieval_reasoning_state():
    state = make_state()

    child = execute_action(
        Action.A4_RETRIEVAL_REASONING,
        state,
    )

    assert "Retrieval reasoning" in child.content


def test_a5_produces_retrieval_decomposition_state():
    state = make_state()

    child = execute_action(
        Action.A5_RETRIEVAL_DECOMPOSE,
        state,
    )

    assert "Retrieval + decomposition" in child.content


def test_a6_produces_summary_state():
    state = make_state()

    child = execute_action(
        Action.A6_SUMMARIZED_ANSWER,
        state,
    )

    assert "Summarized answer" in child.content


def test_invalid_string_action_is_rejected():
    state = make_state()

    with pytest.raises(InvalidActionError):
        execute_action(
            "A99",
            state,
        )


def test_invalid_integer_action_is_rejected():
    state = make_state()

    with pytest.raises(InvalidActionError):
        execute_action(
            123,
            state,
        )


def test_a4_does_not_perform_retrieval():
    state = make_state()

    child = execute_action(
        Action.A4_RETRIEVAL_REASONING,
        state,
    )

    assert child.metadata == {}


def test_a5_does_not_perform_retrieval():
    state = make_state()

    child = execute_action(
        Action.A5_RETRIEVAL_DECOMPOSE,
        state,
    )

    assert child.metadata == {}