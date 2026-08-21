from collections.abc import Callable

from .action import Action
from .executors import (
    execute_a1,
    execute_a2,
    execute_a3,
    execute_a4,
    execute_a5,
    execute_a6,
)
from .state import ActionState


class InvalidActionError(ValueError):
    """
    Raised when an invalid or unsupported action is requested.
    """


ActionExecutor = Callable[[ActionState], ActionState]


_ACTION_EXECUTORS: dict[Action, ActionExecutor] = {
    Action.A1_DIRECT_ANSWER: execute_a1,
    Action.A2_QUICK_REASONING: execute_a2,
    Action.A3_DECOMPOSE_QUESTION: execute_a3,
    Action.A4_RETRIEVAL_REASONING: execute_a4,
    Action.A5_RETRIEVAL_DECOMPOSE: execute_a5,
    Action.A6_SUMMARIZED_ANSWER: execute_a6,
}


def execute_action(
    action: Action,
    state: ActionState,
    llm=None,
) -> ActionState:
    """
    Execute one of the paper's six actions.
    """

    if not isinstance(action, Action):
        raise InvalidActionError(
            f"Invalid action: {action!r}. "
            f"Expected an Action enum member."
        )

    executor = _ACTION_EXECUTORS.get(action)

    if executor is None:
        raise InvalidActionError(
            f"No executor registered for action: {action}"
        )

    child_state = executor(state, llm=llm)

    if not isinstance(child_state, ActionState):
        raise TypeError(
            f"Executor for {action.value} returned "
            f"{type(child_state).__name__}, "
            f"expected ActionState."
        )

    return child_state


def get_available_actions(
    state: ActionState,
) -> list[Action]:
    """
    Return the actions available from the current state.

    Phase 5 has no action masking, so all six actions are
    currently available.
    """

    return list(Action)