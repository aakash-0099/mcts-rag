from .action import Action
from .state import ActionState


ANSWER_ACTIONS = {
    Action.A1_DIRECT_ANSWER,
    Action.A6_SUMMARIZED_ANSWER,
}


def is_terminal_action(action: Action) -> bool:
    """
    Return whether an action is capable of producing
    a terminal candidate answer.
    """

    return action in ANSWER_ACTIONS


def is_terminal_state(state: ActionState) -> bool:
    """
    Return whether the state represents a completed candidate answer.
    """

    return state.is_terminal