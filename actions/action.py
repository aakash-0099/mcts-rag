from enum import Enum


class Action(Enum):
    """
    The six actions defined by the paper.
    """

    A1_DIRECT_ANSWER = "A1"
    A2_QUICK_REASONING = "A2"
    A3_DECOMPOSE_QUESTION = "A3"
    A4_RETRIEVAL_REASONING = "A4"
    A5_RETRIEVAL_DECOMPOSE = "A5"
    A6_SUMMARIZED_ANSWER = "A6"