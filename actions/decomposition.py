from abc import ABC, abstractmethod


class DecompositionStrategy(ABC):

    @abstractmethod
    def decompose(self, question: str):
        raise NotImplementedError


class DefaultDecomposition(DecompositionStrategy):

    def decompose(self, question: str):
        if not isinstance(question, str):
            raise TypeError("question must be a string")

        question = question.strip()

        if not question:
            raise ValueError("question cannot be empty")

        return [question]