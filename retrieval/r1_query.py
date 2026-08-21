from abc import ABC, abstractmethod


class QueryGenerator(ABC):

    @abstractmethod
    def generate(self, question: str) -> str:
        raise NotImplementedError


class DefaultQueryGenerator(QueryGenerator):

    def generate(self, question: str) -> str:
        if not isinstance(question, str):
            raise TypeError("question must be a string")

        query = question.strip()

        if not query:
            raise ValueError("question cannot be empty")

        return query