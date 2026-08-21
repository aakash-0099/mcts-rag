from abc import ABC, abstractmethod


class KnowledgeSummarizer(ABC):

    @abstractmethod
    def summarize(self, query: str, reflection):
        raise NotImplementedError


class DefaultKnowledgeSummarizer(KnowledgeSummarizer):

    def summarize(self, query: str, reflection):
        if not isinstance(query, str):
            raise TypeError("query must be a string")

        if reflection is None:
            raise ValueError("reflection cannot be None")

        results = reflection.get("results", [])

        if not results:
            return ""

        summaries = []

        for result in results:
            document = result.get("document")

            if document is not None:
                summaries.append(str(document))

        return "\n".join(summaries)