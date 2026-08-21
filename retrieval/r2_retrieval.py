from abc import ABC, abstractmethod


class Retriever(ABC):

    @abstractmethod
    def retrieve(self, query: str):
        raise NotImplementedError


class DefaultRetriever(Retriever):

    def __init__(self, documents=None):
        self.documents = documents or []

    def retrieve(self, query: str):
        if not isinstance(query, str):
            raise TypeError("query must be a string")

        query = query.strip()

        if not query:
            raise ValueError("query cannot be empty")

        query_terms = set(query.lower().split())

        results = []

        for document in self.documents:
            if isinstance(document, str):
                text = document
            else:
                text = str(document)

            document_terms = set(text.lower().split())

            score = len(query_terms.intersection(document_terms))

            if score > 0:
                results.append(
                    {
                        "document": document,
                        "score": score,
                    }
                )

        results.sort(
            key=lambda result: result["score"],
            reverse=True,
        )

        return results