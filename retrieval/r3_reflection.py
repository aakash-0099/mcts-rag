from abc import ABC, abstractmethod


class KnowledgeReflector(ABC):

    @abstractmethod
    def reflect(self, query: str, results):
        raise NotImplementedError


class DefaultKnowledgeReflector(KnowledgeReflector):

    def reflect(self, query: str, results):
        if not isinstance(query, str):
            raise TypeError("query must be a string")

        if results is None:
            raise ValueError("results cannot be None")

        if not results:
            return {
                "useful": False,
                "reason": "No relevant knowledge was retrieved.",
                "results": [],
            }

        useful_results = [
            result
            for result in results
            if result.get("score", 0) > 0
        ]

        return {
            "useful": bool(useful_results),
            "reason": (
                "Relevant knowledge was retrieved."
                if useful_results
                else "Retrieved knowledge was not relevant."
            ),
            "results": useful_results,
        }