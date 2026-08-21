from abc import ABC, abstractmethod


class RetrievalStrategy(ABC):

    @abstractmethod
    def retrieve(self, query: str):
        """
        Execute the retrieval workflow for a query.

        Returns:
            RetrievalResult
        """
        raise NotImplementedError