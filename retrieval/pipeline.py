from dataclasses import dataclass

from .strategy import RetrievalStrategy
from .r1_query import (
    QueryGenerator,
    DefaultQueryGenerator,
)
from .r2_retrieval import (
    Retriever,
    DefaultRetriever,
)
from .r3_reflection import (
    KnowledgeReflector,
    DefaultKnowledgeReflector,
)
from .r4_summary import (
    KnowledgeSummarizer,
    DefaultKnowledgeSummarizer,
)


@dataclass
class RetrievalResult:
    query: str
    results: list
    reflection: dict
    summary: str


class RetrievalPipeline(RetrievalStrategy):

    def __init__(
        self,
        query_generator: QueryGenerator | None = None,
        retriever: Retriever | None = None,
        reflector: KnowledgeReflector | None = None,
        summarizer: KnowledgeSummarizer | None = None,
    ):
        self.query_generator = (
            query_generator or DefaultQueryGenerator()
        )

        self.retriever = (
            retriever or DefaultRetriever()
        )

        self.reflector = (
            reflector or DefaultKnowledgeReflector()
        )

        self.summarizer = (
            summarizer or DefaultKnowledgeSummarizer()
        )

    def retrieve(self, question: str) -> RetrievalResult:

        # R1 — Query generation
        query = self.query_generator.generate(question)

        # R2 — Retrieval
        results = self.retriever.retrieve(query)

        # R3 — Knowledge reflection
        reflection = self.reflector.reflect(
            query,
            results,
        )

        # R4 — Knowledge summarization
        summary = self.summarizer.summarize(
            query,
            reflection,
        )

        return RetrievalResult(
            query=query,
            results=results,
            reflection=reflection,
            summary=summary,
        )