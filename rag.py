from retriever import Retriever
from generator import Generator


class RAG:
    def __init__(
        self,
        retriever: Retriever,
        generator: Generator,
        top_k: int = 3,
    ):
        self.retriever = retriever
        self.generator = generator
        self.top_k = top_k

    def run(self, question: str):

        documents = self.retriever.retrieve(
            question,
            top_k=self.top_k,
        )

        answer = self.generator.generate(
            question,
            documents,
        )

        return {
            "question": question,
            "documents": documents,
            "answer": answer,
        }