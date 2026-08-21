from pathlib import Path
from typing import List, Dict

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class Retriever:
    def __init__(self, documents: List[str]):
        self.documents = documents

        self.vectorizer = TfidfVectorizer(
            stop_words="english"
        )

        self.document_vectors = self.vectorizer.fit_transform(
            documents
        )

    @classmethod
    def from_file(
        cls,
        path: str,
        chunk_size: int = 500,
    ):
        text = Path(path).read_text(
            encoding="utf-8"
        )

        documents = [
            text[i:i + chunk_size]
            for i in range(
                0,
                len(text),
                chunk_size,
            )
        ]

        return cls(documents)

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
    ) -> List[Dict]:

        query_vector = self.vectorizer.transform(
            [query]
        )

        scores = cosine_similarity(
            query_vector,
            self.document_vectors,
        )[0]

        ranked_indices = scores.argsort()[::-1]

        results = []

        for index in ranked_indices[:top_k]:
            results.append(
                {
                    "document": self.documents[index],
                    "score": float(scores[index]),
                    "index": int(index),
                }
            )

        return results