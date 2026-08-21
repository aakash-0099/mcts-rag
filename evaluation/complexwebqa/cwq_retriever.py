import json
import math
from pathlib import Path


class CWQRetriever:

    def __init__(
        self,
        corpus_path,
        top_k=5,
    ):
        self.corpus_path = Path(corpus_path)
        self.top_k = top_k

        self.documents = []

        self._load_corpus()

    def _load_corpus(self):
        with self.corpus_path.open(
            "r",
            encoding="utf-8",
        ) as f:

            for line in f:
                line = line.strip()

                if not line:
                    continue

                document = json.loads(line)

                self.documents.append(document)

        print(
            f"Loaded {len(self.documents)} CWQ documents."
        )

    def retrieve(self, query: str):

        if not isinstance(query, str):
            raise TypeError(
                "query must be a string"
            )

        query = query.strip()

        if not query:
            raise ValueError(
                "query cannot be empty"
            )

        query_terms = set(
            query.lower().split()
        )

        results = []

        for document in self.documents:

            text = document.get(
                "text",
                "",
            )

            document_terms = set(
                text.lower().split()
            )

            overlap = query_terms.intersection(
                document_terms
            )

            score = len(overlap)

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

        return results[: self.top_k]