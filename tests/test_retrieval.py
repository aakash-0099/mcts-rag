from retrieval import (
    RetrievalPipeline,
    DefaultQueryGenerator,
    DefaultRetriever,
    DefaultKnowledgeReflector,
    DefaultKnowledgeSummarizer,
)


def test_query_generation():
    generator = DefaultQueryGenerator()

    result = generator.generate(
        "What is Monte Carlo Tree Search?"
    )

    assert result == "What is Monte Carlo Tree Search?"


def test_query_generation_rejects_empty_query():
    generator = DefaultQueryGenerator()

    try:
        generator.generate("   ")
        assert False
    except ValueError:
        pass


def test_retrieval_execution():
    retriever = DefaultRetriever(
        documents=[
            "Monte Carlo Tree Search uses simulations.",
            "Retrieval augmented generation uses documents.",
            "Python is a programming language.",
        ]
    )

    results = retriever.retrieve(
        "Monte Carlo Tree Search"
    )

    assert len(results) > 0
    assert "Monte" in results[0]["document"]


def test_knowledge_reflection():
    reflector = DefaultKnowledgeReflector()

    reflection = reflector.reflect(
        "MCTS",
        [
            {
                "document": "MCTS uses simulations.",
                "score": 1,
            }
        ],
    )

    assert reflection["useful"] is True
    assert len(reflection["results"]) == 1


def test_knowledge_reflection_without_results():
    reflector = DefaultKnowledgeReflector()

    reflection = reflector.reflect(
        "MCTS",
        [],
    )

    assert reflection["useful"] is False
    assert reflection["results"] == []


def test_knowledge_summary():
    summarizer = DefaultKnowledgeSummarizer()

    summary = summarizer.summarize(
        "MCTS",
        {
            "useful": True,
            "results": [
                {
                    "document": "MCTS uses simulations.",
                    "score": 1,
                }
            ],
        },
    )

    assert summary == "MCTS uses simulations."


def test_full_retrieval_pipeline():
    pipeline = RetrievalPipeline(
        retriever=DefaultRetriever(
            documents=[
                "MCTS uses simulations to estimate rewards.",
                "RAG retrieves relevant documents.",
            ]
        )
    )

    result = pipeline.retrieve(
        "MCTS simulations"
    )

    assert result.query == "MCTS simulations"
    assert len(result.results) > 0
    assert result.reflection["useful"] is True
    assert result.summary != ""