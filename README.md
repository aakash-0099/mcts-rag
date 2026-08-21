# MCTS-RAG

A retrieval-augmented generation system that uses Monte Carlo Tree Search (MCTS) to explore alternative reasoning paths, with retrieval, decomposition, and answer selection built into the action space.

This project models a reasoning process as a tree of actions such as direct answering, quick reasoning, decomposition, retrieval-based reasoning, retrieval-plus-decomposition, and summary generation. It then ranks candidate answers using a selection and clustering pipeline.

## Overview

The repository implements a research-style MCTS workflow for multi-step question answering:

- the root node starts from the raw question
- actions expand the search tree
- retrieval can be triggered as part of the reasoning path
- terminal states emit candidate answers
- a reward model evaluates whether the answer is grounded in the retrieved context
- the best-supported answer cluster is selected as the final answer

This is inspired by action-driven reasoning frameworks where agents choose among several reasoning policies instead of following a single fixed chain-of-thought.

## Features

- MCTS-based search over reasoning trajectories
- Six action types: A1-A6
- Retrieval pipeline with query generation, retrieval, reflection, and summarization
- Candidate answer collection and clustering
- Reward evaluation grounded in retrieved context
- Interactive local run via the command line
- Evaluation scripts for benchmark datasets

## Project structure

```text
.
├── actions/
│   ├── __init__.py
│   ├── action.py                # Action enum for the six reasoning actions
│   ├── answer_selection.py      # Candidate clustering and answer selection
│   ├── candidates.py            # Candidate answer collection from the MCTS tree
│   ├── clustering.py            # Answer clustering logic
│   ├── decomposition.py         # Query decomposition strategy
│   ├── dispatcher.py            # Action dispatch / executor mapping
│   ├── evaluator.py             # Grounding-based reward evaluator
│   ├── executors.py             # Action executors for A1-A6
│   ├── integration.py           # MCTS expansion and simulation integration
│   ├── llm.py                   # LLM wrapper (currently Ollama-backed)
│   ├── reward.py                # Reward scoring for answer clusters
│   ├── state.py                 # ActionState data structure
│   ├── terminal.py              # Terminal status / candidate logic
│   └── trajectory.py            # Trajectory helpers
├── documents/
│   └── doc.txt                 # Default local document corpus used by the demo app
├── evaluation/
│   ├── complexwebqa/
│   └── foolmetwice/
├── mcts/
│   ├── __init__.py
│   ├── backpropagation.py       # Backpropagation logic
│   ├── expansion.py             # Generic expansion interface
│   ├── mcts.py                  # Generic MCTS implementation
│   ├── node.py                  # Tree node representation
│   ├── selection.py             # Selection policy (UCT)
│   └── simulation.py            # Simulation strategy interface
├── mcts_rag/
│   ├── __init__.py
│   └── pipeline.py              # End-to-end MCTS-RAG orchestration
├── retrieval/
│   ├── __init__.py
│   ├── pipeline.py              # Retrieval pipeline orchestration
│   ├── r1_query.py              # Query generation stages
│   ├── r2_retrieval.py          # Default retriever implementation
│   ├── r3_reflection.py         # Result relevance reflection
│   ├── r4_summary.py            # Result summarization
│   └── strategy.py              # RetrievalStrategy abstraction
├── .env                         # Local environment variables
├── .gitignore
├── evaluation.py                # Top-level evaluation helper
├── generator.py                 # Simple generator abstraction
├── main.py                      # Main interactive entry point
├── pytest.ini                   # Pytest config
├── rag.py                       # Lightweight RAG wrapper example
├── requirements.txt             # Python dependencies
├── retriever.py                 # Base retriever utility
├── search_state.py              # Search-state related helpers
├── tests/                       # Unit tests for action expansion, MCTS, retrieval, etc.
├── README.md
└── documents/doc.txt
```

## Architecture

### 1. Retrieval pipeline

The retrieval layer is split into four stages:

- R1: query generation
- R2: retrieval
- R3: reflection on retrieved knowledge
- R4: summarization of useful evidence

This is implemented in the retrieval package and exposes `RetrievalPipeline`, which wraps a `QueryGenerator`, `Retriever`, `KnowledgeReflector`, and `KnowledgeSummarizer`.

### 2. Reasoning actions

The action layer defines the six actions used during search:

- A1: direct answer
- A2: quick reasoning
- A3: decompose question
- A4: retrieval-based reasoning
- A5: retrieval + decomposition
- A6: summarized answer

The executor logic lives in `actions/executors.py`, and the action set is defined in `actions/action.py`.

### 3. MCTS loop

The tree search loop is implemented in `mcts/mcts.py` and relies on:

- `Node` for state representation
- `UCTSelection` for selecting promising nodes
- `ActionExpansion` to expand untried actions
- `ActionSimulation` to evaluate a trajectory
- `Backpropagation` to update node values

### 4. Grounded evaluation

Rewarding is not purely heuristic. In `actions/evaluator.py`, `TrajectoryEvaluator` checks whether the answer is actually supported by the retrieved evidence accumulated along the trajectory. This prevents branches that do not retrieve relevant context from being treated as equally good simply because they output a plausible answer.

### 5. Answer selection

After MCTS builds candidate answers from the tree, the system clusters them and picks the highest-scoring cluster in `actions/answer_selection.py`.

## Setup

### Prerequisites

- Python 3.10+ recommended
- Ollama installed and running locally
- A compatible model available, defaulting to `llama3.2:3b`

To install the model locally with Ollama:

```bash
ollama pull llama3.2:3b
```

### Install dependencies

```bash
python -m venv .venv
```

On macOS/Linux:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Running the app

The project is designed to run interactively from `main.py`.

```bash
python main.py
```

You will be prompted with:

```text
Question:
```

Then the system will:

1. build a retrieval strategy
2. run the MCTS search
3. print the search tree
4. print candidate answers with rewards
5. print the final selected answer

## Example usage

With the default setup, `main.py` loads the document corpus from `documents/doc.txt` and runs a retrieval-based MCTS pipeline:

```python
from pathlib import Path
from actions.llm import LLM
from mcts_rag.pipeline import MCTSRAG
from retrieval import RetrievalPipeline
from retrieval.r2_retrieval import DefaultRetriever

documents = [
    Path("documents/doc.txt").read_text(encoding="utf-8")
]

retrieval_strategy = RetrievalPipeline(
    retriever=DefaultRetriever(documents=documents)
)

llm = LLM()

mcts_rag = MCTSRAG(
    num_simulations=200,
    retrieval_strategy=retrieval_strategy,
    llm=llm,
)

result = mcts_rag.run("What is the main question?")
print(result.final_answer)
```

## Configuration notes

- The LLM backend is defined in `actions/llm.py` and currently uses `ChatOllama`.
- The default model is `llama3.2:3b`.
- The demo dataset is `documents/doc.txt`.
- `main.py` demonstrates the expected runtime workflow for the project.

## Evaluation and testing

The repository includes a test suite under `tests/` and evaluation scripts under `evaluation/`.

Run the project tests:

```bash
pytest
```

or more explicitly:

```bash
pytest -q
```

The benchmark folders include scripts for dataset preparation and evaluation, particularly for `complexwebqa` and `foolmetwice` corpora.

## Notes

This codebase is best thought of as a research prototype and an experimentation scaffold. It is intentionally modular so that retrieval, action logic, search strategy, answer selection, and LLM integration can be swapped or extended independently.

## License

This project does not currently declare a license in the repository metadata. If you plan to redistribute or reuse it, check the repository owner or project policy before publication.

## Contributing

Contributions are welcome if they improve:

- retrieval quality
- reward grounding
- action selection policy
- MCTS efficiency
- evaluation reliability

Before proposing a change, it is recommended to run the relevant tests and evaluate the effect on the answer quality pipeline.
