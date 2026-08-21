import json
import re
from pathlib import Path


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

# File:
# D:\MCTS RAG\evaluation\foolmetwice\scripts\verify_evidence.py
#
# parents[3] = D:\MCTS RAG

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATASET_FILE = (
    PROJECT_ROOT
    / "evaluation"
    / "foolmetwice"
    / "processed"
    / "dev_100.jsonl"
)

CORPUS_DIR = (
    PROJECT_ROOT
    / "evaluation"
    / "foolmetwice"
    / "corpus"
)

REPORT_FILE = (
    PROJECT_ROOT
    / "evaluation"
    / "foolmetwice"
    / "processed"
    / "evidence_verification.json"
)


# ---------------------------------------------------------
# Text normalization
# ---------------------------------------------------------

def normalize(text: str) -> str:
    """
    Normalize text so small formatting differences between
    FM2 evidence and Wikipedia do not cause false mismatches.
    """

    text = text.lower()

    # Normalize Unicode-style whitespace
    text = text.replace("\xa0", " ")

    # Normalize quotation marks
    text = (
        text
        .replace("“", '"')
        .replace("”", '"')
        .replace("‘", "'")
        .replace("’", "'")
    )

    # Normalize dashes
    text = (
        text
        .replace("–", "-")
        .replace("—", "-")
        .replace("-", "-")
    )

    # Remove citation markers such as [1], [23]
    text = re.sub(r"\[\d+\]", "", text)

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ---------------------------------------------------------
# Load dataset
# ---------------------------------------------------------

def load_dataset():
    examples = []

    with DATASET_FILE.open(
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            examples.append(
                json.loads(line)
            )

    return examples


# ---------------------------------------------------------
# Load corpus
# ---------------------------------------------------------

def load_corpus():

    documents = {}

    for path in sorted(
        CORPUS_DIR.glob("wiki_*.json")
    ):

        with path.open(
            "r",
            encoding="utf-8",
        ) as f:

            document = json.load(f)

        documents[
            document["title"]
        ] = document

    return documents


# ---------------------------------------------------------
# Build searchable document text
# ---------------------------------------------------------

def document_text(document):

    parts = []

    for section in document.get(
        "sections",
        []
    ):

        section_name = section.get(
            "section",
            ""
        )

        section_text = section.get(
            "text",
            ""
        )

        parts.append(section_name)
        parts.append(section_text)

    return normalize(
        "\n".join(parts)
    )


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    if not DATASET_FILE.exists():

        raise FileNotFoundError(
            f"Dataset not found:\n"
            f"{DATASET_FILE}"
        )

    if not CORPUS_DIR.exists():

        raise FileNotFoundError(
            f"Corpus directory not found:\n"
            f"{CORPUS_DIR}"
        )

    examples = load_dataset()
    documents = load_corpus()

    print(
        f"Loaded {len(examples)} FM2 examples."
    )

    print(
        f"Loaded {len(documents)} Wikipedia documents."
    )

    print()

    # -----------------------------------------------------
    # Statistics
    # -----------------------------------------------------

    fully_matched = 0
    missing_pages = 0
    missing_evidence = 0

    results = []

    # -----------------------------------------------------
    # Check every example
    # -----------------------------------------------------

    for example in examples:

        example_id = example["id"]
        title = example["wikipedia_page"]

        document = documents.get(title)

        result = {
            "id": example_id,
            "wikipedia_page": title,
            "page_found": False,
            "gold_evidence_count": len(
                example.get("gold_evidence", [])
            ),
            "matched_evidence_count": 0,
            "missing_evidence": [],
        }

        # -------------------------------------------------
        # Page missing
        # -------------------------------------------------

        if document is None:

            print(
                f"[MISSING PAGE] "
                f"{title}"
            )

            missing_pages += 1

            results.append(result)

            continue

        result["page_found"] = True

        corpus_text = document_text(
            document
        )

        # -------------------------------------------------
        # Check gold evidence
        # -------------------------------------------------

        example_complete = True

        for evidence in example.get(
            "gold_evidence",
            []
        ):

            gold_text = normalize(
                evidence.get(
                    "text",
                    ""
                )
            )

            if not gold_text:
                continue

            if gold_text in corpus_text:

                result[
                    "matched_evidence_count"
                ] += 1

            else:

                example_complete = False

                result[
                    "missing_evidence"
                ].append(
                    {
                        "section_header":
                            evidence.get(
                                "section_header"
                            ),
                        "text":
                            evidence.get(
                                "text"
                            ),
                    }
                )

        # -------------------------------------------------
        # Example result
        # -------------------------------------------------

        if example_complete:

            fully_matched += 1

        else:

            missing_evidence += 1

            print(
                f"[MISSING EVIDENCE] "
                f"{example_id} | "
                f"{title} | "
                f"{result['matched_evidence_count']}/"
                f"{result['gold_evidence_count']}"
            )

        results.append(result)

    # -----------------------------------------------------
    # Save report
    # -----------------------------------------------------

    report = {
        "total_examples": len(examples),
        "pages_available": len(documents),
        "fully_matched_examples": fully_matched,
        "missing_pages": missing_pages,
        "examples_with_missing_evidence":
            missing_evidence,
        "results": results,
    }

    with REPORT_FILE.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            report,
            f,
            ensure_ascii=False,
            indent=2,
        )

    # -----------------------------------------------------
    # Final summary
    # -----------------------------------------------------

    print()
    print("=" * 60)
    print("FM2 EVIDENCE VERIFICATION")
    print("=" * 60)

    print(
        f"Examples              : {len(examples)}"
    )

    print(
        f"Pages available       : {len(documents)}"
    )

    print(
        f"Examples fully matched: {fully_matched}"
    )

    print(
        f"Missing pages         : {missing_pages}"
    )

    print(
        f"Missing evidence      : {missing_evidence}"
    )

    print()
    print(
        f"Report saved to:\n{REPORT_FILE}"
    )

    # -----------------------------------------------------
    # Final status
    # -----------------------------------------------------

    if (
        fully_matched == len(examples)
        and missing_pages == 0
        and missing_evidence == 0
    ):

        print()
        print(
            "SUCCESS: Corpus fully matches "
            "the FM2 gold evidence."
        )

    else:

        print()
        print(
            "WARNING: Some FM2 evidence could "
            "not be matched."
        )


if __name__ == "__main__":
    main()