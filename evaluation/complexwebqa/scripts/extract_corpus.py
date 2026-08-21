import json
from pathlib import Path


EVAL_FILE = Path(
    "evaluation/complexwebqa/data/test_100.jsonl"
)

SNIPPET_FILE = Path(
    "evaluation/complexwebqa/data/web_snippets_test.json"
)

OUTPUT_FILE = Path(
    "evaluation/complexwebqa/data/corpus.jsonl"
)

REPORT_FILE = Path(
    "evaluation/complexwebqa/data/corpus_report.json"
)


def load_target_ids():
    target_ids = set()

    with EVAL_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                target_ids.add(item["id"])

    return target_ids


def iter_json_array(path):
    """
    Stream objects from a large JSON array without loading
    the entire file into memory.
    """

    decoder = json.JSONDecoder()

    with path.open("r", encoding="utf-8") as f:

        # Find opening '['
        while True:
            char = f.read(1)

            if not char:
                raise ValueError("Unexpected end of file.")

            if not char.isspace():
                break

        if char != "[":
            raise ValueError("Expected a JSON array.")

        buffer = ""

        while True:
            chunk = f.read(1024 * 1024)

            if not chunk:
                break

            buffer += chunk

            while True:
                buffer = buffer.lstrip()

                if not buffer:
                    break

                if buffer[0] == "]":
                    return

                if buffer[0] == ",":
                    buffer = buffer[1:]
                    continue

                try:
                    obj, index = decoder.raw_decode(buffer)
                except json.JSONDecodeError:
                    break

                yield obj
                buffer = buffer[index:]


def main():

    print("Loading evaluation IDs...")

    target_ids = load_target_ids()

    print(f"Target questions: {len(target_ids)}")

    found_ids = set()

    snippet_counts = {
        question_id: 0
        for question_id in target_ids
    }

    matched_records = 0
    total_snippets = 0

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8"
    ) as output:

        print("Scanning CWQ web snippets...")
        print(f"Source: {SNIPPET_FILE}")

        for record in iter_json_array(SNIPPET_FILE):

            question_id = record.get("question_ID")

            if question_id not in target_ids:
                continue

            found_ids.add(question_id)
            matched_records += 1

            snippets = record.get(
                "web_snippets",
                []
            )

            for index, item in enumerate(snippets):

                text = item.get(
                    "snippet",
                    ""
                ).strip()

                title = item.get(
                    "title",
                    ""
                ).strip()

                if not text:
                    continue

                doc_id = (
                    f"{question_id}_"
                    f"{matched_records}_"
                    f"{index}"
                )

                document = {
                    "doc_id": doc_id,
                    "question_id": question_id,
                    "text": text,
                    "title": title,
                    "web_query": record.get(
                        "web_query",
                        ""
                    ),
                    "split_type": record.get(
                        "split_type"
                    ),
                    "split_source": record.get(
                        "split_source",
                        []
                    ),
                    "source": "complexwebqa"
                }

                output.write(
                    json.dumps(
                        document,
                        ensure_ascii=False
                    )
                    + "\n"
                )

                snippet_counts[question_id] += 1
                total_snippets += 1

    missing_ids = sorted(
        target_ids - found_ids
    )

    report = {
        "target_questions": len(target_ids),
        "matched_question_ids": len(found_ids),
        "missing_question_ids": len(missing_ids),
        "matched_records": matched_records,
        "total_snippets": total_snippets,
        "snippet_counts": snippet_counts,
        "missing_ids": missing_ids,
    }

    with REPORT_FILE.open(
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            report,
            f,
            indent=2,
            ensure_ascii=False
        )

    print()
    print("=" * 60)
    print("CORPUS EXTRACTION COMPLETE")
    print("=" * 60)
    print(f"Target questions:       {len(target_ids)}")
    print(f"Matched question IDs:   {len(found_ids)}")
    print(f"Missing question IDs:   {len(missing_ids)}")
    print(f"Matched records:        {matched_records}")
    print(f"Total snippets:         {total_snippets}")
    print()
    print(f"Corpus:  {OUTPUT_FILE}")
    print(f"Report:  {REPORT_FILE}")

    if missing_ids:
        print()
        print("MISSING QUESTION IDs:")
        for question_id in missing_ids:
            print(f"  {question_id}")


if __name__ == "__main__":
    main()