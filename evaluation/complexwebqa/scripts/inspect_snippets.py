import json
from pathlib import Path


SNIPPET_FILE = Path(
    "evaluation/complexwebqa/data/web_snippets_test.json"
)


def main():
    print(f"Reading: {SNIPPET_FILE}")
    print(f"Size: {SNIPPET_FILE.stat().st_size / (1024**2):.2f} MB")

    with SNIPPET_FILE.open(
        "r",
        encoding="utf-8"
    ) as f:

        # Skip whitespace and the opening '['
        first_char = f.read(1)

        while first_char.isspace():
            first_char = f.read(1)

        if first_char != "[":
            raise ValueError(
                "Expected the snippet file to contain a JSON array."
            )

        # Read enough data to contain the first object.
        buffer = ""
        decoder = json.JSONDecoder()

        while True:
            chunk = f.read(1024 * 1024)

            if not chunk:
                raise ValueError(
                    "Could not find the first JSON object."
                )

            buffer += chunk

            try:
                obj, _ = decoder.raw_decode(buffer.lstrip())
                break
            except json.JSONDecodeError:
                continue

    print("\nFirst record:")
    print(json.dumps(obj, indent=2, ensure_ascii=False))

    print("\nTop-level fields:")
    print(list(obj.keys()))


if __name__ == "__main__":
    main()