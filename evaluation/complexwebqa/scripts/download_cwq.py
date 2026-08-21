from pathlib import Path
import urllib.request


BASE_URL = (
    "https://www.dropbox.com/sh/7pkwkrfnwqhsnpo/"
    "AADH8beLbOUWxwvY_K38E3ADa/"
    "ComplexWebQuestions_dev.json?dl=1"
)

OUTPUT_DIR = Path("evaluation/complexwebqa/data")
OUTPUT_FILE = OUTPUT_DIR / "ComplexWebQuestions_dev.json"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if OUTPUT_FILE.exists():
        print(f"Already exists: {OUTPUT_FILE}")
        return

    print("Downloading CWQ dev set...")

    urllib.request.urlretrieve(
        BASE_URL,
        OUTPUT_FILE,
    )

    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()