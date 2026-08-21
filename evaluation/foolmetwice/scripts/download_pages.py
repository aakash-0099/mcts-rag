import json
import re
import time
from pathlib import Path
from urllib.parse import quote

import requests
from bs4 import BeautifulSoup


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]

PAGES_FILE = (
    PROJECT_ROOT
    / "evaluation"
    / "foolmetwice"
    / "processed"
    / "dev_100_pages.txt"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "evaluation"
    / "foolmetwice"
    / "corpus"
)


# ---------------------------------------------------------
# Wikipedia API
# ---------------------------------------------------------

API_BASE = "https://en.wikipedia.org/w/rest.php/v1/page"

HEADERS = {
    "User-Agent": (
        "MCTS-RAG-FoolMeTwice-Evaluation/1.0 "
        "(research evaluation)"
    )
}


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def clean_text(text: str) -> str:
    """Clean extracted Wikipedia text."""

    text = text.replace("\xa0", " ")

    # Normalize whitespace
    text = re.sub(r"[ \t]+", " ", text)

    # Remove excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def fetch_page(title: str, max_retries: int = 6) -> dict | None:
    """Fetch a Wikipedia page with rate-limit handling."""

    encoded_title = quote(title, safe="")
    url = f"{API_BASE}/{encoded_title}/with_html"

    for attempt in range(max_retries):

        try:
            response = requests.get(
                url,
                headers=HEADERS,
                timeout=30,
            )

            if response.status_code == 404:
                print(f"  [NOT FOUND] {title}")
                return None

            if response.status_code == 429:
                retry_after = response.headers.get(
                    "Retry-After"
                )

                if retry_after:
                    wait_seconds = int(retry_after)
                else:
                    wait_seconds = min(
                        60,
                        5 * (2 ** attempt)
                    )

                print(
                    f"  [RATE LIMITED] "
                    f"Waiting {wait_seconds}s..."
                )

                time.sleep(wait_seconds)
                continue

            response.raise_for_status()

            return response.json()

        except requests.RequestException as e:

            if attempt == max_retries - 1:
                print(
                    f"  [ERROR] Failed after "
                    f"{max_retries} attempts: {e}"
                )
                return None

            wait_seconds = min(
                60,
                5 * (2 ** attempt)
            )

            print(
                f"  [RETRY] Waiting "
                f"{wait_seconds}s..."
            )

            time.sleep(wait_seconds)

    return None


def html_to_sections(html: str) -> list[dict]:
    """
    Convert Wikipedia HTML into section-level text.
    """

    soup = BeautifulSoup(html, "html.parser")

    sections = []

    current_section = "Introduction"
    current_text = []

    for element in soup.find_all(
        ["h1", "h2", "h3", "h4", "p", "li"]
    ):

        tag = element.name

        text = element.get_text(
            " ",
            strip=True,
        )

        if not text:
            continue

        # New section
        if tag in {"h1", "h2", "h3", "h4"}:

            if current_text:
                sections.append(
                    {
                        "section": current_section,
                        "text": clean_text(
                            "\n".join(current_text)
                        ),
                    }
                )

            current_section = text
            current_text = []

        else:
            current_text.append(text)

    # Final section
    if current_text:
        sections.append(
            {
                "section": current_section,
                "text": clean_text(
                    "\n".join(current_text)
                ),
            }
        )

    return sections


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    if not PAGES_FILE.exists():
        raise FileNotFoundError(
            f"Page list not found:\n{PAGES_FILE}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with PAGES_FILE.open(
        "r",
        encoding="utf-8",
    ) as f:

        pages = [
            line.strip()
            for line in f
            if line.strip()
        ]

    print(f"Pages to download: {len(pages)}")
    print()

    successful = 0
    failed = []

    for index, title in enumerate(pages, start=1):

        output_file = (
            OUTPUT_DIR
            / f"wiki_{index:04d}.json"
        )
        if output_file.exists():
            print(
                f"  [SKIP] Already downloaded"
            )
            successful += 1
            continue
        print(
            f"[{index}/{len(pages)}] "
            f"Downloading: {title}"
        )

        try:

            data = fetch_page(title)

            if data is None:
                failed.append(title)
                continue

            html = data.get("html", "")

            if not html:
                print(
                    f"  [WARNING] No HTML returned"
                )
                failed.append(title)
                continue

            sections = html_to_sections(html)

            document = {
                "doc_id": f"wiki_{index:04d}",
                "title": title,
                "source": "wikipedia",
                "sections": sections,
            }

            with output_file.open(
                "w",
                encoding="utf-8",
            ) as f:

                json.dump(
                    document,
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

            successful += 1

            print(
                f"  Saved {len(sections)} sections"
            )

            # Be polite to Wikipedia
            time.sleep(3)

        except Exception as e:

            print(
                f"  [ERROR] {e}"
            )

            failed.append(title)

    # -----------------------------------------------------
    # Summary
    # -----------------------------------------------------

    print()
    print("=" * 50)
    print("Wikipedia download complete")
    print("=" * 50)

    print(f"Successful : {successful}")
    print(f"Failed     : {len(failed)}")

    if failed:

        failed_file = (
            OUTPUT_DIR
            / "failed_pages.txt"
        )

        with failed_file.open(
            "w",
            encoding="utf-8",
        ) as f:

            for title in failed:
                f.write(title + "\n")

        print()
        print(
            f"Failed pages saved to:\n"
            f"{failed_file}"
        )


if __name__ == "__main__":
    main()