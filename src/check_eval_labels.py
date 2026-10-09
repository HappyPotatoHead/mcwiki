from collections import Counter

from .config import CHROMA_DIR, COLLECTION_NAME
from .evaluate import TEST_CASES
from .store import ChromaStore


def main() -> None:
    database = ChromaStore(str(CHROMA_DIR), COLLECTION_NAME)

    indexed = {title.casefold(): title for title in database.indexed_titles()}

    missing = Counter()

    for case in TEST_CASES:
        for title in case["relevant_titles"]:
            if title.casefold() not in indexed:
                missing[title] += 1

    print(f"Test cases: {len(TEST_CASES)}")
    print(f"Unique indexed titles: {len(indexed)}")
    print(f"Missing label occurrences: {sum(missing.values())}")
    print(f"Unique missing titles: {len(missing)}\n")

    print("Missing titles, sorted by frequency:")
    for title, count in missing.most_common():
        print(f"{count:>3}  {title}")


if __name__ == "__main__":
    main()
