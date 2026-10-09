from .config import CHROMA_DIR, COLLECTION_NAME
from .store import ChromaStore


def main() -> None:
    database = ChromaStore(str(CHROMA_DIR), COLLECTION_NAME)
    titles = sorted(set(database.indexed_titles()), key=str.casefold)

    print(f"Indexed pages: {len(titles)}\n")

    for title in titles:
        print(title)


if __name__ == "__main__":
    main()
