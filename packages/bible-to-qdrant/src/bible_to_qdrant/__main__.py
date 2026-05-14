"""CLI: fetch Polish New Testament and load into Qdrant."""

from bible_to_qdrant.books import NT_BOOKS
from bible_to_qdrant.fetcher import fetch_new_testament


def _fetch_verses():
    print(f"Fetching {len(NT_BOOKS)} New Testament books from TextGrid...")
    verses = fetch_new_testament(NT_BOOKS)
    print(f"\nTotal: {len(verses)} verses\n")
    return verses


def main() -> None:
    from bible_to_qdrant.loader import upload_verses
    upload_verses(_fetch_verses())


def main_voyage() -> None:
    from bible_to_qdrant.loader_voyage import upload_verses_voyage
    upload_verses_voyage(_fetch_verses())


def main_chapters_voyage() -> None:
    from bible_to_qdrant.chapters import verses_to_chapters
    from bible_to_qdrant.loader_chapters_voyage import upload_chapters_voyage
    verses = _fetch_verses()
    chapters = verses_to_chapters(verses)
    print(f"Aggregated into {len(chapters)} chapters\n")
    upload_chapters_voyage(chapters)


if __name__ == "__main__":
    main()
