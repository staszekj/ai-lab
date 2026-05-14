"""Aggregate verses into chapters."""

from collections import OrderedDict
from dataclasses import dataclass

from bible_to_qdrant.fetcher import Verse


@dataclass(frozen=True)
class Chapter:
    book_code: str
    book_name: str
    book_name_pl: str
    chapter: int
    text: str
    verse_count: int

    @property
    def reference(self) -> str:
        return f"{self.book_name} {self.chapter}"


def verses_to_chapters(verses: list[Verse]) -> list[Chapter]:
    """Group verses by book+chapter and concatenate their texts."""
    grouped: OrderedDict[tuple[str, int], list[Verse]] = OrderedDict()
    for v in verses:
        key = (v.book_code, v.chapter)
        grouped.setdefault(key, []).append(v)

    chapters: list[Chapter] = []
    for (book_code, chapter_num), chapter_verses in grouped.items():
        first = chapter_verses[0]
        text = " ".join(v.text for v in chapter_verses)
        chapters.append(Chapter(
            book_code=book_code,
            book_name=first.book_name,
            book_name_pl=first.book_name_pl,
            chapter=chapter_num,
            text=text,
            verse_count=len(chapter_verses),
        ))
    return chapters
