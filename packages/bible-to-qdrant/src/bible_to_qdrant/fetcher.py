"""Fetch and parse Bible TEI XML from TextGrid repository."""

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import httpx

from bible_to_qdrant.books import Book

TEI_NS = "http://www.tei-c.org/ns/1.0"
RDF_NS = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
ORE_NS = "http://www.openarchives.org/ore/terms/"


@dataclass(frozen=True)
class Verse:
    book_code: str
    book_name: str
    book_name_pl: str
    chapter: int
    verse: int
    text: str

    @property
    def reference(self) -> str:
        return f"{self.book_name} {self.chapter}:{self.verse}"


def _resolve_aggregation(client: httpx.Client, url: str) -> str:
    """Fetch aggregation RDF and return the URL of the actual TEI resource."""
    resp = client.get(url)
    resp.raise_for_status()
    root = ET.fromstring(resp.text)
    # Find the aggregated resource ID
    for desc in root.iter(f"{{{RDF_NS}}}Description"):
        for agg in desc.iter():
            resource = agg.get(f"{{{RDF_NS}}}resource")
            if resource and resource.startswith("textgrid:"):
                tg_id = resource.removeprefix("textgrid:")
                return f"https://textgridlab.org/1.0/tgcrud-public/rest/textgrid:{tg_id}/data"
    raise ValueError(f"No aggregated resource found in {url}")


def _parse_tei(xml_text: str, book: Book) -> list[Verse]:
    """Parse TEI XML and extract verses."""
    root = ET.fromstring(xml_text)
    verses: list[Verse] = []
    for ab in root.iter(f"{{{TEI_NS}}}ab"):
        if ab.get("type") != "verse":
            continue
        xml_id = ab.get("{http://www.w3.org/XML/1998/namespace}id", "")
        text = (ab.text or "").strip()
        if not text:
            continue
        # Parse ID like "b.MAT.001.003"
        match = re.match(r"b\.(\w+)\.(\d+)\.(\d+)", xml_id)
        if not match:
            continue
        chapter = int(match.group(2))
        verse_num = int(match.group(3))
        verses.append(Verse(
            book_code=book.code,
            book_name=book.name,
            book_name_pl=book.name_pl,
            chapter=chapter,
            verse=verse_num,
            text=text,
        ))
    return verses


def fetch_book(client: httpx.Client, book: Book) -> list[Verse]:
    """Download a book from TextGrid and return parsed verses."""
    tei_url = _resolve_aggregation(client, book.url)
    resp = client.get(tei_url)
    resp.raise_for_status()
    return _parse_tei(resp.text, book)


def fetch_new_testament(books: list[Book]) -> list[Verse]:
    """Fetch all New Testament books and return all verses."""
    all_verses: list[Verse] = []
    with httpx.Client(timeout=60) as client:
        for book in books:
            print(f"  Fetching {book.name_pl} ({book.code})...")
            verses = fetch_book(client, book)
            print(f"    → {len(verses)} verses")
            all_verses.extend(verses)
    return all_verses
