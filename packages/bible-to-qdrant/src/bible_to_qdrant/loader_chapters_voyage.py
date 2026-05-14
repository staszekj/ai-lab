"""Load Bible chapters into Qdrant using Voyage AI embeddings."""

from pathlib import Path

import voyageai
from qdrant_client import models

from core.qdrant import get_client
from bible_to_qdrant.chapters import Chapter

COLLECTION_NAME = "bible-nt-chapters-voyage"
VOYAGE_MODEL = "voyage-4-large"
API_KEY_FILE = Path.home() / ".secrets" / "voyageai-for-bible-to-qdrant-api-key"
EMBED_BATCH_SIZE = 16  # chapters are larger, fewer per batch
UPLOAD_BATCH_SIZE = 8


def _get_voyage_client() -> voyageai.Client:
    api_key = API_KEY_FILE.read_text().strip()
    return voyageai.Client(api_key=api_key)


def _embed_chapters(chapters: list[Chapter]) -> list[list[float]]:
    """Compute embeddings for chapter texts using Voyage AI."""
    vo = _get_voyage_client()
    texts = [ch.text for ch in chapters]
    total = len(texts)
    all_embeddings: list[list[float]] = []

    for start in range(0, total, EMBED_BATCH_SIZE):
        batch = texts[start : start + EMBED_BATCH_SIZE]
        result = vo.embed(batch, model=VOYAGE_MODEL, input_type="document")
        all_embeddings.extend(result.embeddings)
        done = len(all_embeddings)
        print(f"  Embedded {done}/{total} chapters ({done * 100 // total}%) — {result.total_tokens} tokens")

    return all_embeddings


def upload_chapters_voyage(chapters: list[Chapter]) -> None:
    """Embed chapters with Voyage AI and upload to Qdrant."""
    print(f"Computing embeddings for {len(chapters)} chapters ({VOYAGE_MODEL})...")
    embeddings = _embed_chapters(chapters)

    vector_size = len(embeddings[0])
    print(f"Vector size: {vector_size}")

    print("Creating Qdrant collection...")
    client = get_client()
    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=vector_size,
            distance=models.Distance.COSINE,
        ),
    )

    points = [
        models.PointStruct(
            id=i,
            vector=emb,
            payload={
                "book_code": ch.book_code,
                "book_name": ch.book_name,
                "book_name_pl": ch.book_name_pl,
                "chapter": ch.chapter,
                "reference": ch.reference,
                "verse_count": ch.verse_count,
                "text": ch.text,
            },
        )
        for i, (ch, emb) in enumerate(zip(chapters, embeddings))
    ]

    for start in range(0, len(points), UPLOAD_BATCH_SIZE):
        batch = points[start : start + UPLOAD_BATCH_SIZE]
        client.upsert(collection_name=COLLECTION_NAME, points=batch)
        done = min(start + UPLOAD_BATCH_SIZE, len(points))
        print(f"  Uploaded {done}/{len(points)} points")

    print(f"Done! Collection '{COLLECTION_NAME}' has {len(points)} chapters.")
