"""Load Bible verses into Qdrant using Voyage AI embeddings."""

from pathlib import Path

import voyageai
from qdrant_client import models

from core.qdrant import get_client
from bible_to_qdrant.fetcher import Verse

COLLECTION_NAME = "bible-nt-voyage"
VOYAGE_MODEL = "voyage-4-large"
API_KEY_FILE = Path.home() / ".secrets" / "voyageai-for-bible-to-qdrant-api-key"
BATCH_SIZE = 128
UPLOAD_BATCH_SIZE = 32  # smaller upload batches for 1024d vectors (nginx body limit)


def _get_voyage_client() -> voyageai.Client:
    api_key = API_KEY_FILE.read_text().strip()
    return voyageai.Client(api_key=api_key)


def _embed_verses(verses: list[Verse]) -> list[list[float]]:
    """Compute embeddings using Voyage AI."""
    vo = _get_voyage_client()
    texts = [v.text for v in verses]
    total = len(texts)
    all_embeddings: list[list[float]] = []

    for start in range(0, total, BATCH_SIZE):
        batch = texts[start : start + BATCH_SIZE]
        result = vo.embed(batch, model=VOYAGE_MODEL, input_type="document")
        all_embeddings.extend(result.embeddings)
        done = len(all_embeddings)
        print(f"  Embedded {done}/{total} verses ({done * 100 // total}%) — {result.total_tokens} tokens")

    return all_embeddings


def upload_verses_voyage(verses: list[Verse]) -> None:
    """Embed verses with Voyage AI and upload to Qdrant."""
    print(f"Computing embeddings for {len(verses)} verses ({VOYAGE_MODEL})...")
    embeddings = _embed_verses(verses)

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
                "book_code": v.book_code,
                "book_name": v.book_name,
                "book_name_pl": v.book_name_pl,
                "chapter": v.chapter,
                "verse": v.verse,
                "reference": v.reference,
                "text": v.text,
            },
        )
        for i, (v, emb) in enumerate(zip(verses, embeddings))
    ]

    for start in range(0, len(points), UPLOAD_BATCH_SIZE):
        batch = points[start : start + UPLOAD_BATCH_SIZE]
        client.upsert(collection_name=COLLECTION_NAME, points=batch)
        done = min(start + UPLOAD_BATCH_SIZE, len(points))
        if done % 128 == 0 or done == len(points):
            print(f"  Uploaded {done}/{len(points)} points")

    print(f"Done! Collection '{COLLECTION_NAME}' has {len(points)} verses.")
