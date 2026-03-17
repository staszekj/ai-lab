"""Load Bible verses into Qdrant with embeddings."""

from fastembed import TextEmbedding
from qdrant_client import models

from core.qdrant import get_client
from bible_to_qdrant.fetcher import Verse

COLLECTION_NAME = "bible-nt"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
BATCH_SIZE = 128


def _embed_verses(verses: list[Verse]) -> list[list[float]]:
    """Compute embeddings for verse texts using a multilingual model."""
    print(f"  Loading model {EMBEDDING_MODEL}...")
    model = TextEmbedding(model_name=EMBEDDING_MODEL)
    texts = [v.text for v in verses]
    total = len(texts)
    embeddings: list[list[float]] = []
    for i, batch_emb in enumerate(model.embed(texts, batch_size=BATCH_SIZE)):
        embeddings.append(batch_emb.tolist())
        done = len(embeddings)
        if done % BATCH_SIZE == 0 or done == total:
            print(f"  Embedded {done}/{total} verses ({done * 100 // total}%)")
    return embeddings


def create_collection(vector_size: int) -> None:
    """Create the Qdrant collection (recreate if exists)."""
    client = get_client()
    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=vector_size,
            distance=models.Distance.COSINE,
        ),
    )


def upload_verses(verses: list[Verse]) -> None:
    """Embed verses and upload to Qdrant."""
    print(f"Computing embeddings for {len(verses)} verses ({EMBEDDING_MODEL})...")
    embeddings = _embed_verses(verses)

    vector_size = len(embeddings[0])
    print(f"Vector size: {vector_size}")

    print("Creating Qdrant collection...")
    create_collection(vector_size)

    client = get_client()
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

    # Upload in batches
    for start in range(0, len(points), BATCH_SIZE):
        batch = points[start : start + BATCH_SIZE]
        client.upsert(collection_name=COLLECTION_NAME, points=batch)
        print(f"  Uploaded {min(start + BATCH_SIZE, len(points))}/{len(points)} points")

    print(f"Done! Collection '{COLLECTION_NAME}' has {len(points)} verses.")
