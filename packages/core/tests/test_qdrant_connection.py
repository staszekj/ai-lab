"""Smoke test: verify connectivity to qdrant.eltrue."""

from core.qdrant import get_client


def test_qdrant_connection():
    client = get_client()
    result = client.get_collections()
    assert result is not None

    collection_names = {c.name for c in result.collections}
    expected_collections = {"bible-nt", "bible-nt-voyage", "bible-nt-chapters-voyage"}
    assert expected_collections.issubset(collection_names), (
        f"Missing collections: {expected_collections - collection_names}"
    )
