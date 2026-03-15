"""Smoke test: verify connectivity to qdrant.eltrue."""

from core.qdrant import get_client


def test_qdrant_connection():
    client = get_client()
    result = client.get_collections()
    assert result is not None
