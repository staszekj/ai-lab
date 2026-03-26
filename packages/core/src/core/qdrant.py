"""Qdrant client factory with mTLS support.

Connection to https://qdrant.eltrue using:
- CA certificate:  ~/.secrets/eltrue-ca.crt
- Client cert:     ~/.secrets/eltrue-client.crt
- Client key:      ~/.secrets/eltrue-client.key
- API key:         ~/.secrets/qdrant-api-key

Usage:
    from core.qdrant import get_client

    client = get_client()
    print(client.get_collections())
"""

import ssl
from pathlib import Path

from qdrant_client import QdrantClient

SECRETS_DIR = Path.home() / ".secrets" / "dev"

QDRANT_URL = "https://qdrant.eltrue:443"
CA_CERT = SECRETS_DIR / "eltrue-ca.crt"
CLIENT_CERT = SECRETS_DIR / "eltrue-client-tls.crt"
CLIENT_KEY = SECRETS_DIR / "eltrue-client-tls.key"
API_KEY_FILE = SECRETS_DIR / "qdrant-api-key"


def _build_ssl_context() -> ssl.SSLContext:
    ctx = ssl.create_default_context(cafile=str(CA_CERT))
    ctx.load_cert_chain(certfile=str(CLIENT_CERT), keyfile=str(CLIENT_KEY))
    return ctx


def get_client(
    url: str = QDRANT_URL,
    timeout: int = 30,
    prefer_grpc: bool = False,
) -> QdrantClient:
    """Create a QdrantClient with mTLS + API key authentication."""
    return QdrantClient(
        url=url,
        api_key=API_KEY_FILE.read_text().strip(),
        verify=_build_ssl_context(),
        prefer_grpc=prefer_grpc,
        timeout=timeout,
    )
