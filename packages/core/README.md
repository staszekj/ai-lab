# core

Shared utilities for the **ai-lab** monorepo.

## What's inside

| Module | Description |
|---|---|
| `core.qdrant` | Qdrant client factory with mTLS + API key |

## Usage

```python
from core.qdrant import get_client

client = get_client()
client.get_collections()
```

## Qdrant connection

Connects to `https://qdrant.eltrue` using mTLS. Required files in `~/.secrets/`:

| File | Purpose |
|---|---|
| `eltrue-ca.crt` | CA certificate |
| `eltrue-client.crt` | Client certificate |
| `eltrue-client.key` | Client private key |
| `qdrant-api-key` | API key |
