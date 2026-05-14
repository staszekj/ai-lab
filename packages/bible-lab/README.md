# bible-lab

Bible AI experiments.

## Dependencies

- [`core`](../core/) — shared utilities (Qdrant client, etc.)

## Quick start

```python
from core.qdrant import get_client

client = get_client()
# use client for Bible-related vector search experiments
```
