# bible-to-qdrant

Load Polish Bible (New Testament, Biblia Gdańska 1881) verses into Qdrant as vector embeddings.

## Source

[TextGrid Repository – Polish Bible Collection](https://textgridrep.org/browse/49b6j.0)

## Usage

```bash
uv run --package bible-to-qdrant bible-to-qdrant
```

## What it does

1. Fetches all 27 New Testament books (TEI XML) from TextGrid
2. Parses individual verses (~8000 verses)
3. Computes embeddings using `paraphrase-multilingual-MiniLM-L12-v2`
4. Uploads to Qdrant collection `bible-nt`

Each point in Qdrant stores:
- **vector**: multilingual sentence embedding
- **payload**: `book_code`, `book_name`, `book_name_pl`, `chapter`, `verse`, `reference`, `text`
