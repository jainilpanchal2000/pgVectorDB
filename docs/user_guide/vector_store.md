# Vector Store Operations

The core functionality of pgVectorDB involves interacting with the underlying PostgreSQL tables to manage your document collections. This guide covers all CRUD operations including advanced batch processing, updates, upserts, and the DiskANN label filtering workflow.

---

## Initialization (`initialize`)

Before performing any operations, you must instantiate and initialize the `pgVectorDB` object. This sets up the full PostgreSQL schema — not just a simple `CREATE TABLE`.

```python
from pgvectordb import pgVectorDB, IndexType

db = pgVectorDB(
    collection_name="my_collection",
    embedding_model=my_embedding_model,
    connection_string="postgresql+asyncpg://user:pass@localhost/db",
    index_type=IndexType.HNSW
)

await db.initialize()
```

!!! note "What `initialize()` does under the hood"
    1. **Validates & Ensures Extensions** — Checks that `vector`, `pg_trgm`, `vectorscale`, and `pg_textsearch` are installed. Raises clear errors if dependencies are missing (e.g., `vectorscale` is required for DiskANN).
    2. **Table Creation** — Uses LangChain's `PGVectorStore` engine to create the base table.
    3. **Full-Text Search Setup** — Adds a `tsvector` column and a `BEFORE INSERT OR UPDATE` trigger to auto-populate it from `content`. Builds a GIN index on the tsvector.
    4. **Trigram Indexing** — Creates a `gin_trgm_ops` GIN index on `content` for fuzzy matching.

!!! tip
    Run `initialize()` only **once per collection**. It is idempotent — safe to call again without losing data (`overwrite_existing=False` by default).

---

## Adding Documents

### `add_documents`

The most common insertion method. It accepts LangChain `Document` objects, automatically computes embeddings using your configured model, and inserts text plus metadata.

```python
from langchain_core.documents import Document

docs = [
    Document(page_content="Content 1", metadata={"source": "wiki"}),
    Document(page_content="Content 2", metadata={"source": "blog"})
]

# Optional: integer label arrays for DiskANN label-based filtering
labels = [[1, 2], [3]]  # One array per document
doc_ids = await db.add_documents(docs, labels=labels)
```

---

## Batch Ingestion

### `add_documents_batch`

Add large numbers of documents efficiently with automatic batching and progress tracking. Prevents memory overflow on large datasets.

```python
# Add 50,000 documents efficiently
all_ids = await db.add_documents_batch(
    large_doc_list,
    batch_size=500,
    show_progress=True   # Logs progress per batch
)
print(f"Added {len(all_ids)} documents")
```

### `add_documents_batch_isolated`

Per-batch error isolation. Each batch commits independently — if one batch fails, previously committed batches are **not** rolled back. Useful for agent/pipeline workflows requiring partial success.

```python
added_ids, failed_batches = await db.add_documents_batch_isolated(
    documents,
    batch_size=500,
    continue_on_error=True  # Keep going after a failed batch
)
print(f"Added {len(added_ids)} docs, {len(failed_batches)} batches failed")
```

!!! warning "AGNO Pattern"
    Use `add_documents_batch_isolated` when downstream consumers can tolerate partial commits — for example, in agentic ingestion pipelines where retrying the full dataset is expensive.

### `bulk_load_documents`

Optimized for **initial data loading**. Uses batched `INSERT ... ON CONFLICT DO UPDATE` and optionally drops/rebuilds vector indexes around the load for maximum throughput.

```python
count = await db.bulk_load_documents(
    large_dataset,
    drop_indexes_first=True,  # Recommended for initial loads
    show_progress=True
)
print(f"Loaded {count} documents")
```

!!! tip "When to use `bulk_load_documents`"
    Best for the **first-time population** of a collection. For incremental updates, use `add_documents_batch` or `upsert_documents` instead.

---

## Updating Documents

### `aupdate_documents`

Update existing documents without delete/re-add. Requires `langchain_id` in each document's metadata to identify the target row.

```python
# Update metadata only (fast — no re-embedding)
docs[0].metadata["status"] = "reviewed"
docs[0].metadata["langchain_id"] = "existing-uuid"
updated_ids = await db.aupdate_documents(docs, update_embeddings=False)

# Update content (re-embeds automatically)
docs[1].page_content = "Updated content here"
docs[1].metadata["langchain_id"] = "existing-uuid-2"
updated_ids = await db.aupdate_documents(docs, update_embeddings=True)
```

| `update_embeddings` | Effect |
|---------------------|--------|
| `False` | Updates `content` and `langchain_metadata` only — fast, no model call |
| `True` | Re-computes embedding from `page_content` — slower, required for content changes |

### `update_metadata`

Fast bulk metadata updates without re-embedding. Uses the JSONB `||` merge operator — a single `UPDATE` for all IDs.

```python
# Tag all documents as reviewed in one round-trip
doc_ids = ["id1", "id2", "id3"]
count = await db.update_metadata(
    ids=doc_ids,
    metadata_updates={"status": "reviewed", "reviewer": "alice"}
)
print(f"Updated {count} documents")
```

!!! tip
    `update_metadata` uses `COALESCE(langchain_metadata::jsonb, '{}') || :updates` — it **merges** new fields, preserving existing ones rather than replacing the entire metadata object.

### `upsert_documents`

Insert or update by MD5 content hash. Prevents duplicates when re-ingesting the same data.

```python
inserted, updated = await db.upsert_documents(
    documents,
    dedup_by_content=True   # Hash-based deduplication
)
print(f"Inserted: {inserted}, Updated: {updated}")
```

---

## Retrieving Documents

### `aget_by_ids`

Retrieve specific documents by their `langchain_id` values — useful for building citation/reference features.

```python
doc_ids = ["uuid-1", "uuid-2", "uuid-3"]
docs = await db.aget_by_ids(doc_ids)

for doc in docs:
    print(f"ID: {doc['id']}")
    print(f"Content: {doc['content'][:80]}")
    print(f"Metadata: {doc['metadata']}")
```

---

## Counting & Analytics

### `count_by_metadata`

Count documents matching a filter without returning any data — a lightweight `SELECT COUNT(*)` with full filter support.

```python
# Count total documents in collection
total = await db.count_by_metadata()

# Count by filter (uses the same 13 operators as search)
hr_count = await db.count_by_metadata(
    filter={"status": "active", "department": "HR"}
)
print(f"Active HR docs: {hr_count}")
```

!!! tip
    See [Metadata Filtering](filtering.md) for the full list of 13 filter operators (`$eq`, `$in`, `$between`, `$and`, `$or`, etc.).

---

## Deleting Documents

### `adelete`

Remove specific documents by their `langchain_id` values. Returns the count of deleted rows.

```python
deleted_count = await db.adelete(ids=["id_1", "id_2"])
print(f"Deleted {deleted_count} documents")
```

### `drop_collection`

Completely destroy the collection — drops the table and all associated indexes, triggers, and constraints.

```python
# ⚠️ Irreversible — deletes ALL data
await db.drop_collection()
```

!!! danger
    `drop_collection()` executes `DROP TABLE ... CASCADE`. There is no undo. Back up your data before calling this in production.

---

## DiskANN Label Filtering

DiskANN (via `vectorscale`) supports **label-based graph partitioning** inside the ANN index itself. The index is built with label awareness so filtered searches never traverse irrelevant graph nodes — orders of magnitude faster than post-retrieval metadata filtering at scale (>10M vectors).

### Step 1: Define Your Labels

Create a human-readable label registry:

```python
await db.create_label_definitions([
    {"id": 1, "name": "science",    "description": "Scientific papers and research"},
    {"id": 2, "name": "technology", "description": "Tech news and documentation"},
    {"id": 3, "name": "finance",    "description": "Financial reports and analysis"},
])
```

### Step 2: Add Documents with Labels

Labels are `SMALLINT` integer arrays. Each document can belong to multiple labels:

```python
from langchain_core.documents import Document

docs = [
    Document(page_content="Black holes emit Hawking radiation",
             metadata={"source": "arxiv"}),
    Document(page_content="PostgreSQL 17 released with new features",
             metadata={"source": "pgsql"}),
    Document(page_content="Federal Reserve raises interest rates",
             metadata={"source": "reuters"}),
]
labels = [
    [1],        # science only
    [2],        # technology only
    [2, 3],     # technology + finance
]

doc_ids = await db.add_documents(docs, labels=labels)
```

!!! warning "Label Range"
    Labels must be integers in the `SMALLINT` range (`-32768` to `32767`). pgVectorDB validates this on insert and raises `ValidationError` for out-of-range values.

### Step 3: Build DiskANN Index with Label Awareness

```python
from pgvectordb import IndexType

db.index_type = IndexType.DISKANN
await db.build_index(include_labels=True)
```

### Step 4: Resolve Label Names to IDs at Query Time

```python
# Look up integer IDs by human-readable name
label_ids = await db.get_label_ids_by_names(["science", "technology"])
# Returns: [1, 2]
```

### Step 5: Search with Label Filter

```python
query_embedding = embedding_model.embed_query("quantum computing applications")

results = await db.asimilarity_search_by_vector(
    embedding=query_embedding,
    k=5,
    label_filter=label_ids   # Only search within science + technology
)

for r in results:
    print(f"Score: {r['score']:.4f} | {r['content'][:80]}")
```

!!! note "Labels vs. Metadata Filters"
    Metadata `filter=` applies a SQL `WHERE` clause **after** ANN graph traversal — the index still walks the full graph. DiskANN labels are applied **inside** the graph traversal — nodes in other partitions are never visited. This makes label-filtered search dramatically faster at scale.
