# Configuration & Connection Management

pgVectorDB is designed to run in highly concurrent production environments. Connection management and database tuning are handled automatically via the `Config` object, which configures the underlying SQLAlchemy `AsyncEngine`.

## The `Config` Object

The `Config` class holds all database and query tuning parameters. 

```python
from pgvectordb import Config

custom_config = Config(
    schema_name="private_schema",
    pool_size=50,
    max_overflow=20,
    pool_timeout=60,
    enable_textsearch=True,
    textsearch_config="english"
)

```

### Connection Pooling Parameters
pgVectorDB utilizes an asynchronous connection pool. Every concurrent request to your application borrows a connection from this pool, avoiding the massive overhead of establishing a new TCP connection to PostgreSQL for every search query.

- **`pool_size` (Default: 5)**: The number of permanent connections kept open to the database. If you have a high-traffic web server (like FastAPI), you should increase this to match your worker count.
- **`max_overflow` (Default: 10)**: If all `pool_size` connections are busy, SQLAlchemy will open up to `max_overflow` temporary connections. These are closed and discarded when the spike subsides.
- **`pool_timeout` (Default: 30)**: If the pool is completely exhausted, incoming requests will wait this many seconds for a connection to free up before throwing a `TimeoutError`.

### Schema & Security
- **`schema_name` (Default: "public")**: Specifies the PostgreSQL schema. When you initialize pgVectorDB, it strictly validates this name (allowing only alphanumeric characters and underscores) to prevent SQL injection vulnerabilities before dynamically generating the DDL queries.

### Search Defaults
These parameters configure the default runtime behavior of the PostgreSQL indexes when executing searches.
- **`DEFAULT_IVFFLAT_PROBES` (Default: 10)**: How many neighboring IVFFlat clusters to scan.
- **`DEFAULT_HNSW_EF_SEARCH` (Default: 40)**: The depth of the HNSW graph traversal during search.

## Environment-Specific Factories

pgVectorDB provides pre-built factory functions that instantly load optimized configurations based on your runtime environment.

### 1. `get_production_config()`
Optimized for high concurrency.
- `pool_size`: 20
- `max_overflow`: 10
- `pool_timeout`: 30

```python
from pgvectordb import pgVectorDB, get_production_config

db = pgVectorDB(
    collection_name="docs",
    connection_string="postgresql+asyncpg://user:pass@localhost:9002/postgres",
    config=get_production_config()
)

```

### 2. `get_test_config()`
Optimized for unit testing and CI/CD pipelines. It disables connection pooling entirely by injecting SQLAlchemy's `NullPool`. This guarantees that connections are immediately closed after every test, preventing "Database is being accessed by other users" errors when test runners try to tear down or drop databases.

```python
from pgvectordb import pgVectorDB, get_test_config

# Safe for PyTest fixtures
db = pgVectorDB(
    collection_name="test_docs",
    connection_string="postgresql+asyncpg://user:pass@localhost:9002/postgres",
    config=get_test_config()
)

```

## Environment Variables

If you do not explicitly pass a `connection_string` to the `pgVectorDB` constructor, the system will automatically look for the `PGVECTORDB_URL` environment variable. 

```bash
# Set this in your Docker container or .env file
export PGVECTORDB_URL="postgresql+asyncpg://user:root@localhost:9002/postgres"

```

```python
# The connection string is picked up automatically
db = pgVectorDB(collection_name="my_docs")

```
