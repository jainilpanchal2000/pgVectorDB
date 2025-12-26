# PostgreSQL with pgvector, pgvectorscale, and pg_textsearch

This Docker image includes PostgreSQL 17 with the following extensions:
- **pgvector**: Vector similarity search
- **pgvectorscale**: Scalable vector search from Timescale
- **pg_textsearch**: Full-text search capabilities from Timescale

## Build and Run

Build the image:
```bash
docker-compose build
```

Start the container:
```bash
docker-compose up -d
```

## Connect to the Database

```bash
docker exec -it pg17-vectorscale-textsearch psql -U user -d postgres
```

Or connect from your application:
- Host: localhost
- Port: 9002
- User: user
- Password: root
- Database: postgres

## Verify Extensions

After connecting to the database:
```sql
SELECT extname, extversion FROM pg_extension WHERE extname IN ('vector', 'vectorscale', 'textsearch');
```

## Extensions Documentation

- [pgvector](https://github.com/pgvector/pgvector)
- [pgvectorscale](https://github.com/timescale/pgvectorscale)
- [pg_textsearch](https://github.com/timescale/pg_textsearch)
