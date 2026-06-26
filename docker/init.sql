-- Create required extensions
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS vectorscale CASCADE;
CREATE EXTENSION IF NOT EXISTS pg_textsearch;
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;

-- Verify extensions are installed
SELECT extname, extversion FROM pg_extension WHERE extname IN ('vector', 'pg_trgm', 'vectorscale', 'pg_textsearch', 'pg_stat_statements');
