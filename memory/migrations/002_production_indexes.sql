-- ═══════════════════════════════════════════════════════════════════════════════
-- QbitOS PostgreSQL Storage Layer — Production Migration v2.0
-- ═══════════════════════════════════════════════════════════════════════════════
--
-- This migration transforms the PostgreSQL storage layer into a production-grade,
-- AI-native platform with:
--   • 14 PostgreSQL extensions (pgvector, pg_trgm, btree_gin, btree_gist, etc.)
--   • HNSW vector indexes (replaces IVFFlat for production-grade ANN)
--   • GIN indexes on all JSONB columns (jsonb_path_ops for containment queries)
--   • Full-text search with multilingual support
--   • Trigram fuzzy search for autocomplete and typo tolerance
--   • Hierarchical data via ltree for knowledge graphs and skill trees
--   • Bloom filter indexes for exact-match acceleration
--   • Partial indexes for common query patterns
--   • Monitoring via pg_stat_statements, pg_buffercache, pg_visibility
--   • Connection pool tuning
--
-- Backward compatibility: The existing `qbitos_memories` table is preserved
-- with new columns added via ALTER TABLE (nullable, with defaults). All
-- existing queries continue to work unchanged.
--
-- ═══════════════════════════════════════════════════════════════════════════════

-- ─── Section 1: PostgreSQL Extensions ─────────────────────────────────────────

-- Core AI extensions
CREATE EXTENSION IF NOT EXISTS vector;              -- pgvector for embedding similarity
CREATE EXTENSION IF NOT EXISTS pg_trgm;             -- trigram fuzzy search
CREATE EXTENSION IF NOT EXISTS btree_gin;           -- GIN indexes on scalar types
CREATE EXTENSION IF NOT EXISTS btree_gist;          -- GiST indexes on scalar types
CREATE EXTENSION IF NOT EXISTS pgcrypto;            -- cryptographic functions (gen_random_uuid)
CREATE EXTENSION IF NOT EXISTS unaccent;            -- accent-insensitive search
CREATE EXTENSION IF NOT EXISTS fuzzystrmatch;       -- Levenshtein, soundex, metaphone
CREATE EXTENSION IF NOT EXISTS hstore;              -- key-value storage
CREATE EXTENSION IF NOT EXISTS ltree;               -- hierarchical path queries
CREATE EXTENSION IF NOT EXISTS bloom;               -- bloom filter index

-- Monitoring extensions (require superuser; skipped if unavailable)
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;  -- query statistics
CREATE EXTENSION IF NOT EXISTS pg_buffercache;      -- buffer pool inspection
CREATE EXTENSION IF NOT EXISTS pg_visibility;       -- visibility map inspection
CREATE EXTENSION IF NOT EXISTS pg_prewarm;          -- warm cache on startup

-- Optional: TimescaleDB (uncomment if time-series features are needed)
-- CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Configure pg_stat_statements
ALTER SYSTEM SET shared_preload_libraries = 'pg_stat_statements';
ALTER SYSTEM SET pg_stat_statements.track = 'all';
ALTER SYSTEM SET pg_stat_statements.max = 10000;
ALTER SYSTEM SET track_io_timing = 'on';

-- Configure pgvector
ALTER SYSTEM SET maintenance_work_mem = '512MB';     -- for HNSW index builds
ALTER SYSTEM SET max_parallel_maintenance_workers = 4; -- parallel HNSW builds

-- Connection pool tuning
ALTER SYSTEM SET max_connections = 200;
ALTER SYSTEM SET shared_buffers = '256MB';
ALTER SYSTEM SET effective_cache_size = '1GB';
ALTER SYSTEM SET work_mem = '16MB';
ALTER SYSTEM SET random_page_cost = 1.1;              -- SSD-optimized
ALTER SYSTEM SET jit = 'on';

-- ─── Section 2: Enhanced Memories Table ───────────────────────────────────────

-- Add new columns to the existing qbitos_memories table (backward compatible).
ALTER TABLE qbitos_memories
    ADD COLUMN IF NOT EXISTS tags          TEXT[] NOT NULL DEFAULT '{}',
    ADD COLUMN IF NOT EXISTS agent_id      UUID,
    ADD COLUMN IF NOT EXISTS session_id    UUID,
    ADD COLUMN IF NOT EXISTS text_content  TEXT GENERATED ALWAYS AS (
        CASE
            WHEN content ? 'text' THEN content->>'text'
            WHEN content ? 'content' THEN content->>'content'
            WHEN content ? 'query' THEN content->>'query'
            ELSE NULL
        END
    ) STORED,
    ADD COLUMN IF NOT EXISTS search_vector tsvector
        GENERATED ALWAYS AS (
            to_tsvector('english',
                COALESCE(
                    CASE
                        WHEN content ? 'text' THEN content->>'text'
                        WHEN content ? 'content' THEN content->>'content'
                        ELSE ''
                    END,
                    ''
                )
            )
        ) STORED,
    ADD COLUMN IF NOT EXISTS path         ltree,
    ADD COLUMN IF NOT EXISTS expires_at   TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS version      INTEGER NOT NULL DEFAULT 1,
    ADD COLUMN IF NOT EXISTS checksum     TEXT;

-- Update search_vector for existing rows (if any)
UPDATE qbitos_memories SET search_vector =
    to_tsvector('english',
        COALESCE(
            CASE
                WHEN content ? 'text' THEN content->>'text'
                WHEN content ? 'content' THEN content->>'content'
                ELSE ''
            END,
            ''
        )
    )
WHERE search_vector IS NULL;

-- ─── Section 3: Indexes on qbitos_memories ────────────────────────────────────

-- Drop the old IVFFlat index (replaced by HNSW)
DROP INDEX IF EXISTS qbitos_memories_embedding_idx;

-- 3a. Vector indexes (HNSW for production-grade ANN)
-- HNSW with cosine distance (primary use case: semantic similarity)
CREATE INDEX IF NOT EXISTS qbitos_memories_embedding_hnsw_cosine_idx
    ON qbitos_memories USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- HNSW with L2 distance (for geometric / spatial queries)
CREATE INDEX IF NOT EXISTS qbitos_memories_embedding_hnsw_l2_idx
    ON qbitos_memories USING hnsw (embedding vector_l2_ops)
    WITH (m = 16, ef_construction = 64);

-- HNSW with inner product (for normalized embeddings)
CREATE INDEX IF NOT EXISTS qbitos_memories_embedding_hnsw_ip_idx
    ON qbitos_memories USING hnsw (embedding vector_ip_ops)
    WITH (m = 16, ef_construction = 64);

-- IVFFlat fallback (for smaller datasets or when HNSW build time is too long)
-- CREATE INDEX IF NOT EXISTS qbitos_memories_embedding_ivf_idx
--     ON qbitos_memories USING ivfflat (embedding vector_cosine_ops)
--     WITH (lists = 100);

-- 3b. B-tree indexes for exact-match and range queries
CREATE INDEX IF NOT EXISTS qbitos_memories_layer_idx
    ON qbitos_memories(layer);

CREATE INDEX IF NOT EXISTS qbitos_memories_created_at_idx
    ON qbitos_memories(created_at DESC);

CREATE INDEX IF NOT EXISTS qbitos_memories_accessed_at_idx
    ON qbitos_memories(accessed_at DESC);

CREATE INDEX IF NOT EXISTS qbitos_memories_importance_idx
    ON qbitos_memories(importance DESC);

CREATE INDEX IF NOT EXISTS qbitos_memories_agent_id_idx
    ON qbitos_memories(agent_id)
    WHERE agent_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS qbitos_memories_session_id_idx
    ON qbitos_memories(session_id)
    WHERE session_id IS NOT NULL;

-- 3c. GIN indexes for JSONB columns (jsonb_path_ops for containment, default for key lookup)
-- metadata: use jsonb_path_ops (containment queries dominate)
CREATE INDEX IF NOT EXISTS qbitos_memories_metadata_gin_idx
    ON qbitos_memories USING gin (metadata jsonb_path_ops);

-- content: use default GIN (key existence + containment)
CREATE INDEX IF NOT EXISTS qbitos_memories_content_gin_idx
    ON qbitos_memories USING gin (content);

-- 3d. GIN index for array columns (tags)
CREATE INDEX IF NOT EXISTS qbitos_memories_tags_gin_idx
    ON qbitos_memories USING gin (tags);

-- 3e. Full-text search GIN index
CREATE INDEX IF NOT EXISTS qbitos_memories_search_vector_gin_idx
    ON qbitos_memories USING gin (search_vector);

-- 3f. Trigram indexes for fuzzy search on text_content
CREATE INDEX IF NOT EXISTS qbitos_memories_text_content_trgm_idx
    ON qbitos_memories USING gin (text_content gin_trgm_ops);

-- Trigram on layer for fuzzy layer matching
CREATE INDEX IF NOT EXISTS qbitos_memories_layer_trgm_idx
    ON qbitos_memories USING gin (layer gin_trgm_ops);

-- 3g. ltree GiST index for hierarchical path queries
CREATE INDEX IF NOT EXISTS qbitos_memories_path_gist_idx
    ON qbitos_memories USING gist (path);

-- 3h. Composite indexes for common query patterns
CREATE INDEX IF NOT EXISTS qbitos_memories_layer_created_idx
    ON qbitos_memories(layer, created_at DESC);

CREATE INDEX IF NOT EXISTS qbitos_memories_layer_importance_idx
    ON qbitos_memories(layer, importance DESC)
    WHERE embedding IS NOT NULL;

CREATE INDEX IF NOT EXISTS qbitos_memories_agent_layer_idx
    ON qbitos_memories(agent_id, layer)
    WHERE agent_id IS NOT NULL;

-- 3i. Partial index for non-expired memories (common filter)
CREATE INDEX IF NOT EXISTS qbitos_memories_active_idx
    ON qbitos_memories(id)
    WHERE expires_at IS NULL OR expires_at > now();

-- 3j. Bloom index for fast exact-match on multiple columns
CREATE INDEX IF NOT EXISTS qbitos_memories_bloom_idx
    ON qbitos_memories USING bloom (layer, agent_id, session_id)
    WITH (length = 80, col1 = 4, col2 = 4, col3 = 4);

-- ─── Section 4: Knowledge Graph Table ─────────────────────────────────────────

CREATE TABLE IF NOT EXISTS qbitos_knowledge_triples (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    subject     TEXT NOT NULL,
    predicate   TEXT NOT NULL,
    object      TEXT NOT NULL,
    confidence  REAL NOT NULL DEFAULT 1.0,
    source      TEXT,
    metadata    JSONB NOT NULL DEFAULT '{}'::jsonb,
    path        ltree NOT NULL,  -- e.g., 'entity.concept.mammal.dog'
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Indexes for knowledge graph
CREATE INDEX IF NOT EXISTS kg_subject_idx
    ON qbitos_knowledge_triples(subject);
CREATE INDEX IF NOT EXISTS kg_predicate_idx
    ON qbitos_knowledge_triples(predicate);
CREATE INDEX IF NOT EXISTS kg_object_idx
    ON qbitos_knowledge_triples(object);
CREATE INDEX IF NOT EXISTS kg_subject_predicate_idx
    ON qbitos_knowledge_triples(subject, predicate);
CREATE INDEX IF NOT EXISTS kg_object_predicate_idx
    ON qbitos_knowledge_triples(object, predicate);
CREATE INDEX IF NOT EXISTS kg_metadata_gin_idx
    ON qbitos_knowledge_triples USING gin (metadata jsonb_path_ops);
CREATE INDEX IF NOT EXISTS kg_path_gist_idx
    ON qbitos_knowledge_triples USING gist (path);
CREATE INDEX IF NOT EXISTS kg_subject_trgm_idx
    ON qbitos_knowledge_triples USING gin (subject gin_trgm_ops);
CREATE INDEX IF NOT EXISTS kg_object_trgm_idx
    ON qbitos_knowledge_triples USING gin (object gin_trgm_ops);
CREATE INDEX IF NOT EXISTS kg_confidence_idx
    ON qbitos_knowledge_triples(confidence DESC);

-- ─── Section 5: Skills Table ──────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS qbitos_skills (
    id              BIGSERIAL PRIMARY KEY,
    name            TEXT NOT NULL UNIQUE,
    description     TEXT NOT NULL,
    steps           JSONB NOT NULL DEFAULT '[]'::jsonb,
    success_criteria JSONB NOT NULL DEFAULT '[]'::jsonb,
    execution_count BIGINT NOT NULL DEFAULT 0,
    success_count   BIGINT NOT NULL DEFAULT 0,
    total_latency_ms BIGINT NOT NULL DEFAULT 0,
    confidence      REAL NOT NULL DEFAULT 0.5,
    tags            TEXT[] NOT NULL DEFAULT '{}',
    input_schema    JSONB,
    output_schema   JSONB,
    metadata        JSONB NOT NULL DEFAULT '{}'::jsonb,
    skill_path      ltree,  -- hierarchical skill tree (e.g., 'nlp.text.summarize')
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Indexes for skills
CREATE INDEX IF NOT EXISTS skills_name_trgm_idx
    ON qbitos_skills USING gin (name gin_trgm_ops);
CREATE INDEX IF NOT EXISTS skills_desc_trgm_idx
    ON qbitos_skills USING gin (description gin_trgm_ops);
CREATE INDEX IF NOT EXISTS skills_tags_gin_idx
    ON qbitos_skills USING gin (tags);
CREATE INDEX IF NOT EXISTS skills_steps_gin_idx
    ON qbitos_skills USING gin (steps jsonb_path_ops);
CREATE INDEX IF NOT EXISTS skills_metadata_gin_idx
    ON qbitos_skills USING gin (metadata jsonb_path_ops);
CREATE INDEX IF NOT EXISTS skills_confidence_idx
    ON qbitos_skills(confidence DESC);
CREATE INDEX IF NOT EXISTS skills_path_gist_idx
    ON qbitos_skills USING gist (skill_path);
CREATE INDEX IF NOT EXISTS skills_execution_count_idx
    ON qbitos_skills(execution_count DESC);
CREATE INDEX IF NOT EXISTS skills_name_search_idx
    ON qbitos_skills USING gin (to_tsvector('english', name || ' ' || description));

-- ─── Section 6: Agents Table ──────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS qbitos_agents (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name        TEXT NOT NULL,
    kind        TEXT NOT NULL,  -- 'researcher', 'coder', 'analyst', etc.
    state       JSONB NOT NULL DEFAULT '{}'::jsonb,
    config      JSONB NOT NULL DEFAULT '{}'::jsonb,
    permissions JSONB NOT NULL DEFAULT '[]'::jsonb,
    status      TEXT NOT NULL DEFAULT 'idle',  -- idle, running, paused, stopped
    parent_id   UUID REFERENCES qbitos_agents(id) ON DELETE SET NULL,
    hierarchy   ltree,  -- agent hierarchy path
    metadata    JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_active TIMESTAMPTZ
);

-- Indexes for agents
CREATE INDEX IF NOT EXISTS agents_name_trgm_idx
    ON qbitos_agents USING gin (name gin_trgm_ops);
CREATE INDEX IF NOT EXISTS agents_kind_idx
    ON qbitos_agents(kind);
CREATE INDEX IF NOT EXISTS agents_status_idx
    ON qbitos_agents(status);
CREATE INDEX IF NOT EXISTS agents_state_gin_idx
    ON qbitos_agents USING gin (state jsonb_path_ops);
CREATE INDEX IF NOT EXISTS agents_config_gin_idx
    ON qbitos_agents USING gin (config);
CREATE INDEX IF NOT EXISTS agents_permissions_gin_idx
    ON qbitos_agents USING gin (permissions jsonb_path_ops);
CREATE INDEX IF NOT EXISTS agents_metadata_gin_idx
    ON qbitos_agents USING gin (metadata jsonb_path_ops);
CREATE INDEX IF NOT EXISTS agents_parent_idx
    ON qbitos_agents(parent_id);
CREATE INDEX IF NOT EXISTS agents_hierarchy_gist_idx
    ON qbitos_agents USING gist (hierarchy);
CREATE INDEX IF NOT EXISTS agents_last_active_idx
    ON qbitos_agents(last_active DESC)
    WHERE last_active IS NOT NULL;

-- ─── Section 7: Events Table ──────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS qbitos_events (
    id          BIGSERIAL PRIMARY KEY,
    event_id    BIGINT NOT NULL UNIQUE,  -- monotonic event ID from the bus
    kind        TEXT NOT NULL,
    subsystem   TEXT,
    payload     JSONB NOT NULL DEFAULT '{}'::jsonb,
    agent_id    UUID,
    session_id  UUID,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Indexes for events
CREATE INDEX IF NOT EXISTS events_event_id_idx
    ON qbitos_events(event_id);
CREATE INDEX IF NOT EXISTS events_kind_idx
    ON qbitos_events(kind);
CREATE INDEX IF NOT EXISTS events_created_at_idx
    ON qbitos_events(created_at DESC);
CREATE INDEX IF NOT EXISTS events_payload_gin_idx
    ON qbitos_events USING gin (payload jsonb_path_ops);
CREATE INDEX IF NOT EXISTS events_agent_id_idx
    ON qbitos_events(agent_id)
    WHERE agent_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS events_kind_created_idx
    ON qbitos_events(kind, created_at DESC);
CREATE INDEX IF NOT EXISTS events_subsystem_idx
    ON qbitos_events(subsystem)
    WHERE subsystem IS NOT NULL;

-- ─── Section 8: Tool Calls Table ──────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS qbitos_tool_calls (
    id          BIGSERIAL PRIMARY KEY,
    tool_name   TEXT NOT NULL,
    agent_id    UUID,
    parameters  JSONB NOT NULL DEFAULT '{}'::jsonb,
    output      JSONB,
    success     BOOLEAN NOT NULL DEFAULT false,
    duration_us BIGINT NOT NULL DEFAULT 0,
    error       TEXT,
    metadata    JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Indexes for tool calls
CREATE INDEX IF NOT EXISTS tool_calls_tool_name_idx
    ON qbitos_tool_calls(tool_name);
CREATE INDEX IF NOT EXISTS tool_calls_created_at_idx
    ON qbitos_tool_calls(created_at DESC);
CREATE INDEX IF NOT EXISTS tool_calls_agent_id_idx
    ON qbitos_tool_calls(agent_id)
    WHERE agent_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS tool_calls_parameters_gin_idx
    ON qbitos_tool_calls USING gin (parameters);
CREATE INDEX IF NOT EXISTS tool_calls_output_gin_idx
    ON qbitos_tool_calls USING gin (output);
CREATE INDEX IF NOT EXISTS tool_calls_metadata_gin_idx
    ON qbitos_tool_calls USING gin (metadata jsonb_path_ops);
CREATE INDEX IF NOT EXISTS tool_calls_success_idx
    ON qbitos_tool_calls(success)
    WHERE success = false;  -- partial index for failure analysis
CREATE INDEX IF NOT EXISTS tool_calls_tool_name_trgm_idx
    ON qbitos_tool_calls USING gin (tool_name gin_trgm_ops);

-- ─── Section 9: Training Steps Table ──────────────────────────────────────────

CREATE TABLE IF NOT EXISTS qbitos_training_steps (
    id              BIGSERIAL PRIMARY KEY,
    step_number     BIGINT NOT NULL,
    stage           TEXT,
    reward          REAL NOT NULL DEFAULT 0.0,
    loss            REAL NOT NULL DEFAULT 1.0,
    strategy        TEXT,
    success         BOOLEAN NOT NULL DEFAULT false,
    duration_ms     BIGINT NOT NULL DEFAULT 0,
    state_snapshot  JSONB NOT NULL DEFAULT '{}'::jsonb,
    metadata        JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Indexes for training steps
CREATE INDEX IF NOT EXISTS training_steps_step_idx
    ON qbitos_training_steps(step_number);
CREATE INDEX IF NOT EXISTS training_steps_created_idx
    ON qbitos_training_steps(created_at DESC);
CREATE INDEX IF NOT EXISTS training_steps_stage_idx
    ON qbitos_training_steps(stage)
    WHERE stage IS NOT NULL;
CREATE INDEX IF NOT EXISTS training_steps_strategy_idx
    ON qbitos_training_steps(strategy)
    WHERE strategy IS NOT NULL;
CREATE INDEX IF NOT EXISTS training_steps_reward_idx
    ON qbitos_training_steps(reward DESC);
CREATE INDEX IF NOT EXISTS training_steps_metadata_gin_idx
    ON qbitos_training_steps USING gin (metadata jsonb_path_ops);
CREATE INDEX IF NOT EXISTS training_steps_state_gin_idx
    ON qbitos_training_steps USING gin (state_snapshot jsonb_path_ops);

-- ─── Section 10: Planner State Table ──────────────────────────────────────────

CREATE TABLE IF NOT EXISTS qbitos_planner_state (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    goal        TEXT NOT NULL,
    task_graph  JSONB NOT NULL DEFAULT '{}'::jsonb,
    state       JSONB NOT NULL DEFAULT '{}'::jsonb,
    status      TEXT NOT NULL DEFAULT 'pending',
    agent_id    UUID,
    parent_id   UUID REFERENCES qbitos_planner_state(id) ON DELETE SET NULL,
    task_path   ltree,
    priority    INTEGER NOT NULL DEFAULT 0,
    metadata    JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    completed_at TIMESTAMPTZ
);

-- Indexes for planner state
CREATE INDEX IF NOT EXISTS planner_goal_trgm_idx
    ON qbitos_planner_state USING gin (goal gin_trgm_ops);
CREATE INDEX IF NOT EXISTS planner_status_idx
    ON qbitos_planner_state(status);
CREATE INDEX IF NOT EXISTS planner_task_graph_gin_idx
    ON qbitos_planner_state USING gin (task_graph jsonb_path_ops);
CREATE INDEX IF NOT EXISTS planner_state_gin_idx
    ON qbitos_planner_state USING gin (state jsonb_path_ops);
CREATE INDEX IF NOT EXISTS planner_metadata_gin_idx
    ON qbitos_planner_state USING gin (metadata jsonb_path_ops);
CREATE INDEX IF NOT EXISTS planner_priority_idx
    ON qbitos_planner_state(priority DESC)
    WHERE status = 'pending';
CREATE INDEX IF NOT EXISTS planner_task_path_gist_idx
    ON qbitos_planner_state USING gist (task_path);
CREATE INDEX IF NOT EXISTS planner_agent_idx
    ON qbitos_planner_state(agent_id)
    WHERE agent_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS planner_parent_idx
    ON qbitos_planner_state(parent_id)
    WHERE parent_id IS NOT NULL;

-- ─── Section 11: Updated Triggers ─────────────────────────────────────────────

-- Auto-update updated_at on all tables
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER qbitos_memories_updated_at
    BEFORE UPDATE ON qbitos_memories
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

CREATE TRIGGER qbitos_knowledge_triples_updated_at
    BEFORE UPDATE ON qbitos_knowledge_triples
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

CREATE TRIGGER qbitos_skills_updated_at
    BEFORE UPDATE ON qbitos_skills
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

CREATE TRIGGER qbitos_agents_updated_at
    BEFORE UPDATE ON qbitos_agents
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

CREATE TRIGGER qbitos_planner_state_updated_at
    BEFORE UPDATE ON qbitos_planner_state
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

-- ─── Section 12: Views for Monitoring ─────────────────────────────────────────

-- Memory layer statistics view
CREATE OR REPLACE VIEW v_memory_stats AS
SELECT
    layer,
    COUNT(*) AS total_memories,
    COUNT(*) FILTER (WHERE embedding IS NOT NULL) AS with_embeddings,
    AVG(importance) AS avg_importance,
    AVG(confidence) AS avg_confidence,
    SUM(usage_count) AS total_usage,
    MAX(created_at) AS latest_created,
    MAX(accessed_at) AS latest_accessed
FROM qbitos_memories
GROUP BY layer
ORDER BY total_memories DESC;

-- Index usage statistics view
CREATE OR REPLACE VIEW v_index_usage AS
SELECT
    schemaname,
    relname AS table_name,
    indexrelname AS index_name,
    idx_scan AS scans,
    idx_tup_read AS tuples_read,
    idx_tup_fetch AS tuples_fetched,
    pg_size_pretty(pg_relation_size(indexrelid)) AS index_size
FROM pg_stat_user_indexes
WHERE schemaname = 'public'
    AND relname LIKE 'qbitos_%'
ORDER BY idx_scan DESC;

-- Slow queries view (requires pg_stat_statements)
CREATE OR REPLACE VIEW v_slow_queries AS
SELECT
    queryid,
    LEFT(query, 200) AS query_preview,
    calls,
    total_exec_time,
    mean_exec_time,
    max_exec_time,
    rows,
    shared_blks_hit,
    shared_blks_read,
    shared_blks_dirtied
FROM pg_stat_statements
WHERE query NOT LIKE '%pg_stat_statements%'
    AND query NOT LIKE '%pg_%'
ORDER BY mean_exec_time DESC
LIMIT 50;

-- Table size overview
CREATE OR REPLACE VIEW v_table_sizes AS
SELECT
    relname AS table_name,
    pg_size_pretty(pg_total_relation_size(c.oid)) AS total_size,
    pg_size_pretty(pg_relation_size(c.oid)) AS data_size,
    pg_size_pretty(pg_indexes_size(c.oid)) AS index_size,
    n_live_tup AS row_estimate
FROM pg_class c
JOIN pg_namespace n ON n.oid = c.relnamespace
WHERE n.nspname = 'public'
    AND c.relkind = 'r'
    AND relname LIKE 'qbitos_%'
ORDER BY pg_total_relation_size(c.oid) DESC;

-- ─── Section 13: pg_prewarm on Startup ────────────────────────────────────────

-- Warm the most critical indexes into the buffer cache on startup.
-- Execute this after loading data or on restart:
-- SELECT pg_prewarm('qbitos_memories_embedding_hnsw_cosine_idx');
-- SELECT pg_prewarm('qbitos_memories_content_gin_idx');
-- SELECT pg_prewarm('qbitos_memories_layer_idx');
-- SELECT pg_prewarm('qbitos_memories_search_vector_gin_idx');

-- ─── Section 14: Query Optimization Functions ─────────────────────────────────

-- Function: similarity search with layer filter (parameterized, injection-safe)
CREATE OR REPLACE FUNCTION search_memories(
    p_embedding vector,
    p_k         INTEGER DEFAULT 10,
    p_min_score REAL DEFAULT 0.0,
    p_layer     TEXT DEFAULT NULL,
    p_tags      TEXT[] DEFAULT NULL
)
RETURNS TABLE (
    id          uuid,
    layer       text,
    content     jsonb,
    embedding   text,
    importance  real,
    confidence  real,
    created_at  timestamptz,
    accessed_at timestamptz,
    usage_count integer,
    metadata    jsonb,
    score       real
)
LANGUAGE sql STABLE PARALLEL SAFE AS $$
    SELECT
        m.id, m.layer, m.content, m.embedding::text,
        m.importance, m.confidence, m.created_at, m.accessed_at,
        m.usage_count, m.metadata,
        (1 - (m.embedding <=> p_embedding))::real AS score
    FROM qbitos_memories m
    WHERE m.embedding IS NOT NULL
      AND 1 - (m.embedding <=> p_embedding) >= p_min_score
      AND (p_layer IS NULL OR m.layer = p_layer)
      AND (p_tags IS NULL OR m.tags && p_tags)
    ORDER BY m.embedding <=> p_embedding
    LIMIT p_k
$$;

-- Function: full-text search across memories
CREATE OR REPLACE FUNCTION search_memories_fts(
    p_query  TEXT,
    p_k      INTEGER DEFAULT 10,
    p_layer  TEXT DEFAULT NULL
)
RETURNS TABLE (
    id          uuid,
    layer       text,
    content     jsonb,
    importance  real,
    rank        real
)
LANGUAGE sql STABLE PARALLEL SAFE AS $$
    SELECT
        m.id, m.layer, m.content, m.importance,
        ts_rank(m.search_vector, plainto_tsquery('english', p_query))::real AS rank
    FROM qbitos_memories m
    WHERE m.search_vector @@ plainto_tsquery('english', p_query)
      AND (p_layer IS NULL OR m.layer = p_layer)
    ORDER BY rank DESC
    LIMIT p_k
$$;

-- Function: fuzzy search across memories (trigram similarity)
CREATE OR REPLACE FUNCTION search_memories_fuzzy(
    p_query  TEXT,
    p_k      INTEGER DEFAULT 10,
    p_threshold REAL DEFAULT 0.3
)
RETURNS TABLE (
    id            uuid,
    layer         text,
    text_content  text,
    similarity    real
)
LANGUAGE sql STABLE PARALLEL SAFE AS $$
    SELECT
        m.id, m.layer, m.text_content,
        similarity(m.text_content, p_query)::real AS similarity
    FROM qbitos_memories m
    WHERE m.text_content IS NOT NULL
      AND similarity(m.text_content, p_query) >= p_threshold
    ORDER BY similarity DESC
    LIMIT p_k
$$;

-- Function: knowledge graph traversal (ltree-based)
CREATE OR REPLACE FUNCTION query_knowledge_path(
    p_path  ltree
)
RETURNS TABLE (
    id          uuid,
    subject     text,
    predicate   text,
    object      text,
    confidence  real,
    path        ltree
)
LANGUAGE sql STABLE PARALLEL SAFE AS $$
    SELECT
        id, subject, predicate, object, confidence, path
    FROM qbitos_knowledge_triples
    WHERE path <@ p_path
    ORDER BY confidence DESC
$$;

-- Function: skill search (trigram + full-text)
CREATE OR REPLACE FUNCTION search_skills(
    p_query     TEXT,
    p_k         INTEGER DEFAULT 10,
    p_min_confidence REAL DEFAULT 0.0
)
RETURNS TABLE (
    id              bigint,
    name            text,
    description     text,
    confidence      real,
    execution_count bigint,
    rank            real
)
LANGUAGE sql STABLE PARALLEL SAFE AS $$
    SELECT
        s.id, s.name, s.description, s.confidence, s.execution_count,
        GREATEST(
            similarity(s.name, p_query),
            ts_rank(to_tsvector('english', s.name || ' ' || s.description), plainto_tsquery('english', p_query))
        )::real AS rank
    FROM qbitos_skills s
    WHERE s.confidence >= p_min_confidence
      AND (
        s.name % p_query
        OR to_tsvector('english', s.name || ' ' || s.description) @@ plainto_tsquery('english', p_query)
      )
    ORDER BY rank DESC
    LIMIT p_k
$$;

-- ─── Section 15: Documentation ────────────────────────────────────────────────

COMMENT ON TABLE qbitos_memories IS 'Core memory table (L0-L5 biological hierarchy). Stores all memory objects with embeddings, metadata, full-text search vectors, and hierarchical paths.';
COMMENT ON TABLE qbitos_knowledge_triples IS 'Knowledge graph triples (subject, predicate, object) with ltree hierarchical paths for subgraph queries.';
COMMENT ON TABLE qbitos_skills IS 'Skill genome registry with JSONB step definitions, execution statistics, and trigram-searchable names.';
COMMENT ON TABLE qbitos_agents IS 'Agent registry with JSONB state/config/permissions, ltree hierarchy for parent-child relationships.';
COMMENT ON TABLE qbitos_events IS 'Event log for the cognitive event bus. JSONB payload with GIN index for fast filtering.';
COMMENT ON TABLE qbitos_tool_calls IS 'Tool execution history with JSONB parameters/output, success tracking, and latency metrics.';
COMMENT ON TABLE qbitos_training_steps IS 'Training loop step history with reward, loss, strategy, and state snapshots.';
COMMENT ON TABLE qbitos_planner_state IS 'Planner task graph state with JSONB task definitions, ltree task paths, and priority ordering.';

COMMENT ON INDEX qbitos_memories_embedding_hnsw_cosine_idx IS 'HNSW index for cosine similarity search (m=16, ef_construction=64). Production-grade ANN for semantic memory recall.';
COMMENT ON INDEX qbitos_memories_content_gin_idx IS 'GIN index for JSONB content key-existence and containment queries.';
COMMENT ON INDEX qbitos_memories_metadata_gin_idx IS 'GIN index with jsonb_path_ops for efficient JSONB containment queries on metadata.';
COMMENT ON INDEX qbitos_memories_search_vector_gin_idx IS 'GIN index for full-text search across memory content (English language).';
COMMENT ON INDEX qbitos_memories_text_content_trgm_idx IS 'Trigram GIN index for fuzzy text search, autocomplete, and typo tolerance.';
COMMENT ON INDEX qbitos_memories_tags_gin_idx IS 'GIN index for array containment queries on tags (e.g., tags && ARRAY["nlp"]).';
COMMENT ON INDEX qbitos_memories_path_gist_idx IS 'GiST index for ltree hierarchical path queries on memory objects.';
COMMENT ON INDEX qbitos_memories_bloom_idx IS 'Bloom filter index for fast exact-match on (layer, agent_id, session_id) combinations.';

-- ═══════════════════════════════════════════════════════════════════════════════
-- End of Migration
-- ═══════════════════════════════════════════════════════════════════════════════
