-- ═══════════════════════════════════════════════════════════════════════════════
-- QbitOS Full Production Schema — PostgreSQL with pgvector
-- ═══════════════════════════════════════════════════════════════════════════════
-- This file creates the complete database schema for QbitOS.
-- It should be run once on a fresh PostgreSQL 16+ instance with pgvector extension.

-- ─── Extensions ───────────────────────────────────────────────────────────────
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS btree_gin;
CREATE EXTENSION IF NOT EXISTS btree_gist;
CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE EXTENSION IF NOT EXISTS unaccent;
CREATE EXTENSION IF NOT EXISTS fuzzystrmatch;
CREATE EXTENSION IF NOT EXISTS hstore;
CREATE EXTENSION IF NOT EXISTS ltree;
CREATE EXTENSION IF NOT EXISTS bloom;

-- Monitoring extensions (best-effort — may fail on restricted environments)
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;
CREATE EXTENSION IF NOT EXISTS pg_buffercache;
CREATE EXTENSION IF NOT EXISTS pg_visibility;
CREATE EXTENSION IF NOT EXISTS pg_prewarm;

-- ─── 1. Memories Table ────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS qbitos_memories (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    layer           TEXT NOT NULL,
    content         JSONB NOT NULL,
    embedding       vector(384),
    importance      REAL NOT NULL DEFAULT 0.5,
    confidence      REAL NOT NULL DEFAULT 0.5,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    accessed_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    usage_count     INTEGER NOT NULL DEFAULT 0,
    metadata        JSONB NOT NULL DEFAULT '{}'::jsonb,
    tags            TEXT[] NOT NULL DEFAULT '{}',
    agent_id        UUID,
    session_id      UUID,
    text_content    TEXT GENERATED ALWAYS AS (content::text) STORED,
    search_vector   TSVECTOR GENERATED ALWAYS AS (
                        to_tsvector('english', unaccent(coalesce(content::text, '')))
                    ) STORED,
    path            LTREE,
    expires_at      TIMESTAMPTZ,
    version         INTEGER NOT NULL DEFAULT 1,
    checksum        TEXT
);

-- HNSW vector indexes (production-grade ANN)
CREATE INDEX IF NOT EXISTS qbitos_memories_embedding_hnsw_cosine_idx
    ON qbitos_memories USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 200);

CREATE INDEX IF NOT EXISTS qbitos_memories_embedding_hnsw_l2_idx
    ON qbitos_memories USING hnsw (embedding vector_l2_ops)
    WITH (m = 16, ef_construction = 200);

CREATE INDEX IF NOT EXISTS qbitos_memories_embedding_hnsw_ip_idx
    ON qbitos_memories USING hnsw (embedding vector_ip_ops)
    WITH (m = 16, ef_construction = 200);

-- GIN indexes for JSONB and full-text search
CREATE INDEX IF NOT EXISTS qbitos_memories_content_gin_idx
    ON qbitos_memories USING gin (content jsonb_path_ops);

CREATE INDEX IF NOT EXISTS qbitos_memories_metadata_gin_idx
    ON qbitos_memories USING gin (metadata jsonb_path_ops);

CREATE INDEX IF NOT EXISTS qbitos_memories_tags_gin_idx
    ON qbitos_memories USING gin (tags);

CREATE INDEX IF NOT EXISTS qbitos_memories_search_vector_gin_idx
    ON qbitos_memories USING gin (search_vector);

-- Trigram indexes for fuzzy search
CREATE INDEX IF NOT EXISTS qbitos_memories_text_trgm_idx
    ON qbitos_memories USING gin (text_content gin_trgm_ops);

CREATE INDEX IF NOT EXISTS qbitos_memories_layer_trgm_idx
    ON qbitos_memories USING gin (layer gin_trgm_ops);

-- ltree GiST index for hierarchical paths
CREATE INDEX IF NOT EXISTS qbitos_memories_path_gist_idx
    ON qbitos_memories USING gist (path);

-- Bloom filter index for multi-column exact-match
CREATE INDEX IF NOT EXISTS qbitos_memories_bloom_idx
    ON qbitos_memories USING bloom (layer, agent_id, session_id);

-- Composite indexes for common query patterns
CREATE INDEX IF NOT EXISTS qbitos_memories_layer_importance_idx
    ON qbitos_memories (layer, importance DESC)
    WHERE embedding IS NOT NULL;

CREATE INDEX IF NOT EXISTS qbitos_memories_created_at_idx
    ON qbitos_memories (created_at DESC);

CREATE INDEX IF NOT EXISTS qbitos_memories_accessed_at_idx
    ON qbitos_memories (accessed_at DESC);

CREATE INDEX IF NOT EXISTS qbitos_memories_agent_idx
    ON qbitos_memories (agent_id)
    WHERE agent_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS qbitos_memories_session_idx
    ON qbitos_memories (session_id)
    WHERE session_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS qbitos_memories_expires_idx
    ON qbitos_memories (expires_at)
    WHERE expires_at IS NOT NULL;

-- ─── 2. Knowledge Graph Triples ───────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS qbitos_knowledge_triples (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    subject     TEXT NOT NULL,
    predicate   TEXT NOT NULL,
    object      TEXT NOT NULL,
    confidence  REAL NOT NULL DEFAULT 0.5,
    source      TEXT,
    metadata    JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    triple_path LTREE
);

CREATE INDEX IF NOT EXISTS qbitos_triples_subject_trgm_idx
    ON qbitos_knowledge_triples USING gin (subject gin_trgm_ops);

CREATE INDEX IF NOT EXISTS qbitos_triples_predicate_trgm_idx
    ON qbitos_knowledge_triples USING gin (predicate gin_trgm_ops);

CREATE INDEX IF NOT EXISTS qbitos_triples_object_trgm_idx
    ON qbitos_knowledge_triples USING gin (object gin_trgm_ops);

CREATE INDEX IF NOT EXISTS qbitos_triples_subject_idx
    ON qbitos_knowledge_triples (subject);

CREATE INDEX IF NOT EXISTS qbitos_triples_object_idx
    ON qbitos_knowledge_triples (object);

CREATE INDEX IF NOT EXISTS qbitos_triples_confidence_idx
    ON qbitos_knowledge_triples (confidence DESC);

CREATE INDEX IF NOT EXISTS qbitos_triples_metadata_gin_idx
    ON qbitos_knowledge_triples USING gin (metadata jsonb_path_ops);

CREATE INDEX IF NOT EXISTS qbitos_triples_path_gist_idx
    ON qbitos_knowledge_triples USING gist (triple_path);

-- ─── 3. Skills ────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS qbitos_skills (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name            TEXT NOT NULL UNIQUE,
    description     TEXT NOT NULL,
    steps           JSONB NOT NULL DEFAULT '[]'::jsonb,
    success_criteria TEXT[] NOT NULL DEFAULT '{}',
    tags            TEXT[] NOT NULL DEFAULT '{}',
    signature       JSONB,
    execution_count BIGINT NOT NULL DEFAULT 0,
    success_count   BIGINT NOT NULL DEFAULT 0,
    total_latency_us BIGINT NOT NULL DEFAULT 0,
    confidence      REAL NOT NULL DEFAULT 0.5,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_executed_at TIMESTAMPTZ,
    skill_path      LTREE,
    metadata        JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS qbitos_skills_name_trgm_idx
    ON qbitos_skills USING gin (name gin_trgm_ops);

CREATE INDEX IF NOT EXISTS qbitos_skills_desc_trgm_idx
    ON qbitos_skills USING gin (description gin_trgm_ops);

CREATE INDEX IF NOT EXISTS qbitos_skills_tags_gin_idx
    ON qbitos_skills USING gin (tags);

CREATE INDEX IF NOT EXISTS qbitos_skills_steps_gin_idx
    ON qbitos_skills USING gin (steps jsonb_path_ops);

CREATE INDEX IF NOT EXISTS qbitos_skills_metadata_gin_idx
    ON qbitos_skills USING gin (metadata jsonb_path_ops);

CREATE INDEX IF NOT EXISTS qbitos_skills_confidence_idx
    ON qbitos_skills (confidence DESC);

CREATE INDEX IF NOT EXISTS qbitos_skills_path_gist_idx
    ON qbitos_skills USING gist (skill_path);

CREATE INDEX IF NOT EXISTS qbitos_skills_execution_count_idx
    ON qbitos_skills (execution_count DESC);

-- ─── 4. Agents ────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS qbitos_agents (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name            TEXT NOT NULL UNIQUE,
    description     TEXT,
    state           JSONB NOT NULL DEFAULT '{}'::jsonb,
    config          JSONB NOT NULL DEFAULT '{}'::jsonb,
    permissions     JSONB NOT NULL DEFAULT '[]'::jsonb,
    agent_path      LTREE,
    status          TEXT NOT NULL DEFAULT 'idle',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS qbitos_agents_name_trgm_idx
    ON qbitos_agents USING gin (name gin_trgm_ops);

CREATE INDEX IF NOT EXISTS qbitos_agents_state_gin_idx
    ON qbitos_agents USING gin (state jsonb_path_ops);

CREATE INDEX IF NOT EXISTS qbitos_agents_config_gin_idx
    ON qbitos_agents USING gin (config jsonb_path_ops);

CREATE INDEX IF NOT EXISTS qbitos_agents_permissions_gin_idx
    ON qbitos_agents USING gin (permissions jsonb_path_ops);

CREATE INDEX IF NOT EXISTS qbitos_agents_path_gist_idx
    ON qbitos_agents USING gist (agent_path);

CREATE INDEX IF NOT EXISTS qbitos_agents_status_idx
    ON qbitos_agents (status);

-- ─── 5. Events ────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS qbitos_events (
    id          BIGSERIAL PRIMARY KEY,
    timestamp   TIMESTAMPTZ NOT NULL DEFAULT now(),
    kind        TEXT NOT NULL,
    subsystem   TEXT NOT NULL,
    payload     JSONB NOT NULL DEFAULT '{}'::jsonb,
    agent_id    UUID REFERENCES qbitos_agents(id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS qbitos_events_kind_idx
    ON qbitos_events (kind);

CREATE INDEX IF NOT EXISTS qbitos_events_subsystem_idx
    ON qbitos_events (subsystem);

CREATE INDEX IF NOT EXISTS qbitos_events_timestamp_idx
    ON qbitos_events (timestamp DESC);

CREATE INDEX IF NOT EXISTS qbitos_events_payload_gin_idx
    ON qbitos_events USING gin (payload jsonb_path_ops);

CREATE INDEX IF NOT EXISTS qbitos_events_agent_idx
    ON qbitos_events (agent_id)
    WHERE agent_id IS NOT NULL;

-- ─── 6. Tool Calls ────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS qbitos_tool_calls (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tool_name   TEXT NOT NULL,
    parameters  JSONB NOT NULL DEFAULT '{}'::jsonb,
    output      JSONB,
    success     BOOLEAN NOT NULL DEFAULT false,
    error       TEXT,
    agent_id    UUID REFERENCES qbitos_agents(id) ON DELETE SET NULL,
    duration_us BIGINT NOT NULL DEFAULT 0,
    called_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS qbitos_tool_calls_tool_idx
    ON qbitos_tool_calls (tool_name);

CREATE INDEX IF NOT EXISTS qbitos_tool_calls_called_at_idx
    ON qbitos_tool_calls (called_at DESC);

CREATE INDEX IF NOT EXISTS qbitos_tool_calls_parameters_gin_idx
    ON qbitos_tool_calls USING gin (parameters jsonb_path_ops);

CREATE INDEX IF NOT EXISTS qbitos_tool_calls_output_gin_idx
    ON qbitos_tool_calls USING gin (output jsonb_path_ops);

CREATE INDEX IF NOT EXISTS qbitos_tool_calls_agent_idx
    ON qbitos_tool_calls (agent_id)
    WHERE agent_id IS NOT NULL;

-- Failure-only index for circuit breaker analysis
CREATE INDEX IF NOT EXISTS qbitos_tool_calls_failed_idx
    ON qbitos_tool_calls (tool_name, called_at DESC)
    WHERE success = false;

-- ─── 7. Training Steps ────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS qbitos_training_steps (
    id              BIGSERIAL PRIMARY KEY,
    step            INTEGER NOT NULL,
    stage           TEXT,
    difficulty      TEXT,
    reward          DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    loss            DOUBLE PRECISION NOT NULL DEFAULT 1.0,
    success         BOOLEAN NOT NULL DEFAULT false,
    strategy        TEXT,
    duration_ms     BIGINT NOT NULL DEFAULT 0,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS qbitos_training_step_idx
    ON qbitos_training_steps (step);

CREATE INDEX IF NOT EXISTS qbitos_training_reward_idx
    ON qbitos_training_steps (reward DESC);

CREATE INDEX IF NOT EXISTS qbitos_training_strategy_idx
    ON qbitos_training_steps (strategy)
    WHERE strategy IS NOT NULL;

CREATE INDEX IF NOT EXISTS qbitos_training_created_idx
    ON qbitos_training_steps (created_at DESC);

-- ─── 8. Planner State ─────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS qbitos_planner_state (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    goal            JSONB NOT NULL,
    task_graph      JSONB NOT NULL DEFAULT '{}'::jsonb,
    priority        INTEGER NOT NULL DEFAULT 0,
    status          TEXT NOT NULL DEFAULT 'pending',
    task_path       LTREE,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS qbitos_planner_status_idx
    ON qbitos_planner_state (status);

CREATE INDEX IF NOT EXISTS qbitos_planner_priority_idx
    ON qbitos_planner_state (priority DESC)
    WHERE status = 'pending';

CREATE INDEX IF NOT EXISTS qbitos_planner_goal_gin_idx
    ON qbitos_planner_state USING gin (goal jsonb_path_ops);

CREATE INDEX IF NOT EXISTS qbitos_planner_task_path_gist_idx
    ON qbitos_planner_state USING gist (task_path);

-- ─── 9. Experience Replay ─────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS qbitos_experiences (
    id          BIGSERIAL PRIMARY KEY,
    state       JSONB NOT NULL,
    action      JSONB NOT NULL,
    reward      DOUBLE PRECISION NOT NULL,
    next_state  JSONB NOT NULL,
    terminal    BOOLEAN NOT NULL DEFAULT false,
    priority    DOUBLE PRECISION NOT NULL DEFAULT 1.0,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS qbitos_experiences_priority_idx
    ON qbitos_experiences (priority DESC);

CREATE INDEX IFNOT EXISTS qbitos_experiences_created_idx
    ON qbitos_experiences (created_at DESC);

CREATE INDEX IF NOT EXISTS qbitos_experiences_reward_idx
    ON qbitos_experiences (reward DESC);

-- ─── 10. Reflections ──────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS qbitos_reflections (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    question        TEXT NOT NULL,
    failed_answer   TEXT,
    failure_reason  TEXT,
    correct_approach TEXT,
    lesson          TEXT NOT NULL,
    tags            TEXT[] NOT NULL DEFAULT '{}',
    retrieval_count INTEGER NOT NULL DEFAULT 0,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS qbitos_reflections_question_trgm_idx
    ON qbitos_reflections USING gin (question gin_trgm_ops);

CREATE INDEX IF NOT EXISTS qbitos_reflections_lesson_trgm_idx
    ON qbitos_reflections USING gin (lesson gin_trgm_ops);

CREATE INDEX IF NOT EXISTS qbitos_reflections_tags_gin_idx
    ON qbitos_reflections USING gin (tags);

-- ─── 11. Failure Clusters ─────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS qbitos_failure_clusters (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    root_cause      TEXT NOT NULL,
    error_pattern   TEXT NOT NULL,
    frequency       INTEGER NOT NULL DEFAULT 1,
    suggested_fix   TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS qbitos_failure_clusters_root_cause_idx
    ON qbitos_failure_clusters (root_cause);

CREATE INDEX IF NOT EXISTS qbitos_failure_clusters_pattern_trgm_idx
    ON qbitos_failure_clusters USING gin (error_pattern gin_trgm_ops);

-- ─── 12. Causal Chains ────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS qbitos_causal_chains (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    cause           TEXT NOT NULL,
    intermediate    TEXT NOT NULL,
    effect          TEXT NOT NULL,
    confidence      DOUBLE PRECISION NOT NULL DEFAULT 0.5,
    evidence_count  INTEGER NOT NULL DEFAULT 1,
    last_observed   TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS qbitos_causal_cause_idx
    ON qbitos_causal_chains (cause);

CREATE INDEX IF NOT EXISTS qbitos_causal_effect_idx
    ON qbitos_causal_chains (effect);

CREATE INDEX IF NOT EXISTS qbitos_causal_confidence_idx
    ON qbitos_causal_chains (confidence DESC);

-- ─── 13. World Model Predictions ──────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS qbitos_world_model (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    state_hash          TEXT NOT NULL,
    action              TEXT NOT NULL,
    predicted_next      JSONB NOT NULL,
    actual_next         JSONB,
    prediction_error    DOUBLE PRECISION,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS qbitos_world_model_state_action_idx
    ON qbitos_world_model (state_hash, action);

CREATE INDEX IF NOT EXISTS qbitos_world_model_error_idx
    ON qbitos_world_model (prediction_error DESC)
    WHERE prediction_error IS NOT NULL;

-- ─── 14. Training History ─────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS qbitos_training_history (
    id              BIGSERIAL PRIMARY KEY,
    round           INTEGER NOT NULL,
    method          TEXT NOT NULL,
    loss            DOUBLE PRECISION NOT NULL,
    reward          DOUBLE PRECISION NOT NULL,
    accuracy        DOUBLE PRECISION,
    config          JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS qbitos_training_history_round_idx
    ON qbitos_training_history (round);

CREATE INDEX IF NOT EXISTS qbitos_training_history_method_idx
    ON qbitos_training_history (method);

CREATE INDEX IF NOT EXISTS qbitos_training_history_created_idx
    ON qbitos_training_history (created_at DESC);

-- ─── Triggers ─────────────────────────────────────────────────────────────────
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_memories_updated_at
    BEFORE UPDATE ON qbitos_memories
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

CREATE TRIGGER update_agents_updated_at
    BEFORE UPDATE ON qbitos_agents
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

CREATE TRIGGER update_skills_updated_at
    BEFORE UPDATE ON qbitos_skills
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

CREATE TRIGGER update_planner_updated_at
    BEFORE UPDATE ON qbitos_planner_state
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

CREATE TRIGGER update_failure_clusters_updated_at
    BEFORE UPDATE ON qbitos_failure_clusters
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

-- ─── Monitoring Views ─────────────────────────────────────────────────────────
CREATE OR REPLACE VIEW v_memory_stats AS
SELECT
    layer,
    COUNT(*) AS count,
    AVG(importance) AS avg_importance,
    AVG(confidence) AS avg_confidence,
    MAX(created_at) AS newest,
    MIN(created_at) AS oldest
FROM qbitos_memories
GROUP BY layer
ORDER BY count DESC;

CREATE OR REPLACE VIEW v_index_usage AS
SELECT
    schemaname,
    relname,
    indexrelname,
    idx_scan,
    idx_tup_read,
    idx_tup_fetch,
    pg_size_pretty(pg_relation_size(indexrelid)) AS size
FROM pg_stat_user_indexes
ORDER BY idx_scan DESC;

CREATE OR REPLACE VIEW v_slow_queries AS
SELECT
    query,
    calls,
    total_exec_time,
    mean_exec_time,
    rows
FROM pg_stat_statements
WHERE mean_exec_time > 100
ORDER BY mean_exec_time DESC
LIMIT 20;

CREATE OR REPLACE VIEW v_table_sizes AS
SELECT
    relname AS table_name,
    pg_size_pretty(pg_total_relation_size(relid)) AS total_size,
    pg_size_pretty(pg_relation_size(relid)) AS table_size,
    pg_size_pretty(pg_indexes_size(relid)) AS index_size,
    n_live_tup AS row_estimate
FROM pg_stat_user_tables
ORDER BY pg_total_relation_size(relid) DESC;

-- ─── Server-Side Functions ────────────────────────────────────────────────────
CREATE OR REPLACE FUNCTION search_memories(
    p_embedding vector(384),
    p_limit INTEGER DEFAULT 10,
    p_min_score REAL DEFAULT 0.0,
    p_layer TEXT DEFAULT NULL
)
RETURNS TABLE (
    id UUID,
    layer TEXT,
    content JSONB,
    embedding vector(384),
    importance REAL,
    confidence REAL,
    created_at TIMESTAMPTZ,
    accessed_at TIMESTAMPTZ,
    usage_count INTEGER,
    metadata JSONB,
    score REAL
)
LANGUAGE sql STABLE AS $$
    SELECT
        m.id, m.layer, m.content, m.embedding, m.importance, m.confidence,
        m.created_at, m.accessed_at, m.usage_count, m.metadata,
        (1 - (m.embedding <=> p_embedding))::REAL AS score
    FROM qbitos_memories m
    WHERE m.embedding IS NOT NULL
      AND (p_layer IS NULL OR m.layer = p_layer)
      AND 1 - (m.embedding <=> p_embedding) >= p_min_score
    ORDER BY m.embedding <=> p_embedding
    LIMIT p_limit;
$$;

CREATE OR REPLACE FUNCTION search_memories_fts(
    p_query TEXT,
    p_limit INTEGER DEFAULT 10
)
RETURNS TABLE (
    id UUID,
    layer TEXT,
    content JSONB,
    ts_rank REAL
)
LANGUAGE sql STABLE AS $$
    SELECT
        m.id, m.layer, m.content,
        ts_rank(m.search_vector, plainto_tsquery('english', unaccent(p_query))) AS rank
    FROM qbitos_memories m
    WHERE m.search_vector @@ plainto_tsquery('english', unaccent(p_query))
    ORDER BY rank DESC
    LIMIT p_limit;
$$;

CREATE OR REPLACE FUNCTION search_memories_fuzzy(
    p_query TEXT,
    p_limit INTEGER DEFAULT 10
)
RETURNS TABLE (
    id UUID,
    layer TEXT,
    content JSONB,
    similarity REAL
)
LANGUAGE sql STABLE AS $$
    SELECT
        m.id, m.layer, m.content,
        similarity(m.text_content, p_query) AS sim
    FROM qbitos_memories m
    WHERE m.text_content % p_query
    ORDER BY sim DESC
    LIMIT p_limit;
$$;

CREATE OR REPLACE FUNCTION query_knowledge_path(
    p_path LTREE,
    p_limit INTEGER DEFAULT 100
)
RETURNS TABLE (
    id UUID,
    subject TEXT,
    predicate TEXT,
    object TEXT,
    confidence REAL,
    triple_path LTREE
)
LANGUAGE sql STABLE AS $$
    SELECT
        t.id, t.subject, t.predicate, t.object, t.confidence, t.triple_path
    FROM qbitos_knowledge_triples t
    WHERE t.triple_path <@ p_path
    ORDER BY t.confidence DESC
    LIMIT p_limit;
$$;

CREATE OR REPLACE FUNCTION search_skills(
    p_query TEXT,
    p_limit INTEGER DEFAULT 10
)
RETURNS TABLE (
    id UUID,
    name TEXT,
    description TEXT,
    confidence REAL,
    execution_count BIGINT
)
LANGUAGE sql STABLE AS $$
    SELECT
        s.id, s.name, s.description, s.confidence, s.execution_count
    FROM qbitos_skills s
    WHERE s.name % p_query OR s.description % p_query
    ORDER BY (similarity(s.name, p_query) + similarity(s.description, p_query)) DESC
    LIMIT p_limit;
$$;

-- ─── pg_prewarm Configuration ─────────────────────────────────────────────────
-- Warm critical indexes on startup (uncomment after first data load):
-- SELECT pg_prewarm('qbitos_memories_embedding_hnsw_cosine_idx');
-- SELECT pg_prewarm('qbitos_memories_content_gin_idx');
-- SELECT pg_prewarm('qbitos_memories_layer_idx');
-- SELECT pg_prewarm('qbitos_memories_search_vector_gin_idx');

-- ─── Set HNSW ef_search for recall/speed tradeoff ─────────────────────────────
SET hnsw.ef_search = 40;

-- ═══════════════════════════════════════════════════════════════════════════════
-- End of Schema
-- ═══════════════════════════════════════════════════════════════════════════════
