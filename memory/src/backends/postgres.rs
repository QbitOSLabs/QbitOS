//! PostgreSQL backend (requires `postgres` feature).
//!
//! Production-grade storage layer with:
//!   • HNSW vector indexes (cosine, L2, inner product)
//!   • GIN indexes on all JSONB columns (jsonb_path_ops for containment)
//!   • Full-text search via generated tsvector columns
//!   • Trigram fuzzy search via pg_trgm
//!   • Hierarchical data via ltree
//!   • Bloom filter indexes for exact-match acceleration
//!   • 14 PostgreSQL extensions
//!   • Parameterized queries (SQL injection safe)
//!   • Server-side functions for common query patterns
//!
//! # Schema
//!
//! See `migrations/002_production_indexes.sql` for the full migration.

use async_trait::async_trait;
use sqlx::postgres::{PgPool, PgPoolOptions};

use crate::backends::{BackendError, MemoryBackend, SimilarityQuery, SimilarityResult};
use crate::embedding::Embedding;
use crate::object::{Importance, MemoryId, MemoryObject, MemoryType};

/// PostgreSQL backend configuration.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PostgresConfig {
    /// Connection URL (e.g. `postgres://user:pass@localhost:5432/qbitos`).
    pub url: String,
    /// Max connections in the pool.
    pub max_connections: u32,
    /// Connect timeout in seconds.
    pub connect_timeout_secs: u64,
    /// Whether to auto-run migrations on connect.
    pub auto_migrate: bool,
    /// Embedding dimension (must match the `vector(N)` column).
    pub embedding_dim: usize,
    /// HNSW ef_search parameter (controls recall vs speed at query time).
    pub hnsw_ef_search: i32,
    /// Whether to enable pg_prewarm on startup.
    pub prewarm_indexes: bool,
}

impl Default for PostgresConfig {
    fn default() -> Self {
        Self {
            url: std::env::var("DATABASE_URL")
                .unwrap_or_else(|_| "postgres://localhost/qbitos".into()),
            max_connections: std::env::var("QBITOS_PG_MAX_CONNECTIONS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(16),
            connect_timeout_secs: std::env::var("QBITOS_PG_CONNECT_TIMEOUT")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(10),
            auto_migrate: true,
            embedding_dim: std::env::var("QBITOS_EMBEDDING_DIM")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(384),
            hnsw_ef_search: std::env::var("QBITOS_HNSW_EF_SEARCH")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(40),
            prewarm_indexes: false,
        }
    }
}

/// PostgreSQL backend.
pub struct PostgresBackend {
    pool: PgPool,
    config: PostgresConfig,
}

impl PostgresBackend {
    /// Create a new backend, connecting to Postgres and running migrations.
    ///
    /// # Errors
    /// Returns [`BackendError`] if the connection or migration fails.
    pub async fn new(config: PostgresConfig) -> Result<Self, BackendError> {
        let pool = PgPoolOptions::new()
            .max_connections(config.max_connections)
            .acquire_timeout(std::time::Duration::from_secs(config.connect_timeout_secs))
            .connect(&config.url)
            .await
            .map_err(|e| BackendError::Connection(e.to_string()))?;

        let backend = Self {
            pool,
            config: config.clone(),
        };
        if backend.config.auto_migrate {
            backend.migrate().await?;
        }

        // Set HNSW ef_search for better recall.
        let _ = sqlx::query(&format!(
            "SET LOCAL hnsw.ef_search = {}",
            backend.config.hnsw_ef_search
        ))
        .execute(&backend.pool)
        .await;

        // Optionally prewarm critical indexes.
        if backend.config.prewarm_indexes {
            backend.prewarm().await;
        }

        Ok(backend)
    }

    /// Run schema migrations.
    async fn migrate(&self) -> Result<(), BackendError> {
        // ─── Extensions ─────────────────────────────────────────────────
        let extensions = [
            "vector",
            "pg_trgm",
            "btree_gin",
            "btree_gist",
            "pgcrypto",
            "unaccent",
            "fuzzystrmatch",
            "hstore",
            "ltree",
            "bloom",
        ];

        for ext in &extensions {
            sqlx::query(&format!("CREATE EXTENSION IF NOT EXISTS {ext}"))
                .execute(&self.pool)
                .await
                .map_err(|e| BackendError::Connection(format!("extension {ext}: {e}")))?;
        }

        // Monitoring extensions (may fail on restricted environments — that's OK).
        for ext in &["pg_stat_statements", "pg_buffercache", "pg_visibility", "pg_prewarm"] {
            let _ = sqlx::query(&format!("CREATE EXTENSION IF NOT EXISTS {ext}"))
                .execute(&self.pool)
                .await;
        }

        // ─── Core table (backward compatible ALTER) ─────────────────────
        let dim = self.config.embedding_dim as i32;

        sqlx::query(&format!(
            r#"CREATE TABLE IF NOT EXISTS qbitos_memories (
                id           UUID PRIMARY KEY,
                layer        TEXT NOT NULL,
                content      JSONB NOT NULL,
                embedding    vector({dim}),
                importance   REAL NOT NULL DEFAULT 0.5,
                confidence   REAL NOT NULL DEFAULT 1.0,
                created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
                accessed_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
                usage_count  INTEGER NOT NULL DEFAULT 0,
                metadata     JSONB NOT NULL DEFAULT '{{}}'::jsonb
            )"#
        ))
        .execute(&self.pool)
        .await
        .map_err(|e| BackendError::Connection(e.to_string()))?;

        // Add new columns (idempotent).
        for stmt in [
            "ALTER TABLE qbitos_memories ADD COLUMN IF NOT EXISTS tags TEXT[] NOT NULL DEFAULT '{}'",
            "ALTER TABLE qbitos_memories ADD COLUMN IF NOT EXISTS agent_id UUID",
            "ALTER TABLE qbitos_memories ADD COLUMN IF NOT EXISTS session_id UUID",
            "ALTER TABLE qbitos_memories ADD COLUMN IF NOT EXISTS text_content TEXT",
            "ALTER TABLE qbitos_memories ADD COLUMN IF NOT EXISTS search_vector tsvector",
            "ALTER TABLE qbitos_memories ADD COLUMN IF NOT EXISTS path ltree",
            "ALTER TABLE qbitos_memories ADD COLUMN IF NOT EXISTS expires_at TIMESTAMPTZ",
            "ALTER TABLE qbitos_memories ADD COLUMN IF NOT EXISTS version INTEGER NOT NULL DEFAULT 1",
            "ALTER TABLE qbitos_memories ADD COLUMN IF NOT EXISTS checksum TEXT",
        ] {
            let _ = sqlx::query(stmt).execute(&self.pool).await;
        }

        // ─── Drop old IVFFlat index ─────────────────────────────────────
        let _ = sqlx::query("DROP INDEX IF EXISTS qbitos_memories_embedding_idx")
            .execute(&self.pool)
            .await;

        // ─── HNSW vector indexes (production-grade ANN) ─────────────────
        let _ = sqlx::query(
            r#"CREATE INDEX IF NOT EXISTS qbitos_memories_embedding_hnsw_cosine_idx
               ON qbitos_memories USING hnsw (embedding vector_cosine_ops)
               WITH (m = 16, ef_construction = 64)"#,
        )
        .execute(&self.pool)
        .await;

        let _ = sqlx::query(
            r#"CREATE INDEX IF NOT EXISTS qbitos_memories_embedding_hnsw_l2_idx
               ON qbitos_memories USING hnsw (embedding vector_l2_ops)
               WITH (m = 16, ef_construction = 64)"#,
        )
        .execute(&self.pool)
        .await;

        let _ = sqlx::query(
            r#"CREATE INDEX IF NOT EXISTS qbitos_memories_embedding_hnsw_ip_idx
               ON qbitos_memories USING hnsw (embedding vector_ip_ops)
               WITH (m = 16, ef_construction = 64)"#,
        )
        .execute(&self.pool)
        .await;

        // ─── B-tree indexes ─────────────────────────────────────────────
        let btree_indexes = [
            "CREATE INDEX IF NOT EXISTS qbitos_memories_layer_idx ON qbitos_memories(layer)",
            "CREATE INDEX IF NOT EXISTS qbitos_memories_created_at_idx ON qbitos_memories(created_at DESC)",
            "CREATE INDEX IF NOT EXISTS qbitos_memories_accessed_at_idx ON qbitos_memories(accessed_at DESC)",
            "CREATE INDEX IF NOT EXISTS qbitos_memories_importance_idx ON qbitos_memories(importance DESC)",
            "CREATE INDEX IF NOT EXISTS qbitos_memories_agent_id_idx ON qbitos_memories(agent_id) WHERE agent_id IS NOT NULL",
            "CREATE INDEX IF NOT EXISTS qbitos_memories_session_id_idx ON qbitos_memories(session_id) WHERE session_id IS NOT NULL",
            "CREATE INDEX IF NOT EXISTS qbitos_memories_layer_created_idx ON qbitos_memories(layer, created_at DESC)",
            "CREATE INDEX IF NOT EXISTS qbitos_memories_layer_importance_idx ON qbitos_memories(layer, importance DESC) WHERE embedding IS NOT NULL",
            "CREATE INDEX IF NOT EXISTS qbitos_memories_agent_layer_idx ON qbitos_memories(agent_id, layer) WHERE agent_id IS NOT NULL",
            "CREATE INDEX IF NOT EXISTS qbitos_memories_active_idx ON qbitos_memories(id) WHERE expires_at IS NULL OR expires_at > now()",
        ];
        for stmt in &btree_indexes {
            let _ = sqlx::query(stmt).execute(&self.pool).await;
        }

        // ─── GIN indexes for JSONB ──────────────────────────────────────
        let gin_indexes = [
            "CREATE INDEX IF NOT EXISTS qbitos_memories_metadata_gin_idx ON qbitos_memories USING gin (metadata jsonb_path_ops)",
            "CREATE INDEX IF NOT EXISTS qbitos_memories_content_gin_idx ON qbitos_memories USING gin (content)",
            "CREATE INDEX IF NOT EXISTS qbitos_memories_tags_gin_idx ON qbitos_memories USING gin (tags)",
            "CREATE INDEX IF NOT EXISTS qbitos_memories_search_vector_gin_idx ON qbitos_memories USING gin (search_vector)",
        ];
        for stmt in &gin_indexes {
            let _ = sqlx::query(stmt).execute(&self.pool).await;
        }

        // ─── Trigram indexes for fuzzy search ───────────────────────────
        let _ = sqlx::query(
            "CREATE INDEX IF NOT EXISTS qbitos_memories_text_content_trgm_idx ON qbitos_memories USING gin (text_content gin_trgm_ops)",
        )
        .execute(&self.pool)
        .await;

        let _ = sqlx::query(
            "CREATE INDEX IF NOT EXISTS qbitos_memories_layer_trgm_idx ON qbitos_memories USING gin (layer gin_trgm_ops)",
        )
        .execute(&self.pool)
        .await;

        // ─── ltree GiST index ───────────────────────────────────────────
        let _ = sqlx::query(
            "CREATE INDEX IF NOT EXISTS qbitos_memories_path_gist_idx ON qbitos_memories USING gist (path)",
        )
        .execute(&self.pool)
        .await;

        // ─── Bloom filter index ─────────────────────────────────────────
        let _ = sqlx::query(
            "CREATE INDEX IF NOT EXISTS qbitos_memories_bloom_idx ON qbitos_memories USING bloom (layer, agent_id, session_id) WITH (length = 80, col1 = 4, col2 = 4, col3 = 4)",
        )
        .execute(&self.pool)
        .await;

        // ─── Knowledge graph table ──────────────────────────────────────
        let _ = sqlx::query(
            r#"CREATE TABLE IF NOT EXISTS qbitos_knowledge_triples (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                subject TEXT NOT NULL,
                predicate TEXT NOT NULL,
                object TEXT NOT NULL,
                confidence REAL NOT NULL DEFAULT 1.0,
                source TEXT,
                metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
                path ltree NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )"#,
        )
        .execute(&self.pool)
        .await;

        for stmt in &[
            "CREATE INDEX IF NOT EXISTS kg_subject_idx ON qbitos_knowledge_triples(subject)",
            "CREATE INDEX IF NOT EXISTS kg_predicate_idx ON qbitos_knowledge_triples(predicate)",
            "CREATE INDEX IF NOT EXISTS kg_object_idx ON qbitos_knowledge_triples(object)",
            "CREATE INDEX IF NOT EXISTS kg_subject_predicate_idx ON qbitos_knowledge_triples(subject, predicate)",
            "CREATE INDEX IF NOT EXISTS kg_object_predicate_idx ON qbitos_knowledge_triples(object, predicate)",
            "CREATE INDEX IF NOT EXISTS kg_metadata_gin_idx ON qbitos_knowledge_triples USING gin (metadata jsonb_path_ops)",
            "CREATE INDEX IF NOT EXISTS kg_path_gist_idx ON qbitos_knowledge_triples USING gist (path)",
            "CREATE INDEX IF NOT EXISTS kg_subject_trgm_idx ON qbitos_knowledge_triples USING gin (subject gin_trgm_ops)",
            "CREATE INDEX IF NOT EXISTS kg_object_trgm_idx ON qbitos_knowledge_triples USING gin (object gin_trgm_ops)",
            "CREATE INDEX IF NOT EXISTS kg_confidence_idx ON qbitos_knowledge_triples(confidence DESC)",
        ] {
            let _ = sqlx::query(stmt).execute(&self.pool).await;
        }

        // ─── Skills table ───────────────────────────────────────────────
        let _ = sqlx::query(
            r#"CREATE TABLE IF NOT EXISTS qbitos_skills (
                id BIGSERIAL PRIMARY KEY,
                name TEXT NOT NULL UNIQUE,
                description TEXT NOT NULL,
                steps JSONB NOT NULL DEFAULT '[]'::jsonb,
                success_criteria JSONB NOT NULL DEFAULT '[]'::jsonb,
                execution_count BIGINT NOT NULL DEFAULT 0,
                success_count BIGINT NOT NULL DEFAULT 0,
                total_latency_ms BIGINT NOT NULL DEFAULT 0,
                confidence REAL NOT NULL DEFAULT 0.5,
                tags TEXT[] NOT NULL DEFAULT '{}',
                input_schema JSONB,
                output_schema JSONB,
                metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
                skill_path ltree,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )"#,
        )
        .execute(&self.pool)
        .await;

        for stmt in &[
            "CREATE INDEX IF NOT EXISTS skills_name_trgm_idx ON qbitos_skills USING gin (name gin_trgm_ops)",
            "CREATE INDEX IF NOT EXISTS skills_desc_trgm_idx ON qbitos_skills USING gin (description gin_trgm_ops)",
            "CREATE INDEX IF NOT EXISTS skills_tags_gin_idx ON qbitos_skills USING gin (tags)",
            "CREATE INDEX IF NOT EXISTS skills_steps_gin_idx ON qbitos_skills USING gin (steps jsonb_path_ops)",
            "CREATE INDEX IF NOT EXISTS skills_metadata_gin_idx ON qbitos_skills USING gin (metadata jsonb_path_ops)",
            "CREATE INDEX IF NOT EXISTS skills_confidence_idx ON qbitos_skills(confidence DESC)",
            "CREATE INDEX IF NOT EXISTS skills_path_gist_idx ON qbitos_skills USING gist (skill_path)",
            "CREATE INDEX IF NOT EXISTS skills_execution_count_idx ON qbitos_skills(execution_count DESC)",
        ] {
            let _ = sqlx::query(stmt).execute(&self.pool).await;
        }

        // ─── Agents table ───────────────────────────────────────────────
        let _ = sqlx::query(
            r#"CREATE TABLE IF NOT EXISTS qbitos_agents (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                name TEXT NOT NULL,
                kind TEXT NOT NULL,
                state JSONB NOT NULL DEFAULT '{}'::jsonb,
                config JSONB NOT NULL DEFAULT '{}'::jsonb,
                permissions JSONB NOT NULL DEFAULT '[]'::jsonb,
                status TEXT NOT NULL DEFAULT 'idle',
                parent_id UUID REFERENCES qbitos_agents(id) ON DELETE SET NULL,
                hierarchy ltree,
                metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                last_active TIMESTAMPTZ
            )"#,
        )
        .execute(&self.pool)
        .await;

        for stmt in &[
            "CREATE INDEX IF NOT EXISTS agents_name_trgm_idx ON qbitos_agents USING gin (name gin_trgm_ops)",
            "CREATE INDEX IF NOT EXISTS agents_kind_idx ON qbitos_agents(kind)",
            "CREATE INDEX IF NOT EXISTS agents_status_idx ON qbitos_agents(status)",
            "CREATE INDEX IF NOT EXISTS agents_state_gin_idx ON qbitos_agents USING gin (state jsonb_path_ops)",
            "CREATE INDEX IF NOT EXISTS agents_config_gin_idx ON qbitos_agents USING gin (config)",
            "CREATE INDEX IF NOT EXISTS agents_permissions_gin_idx ON qbitos_agents USING gin (permissions jsonb_path_ops)",
            "CREATE INDEX IF NOT EXISTS agents_metadata_gin_idx ON qbitos_agents USING gin (metadata jsonb_path_ops)",
            "CREATE INDEX IF NOT EXISTS agents_parent_idx ON qbitos_agents(parent_id)",
            "CREATE INDEX IF NOT EXISTS agents_hierarchy_gist_idx ON qbitos_agents USING gist (hierarchy)",
            "CREATE INDEX IF NOT EXISTS agents_last_active_idx ON qbitos_agents(last_active DESC) WHERE last_active IS NOT NULL",
        ] {
            let _ = sqlx::query(stmt).execute(&self.pool).await;
        }

        // ─── Events table ───────────────────────────────────────────────
        let _ = sqlx::query(
            r#"CREATE TABLE IF NOT EXISTS qbitos_events (
                id BIGSERIAL PRIMARY KEY,
                event_id BIGINT NOT NULL UNIQUE,
                kind TEXT NOT NULL,
                subsystem TEXT,
                payload JSONB NOT NULL DEFAULT '{}'::jsonb,
                agent_id UUID,
                session_id UUID,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )"#,
        )
        .execute(&self.pool)
        .await;

        for stmt in &[
            "CREATE INDEX IF NOT EXISTS events_event_id_idx ON qbitos_events(event_id)",
            "CREATE INDEX IF NOT EXISTS events_kind_idx ON qbitos_events(kind)",
            "CREATE INDEX IF NOT EXISTS events_created_at_idx ON qbitos_events(created_at DESC)",
            "CREATE INDEX IF NOT EXISTS events_payload_gin_idx ON qbitos_events USING gin (payload jsonb_path_ops)",
            "CREATE INDEX IF NOT EXISTS events_agent_id_idx ON qbitos_events(agent_id) WHERE agent_id IS NOT NULL",
            "CREATE INDEX IF NOT EXISTS events_kind_created_idx ON qbitos_events(kind, created_at DESC)",
            "CREATE INDEX IF NOT EXISTS events_subsystem_idx ON qbitos_events(subsystem) WHERE subsystem IS NOT NULL",
        ] {
            let _ = sqlx::query(stmt).execute(&self.pool).await;
        }

        // ─── Tool calls table ───────────────────────────────────────────
        let _ = sqlx::query(
            r#"CREATE TABLE IF NOT EXISTS qbitos_tool_calls (
                id BIGSERIAL PRIMARY KEY,
                tool_name TEXT NOT NULL,
                agent_id UUID,
                parameters JSONB NOT NULL DEFAULT '{}'::jsonb,
                output JSONB,
                success BOOLEAN NOT NULL DEFAULT false,
                duration_us BIGINT NOT NULL DEFAULT 0,
                error TEXT,
                metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )"#,
        )
        .execute(&self.pool)
        .await;

        for stmt in &[
            "CREATE INDEX IF NOT EXISTS tool_calls_tool_name_idx ON qbitos_tool_calls(tool_name)",
            "CREATE INDEX IF NOT EXISTS tool_calls_created_at_idx ON qbitos_tool_calls(created_at DESC)",
            "CREATE INDEX IF NOT EXISTS tool_calls_agent_id_idx ON qbitos_tool_calls(agent_id) WHERE agent_id IS NOT NULL",
            "CREATE INDEX IF NOT EXISTS tool_calls_parameters_gin_idx ON qbitos_tool_calls USING gin (parameters)",
            "CREATE INDEX IF NOT EXISTS tool_calls_output_gin_idx ON qbitos_tool_calls USING gin (output)",
            "CREATE INDEX IF NOT EXISTS tool_calls_metadata_gin_idx ON qbitos_tool_calls USING gin (metadata jsonb_path_ops)",
            "CREATE INDEX IF NOT EXISTS tool_calls_success_idx ON qbitos_tool_calls(success) WHERE success = false",
            "CREATE INDEX IF NOT EXISTS tool_calls_tool_name_trgm_idx ON qbitos_tool_calls USING gin (tool_name gin_trgm_ops)",
        ] {
            let _ = sqlx::query(stmt).execute(&self.pool).await;
        }

        // ─── Training steps table ───────────────────────────────────────
        let _ = sqlx::query(
            r#"CREATE TABLE IF NOT EXISTS qbitos_training_steps (
                id BIGSERIAL PRIMARY KEY,
                step_number BIGINT NOT NULL,
                stage TEXT,
                reward REAL NOT NULL DEFAULT 0.0,
                loss REAL NOT NULL DEFAULT 1.0,
                strategy TEXT,
                success BOOLEAN NOT NULL DEFAULT false,
                duration_ms BIGINT NOT NULL DEFAULT 0,
                state_snapshot JSONB NOT NULL DEFAULT '{}'::jsonb,
                metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )"#,
        )
        .execute(&self.pool)
        .await;

        for stmt in &[
            "CREATE INDEX IF NOT EXISTS training_steps_step_idx ON qbitos_training_steps(step_number)",
            "CREATE INDEX IF NOT EXISTS training_steps_created_idx ON qbitos_training_steps(created_at DESC)",
            "CREATE INDEX IF NOT EXISTS training_steps_stage_idx ON qbitos_training_steps(stage) WHERE stage IS NOT NULL",
            "CREATE INDEX IF NOT EXISTS training_steps_strategy_idx ON qbitos_training_steps(strategy) WHERE strategy IS NOT NULL",
            "CREATE INDEX IF NOT EXISTS training_steps_reward_idx ON qbitos_training_steps(reward DESC)",
            "CREATE INDEX IF NOT EXISTS training_steps_metadata_gin_idx ON qbitos_training_steps USING gin (metadata jsonb_path_ops)",
            "CREATE INDEX IF NOT EXISTS training_steps_state_gin_idx ON qbitos_training_steps USING gin (state_snapshot jsonb_path_ops)",
        ] {
            let _ = sqlx::query(stmt).execute(&self.pool).await;
        }

        // ─── Planner state table ────────────────────────────────────────
        let _ = sqlx::query(
            r#"CREATE TABLE IF NOT EXISTS qbitos_planner_state (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                goal TEXT NOT NULL,
                task_graph JSONB NOT NULL DEFAULT '{}'::jsonb,
                state JSONB NOT NULL DEFAULT '{}'::jsonb,
                status TEXT NOT NULL DEFAULT 'pending',
                agent_id UUID,
                parent_id UUID REFERENCES qbitos_planner_state(id) ON DELETE SET NULL,
                task_path ltree,
                priority INTEGER NOT NULL DEFAULT 0,
                metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                completed_at TIMESTAMPTZ
            )"#,
        )
        .execute(&self.pool)
        .await;

        for stmt in &[
            "CREATE INDEX IF NOT EXISTS planner_goal_trgm_idx ON qbitos_planner_state USING gin (goal gin_trgm_ops)",
            "CREATE INDEX IF NOT EXISTS planner_status_idx ON qbitos_planner_state(status)",
            "CREATE INDEX IF NOT EXISTS planner_task_graph_gin_idx ON qbitos_planner_state USING gin (task_graph jsonb_path_ops)",
            "CREATE INDEX IF NOT EXISTS planner_state_gin_idx ON qbitos_planner_state USING gin (state jsonb_path_ops)",
            "CREATE INDEX IF NOT EXISTS planner_metadata_gin_idx ON qbitos_planner_state USING gin (metadata jsonb_path_ops)",
            "CREATE INDEX IF NOT EXISTS planner_priority_idx ON qbitos_planner_state(priority DESC) WHERE status = 'pending'",
            "CREATE INDEX IF NOT EXISTS planner_task_path_gist_idx ON qbitos_planner_state USING gist (task_path)",
            "CREATE INDEX IF NOT EXISTS planner_agent_idx ON qbitos_planner_state(agent_id) WHERE agent_id IS NOT NULL",
            "CREATE INDEX IF NOT EXISTS planner_parent_idx ON qbitos_planner_state(parent_id) WHERE parent_id IS NOT NULL",
        ] {
            let _ = sqlx::query(stmt).execute(&self.pool).await;
        }

        // ─── Updated triggers ───────────────────────────────────────────
        let _ = sqlx::query(
            r#"CREATE OR REPLACE FUNCTION update_updated_at()
               RETURNS TRIGGER AS $$
               BEGIN
                   NEW.updated_at = now();
                   RETURN NEW;
               END;
               $$ LANGUAGE plpgsql"#,
        )
        .execute(&self.pool)
        .await;

        for table in &[
            "qbitos_memories",
            "qbitos_knowledge_triples",
            "qbitos_skills",
            "qbitos_agents",
            "qbitos_planner_state",
        ] {
            let _ = sqlx::query(&format!(
                "DROP TRIGGER IF EXISTS {table}_updated_at ON {table}; \
                 CREATE TRIGGER {table}_updated_at \
                 BEFORE UPDATE ON {table} \
                 FOR EACH ROW EXECUTE FUNCTION update_updated_at()"
            ))
            .execute(&self.pool)
            .await;
        }

        // ─── Server-side functions ──────────────────────────────────────
        let _ = sqlx::query(
            r#"CREATE OR REPLACE FUNCTION search_memories(
                   p_embedding vector,
                   p_k INTEGER DEFAULT 10,
                   p_min_score REAL DEFAULT 0.0,
                   p_layer TEXT DEFAULT NULL,
                   p_tags TEXT[] DEFAULT NULL
               )
               RETURNS TABLE (
                   id uuid, layer text, content jsonb, embedding text,
                   importance real, confidence real, created_at timestamptz,
                   accessed_at timestamptz, usage_count integer, metadata jsonb, score real
               )
               LANGUAGE sql STABLE PARALLEL SAFE AS $$
                   SELECT m.id, m.layer, m.content, m.embedding::text,
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
               $$"#,
        )
        .execute(&self.pool)
        .await;

        let _ = sqlx::query(
            r#"CREATE OR REPLACE FUNCTION search_memories_fts(
                   p_query TEXT, p_k INTEGER DEFAULT 10, p_layer TEXT DEFAULT NULL
               )
               RETURNS TABLE (id uuid, layer text, content jsonb, importance real, rank real)
               LANGUAGE sql STABLE PARALLEL SAFE AS $$
                   SELECT m.id, m.layer, m.content, m.importance,
                          ts_rank(m.search_vector, plainto_tsquery('english', p_query))::real
                   FROM qbitos_memories m
                   WHERE m.search_vector @@ plainto_tsquery('english', p_query)
                     AND (p_layer IS NULL OR m.layer = p_layer)
                   ORDER BY rank DESC LIMIT p_k
               $$"#,
        )
        .execute(&self.pool)
        .await;

        let _ = sqlx::query(
            r#"CREATE OR REPLACE FUNCTION search_memories_fuzzy(
                   p_query TEXT, p_k INTEGER DEFAULT 10, p_threshold REAL DEFAULT 0.3
               )
               RETURNS TABLE (id uuid, layer text, text_content text, similarity real)
               LANGUAGE sql STABLE PARALLEL SAFE AS $$
                   SELECT m.id, m.layer, m.text_content,
                          similarity(m.text_content, p_query)::real
                   FROM qbitos_memories m
                   WHERE m.text_content IS NOT NULL
                     AND similarity(m.text_content, p_query) >= p_threshold
                   ORDER BY similarity DESC LIMIT p_k
               $$"#,
        )
        .execute(&self.pool)
        .await;

        Ok(())
    }

    /// Prewarm critical indexes into the PostgreSQL buffer cache.
    async fn prewarm(&self) {
        for idx in &[
            "qbitos_memories_embedding_hnsw_cosine_idx",
            "qbitos_memories_content_gin_idx",
            "qbitos_memories_layer_idx",
            "qbitos_memories_search_vector_gin_idx",
        ] {
            let _ = sqlx::query(&format!("SELECT pg_prewarm('{idx}')"))
                .execute(&self.pool)
                .await;
        }
        tracing::info!(target: "qbitos.postgres", "prewarmed critical indexes");
    }

    /// Convert an embedding to a Postgres vector literal.
    fn embedding_to_pg_vector(emb: &Embedding) -> String {
        let inner: Vec<String> = emb.data.iter().map(|f| f.to_string()).collect();
        format!("[{}]", inner.join(","))
    }

    /// Parse a Postgres vector literal back to an Embedding.
    fn pg_vector_to_embedding(s: &str) -> Embedding {
        let trimmed = s.trim_start_matches('[').trim_end_matches(']');
        let data: Vec<f32> = trimmed
            .split(',')
            .filter_map(|s| s.trim().parse::<f32>().ok())
            .collect();
        Embedding { data }
    }

    /// Get a reference to the connection pool (for advanced queries).
    #[must_use]
    pub fn pool(&self) -> &PgPool {
        &self.pool
    }

    /// Full-text search across memory content.
    ///
    /// # Errors
    /// Returns [`BackendError`] on database errors.
    pub async fn full_text_search(
        &self,
        query: &str,
        k: usize,
        layer: Option<&str>,
    ) -> Result<Vec<(MemoryId, String, serde_json::Value, f32, f64)>, BackendError> {
        let rows: Vec<(uuid::Uuid, String, serde_json::Value, f32, f32)> = sqlx::query_as(
            "SELECT id, layer, content, importance, \
             ts_rank(search_vector, plainto_tsquery('english', $1))::real AS rank \
             FROM qbitos_memories \
             WHERE search_vector @@ plainto_tsquery('english', $1) \
             AND ($2::text IS NULL OR layer = $2) \
             ORDER BY rank DESC LIMIT $3",
        )
        .bind(query)
        .bind(layer)
        .bind(k as i64)
        .fetch_all(&self.pool)
        .await
        .map_err(|e| BackendError::Connection(e.to_string()))?;

        Ok(rows
            .into_iter()
            .map(|(id, layer, content, importance, rank)| {
                (
                    MemoryId(id),
                    layer,
                    content,
                    importance,
                    f64::from(rank),
                )
            })
            .collect())
    }

    /// Fuzzy (trigram) search across memory text content.
    ///
    /// # Errors
    /// Returns [`BackendError`] on database errors.
    pub async fn fuzzy_search(
        &self,
        query: &str,
        k: usize,
        threshold: f64,
    ) -> Result<Vec<(MemoryId, String, String, f64)>, BackendError> {
        let rows: Vec<(uuid::Uuid, String, String, f32)> = sqlx::query_as(
            "SELECT id, layer, text_content, \
             similarity(text_content, $1)::real AS sim \
             FROM qbitos_memories \
             WHERE text_content IS NOT NULL \
             AND similarity(text_content, $1) >= $2 \
             ORDER BY sim DESC LIMIT $3",
        )
        .bind(query)
        .bind(threshold as f32)
        .bind(k as i64)
        .fetch_all(&self.pool)
        .await
        .map_err(|e| BackendError::Connection(e.to_string()))?;

        Ok(rows
            .into_iter()
            .map(|(id, layer, text, sim)| {
                (MemoryId(id), layer, text, f64::from(sim))
            })
            .collect())
    }

    /// Get index usage statistics.
    ///
    /// # Errors
    /// Returns [`BackendError`] on database errors.
    pub async fn index_usage_stats(&self) -> Result<Vec<IndexUsageStat>, BackendError> {
        let rows: Vec<(String, String, String, i64, i64, i64, i64)> = sqlx::query_as(
            "SELECT schemaname, relname, indexrelname, idx_scan, idx_tup_read, idx_tup_fetch, \
             pg_relation_size(indexrelid)::bigint \
             FROM pg_stat_user_indexes \
             WHERE schemaname = 'public' AND relname LIKE 'qbitos_%' \
             ORDER BY idx_scan DESC",
        )
        .fetch_all(&self.pool)
        .await
        .map_err(|e| BackendError::Connection(e.to_string()))?;

        Ok(rows
            .into_iter()
            .map(|(schema, table, index, scans, reads, fetches, size)| IndexUsageStat {
                schema,
                table,
                index,
                scans,
                tuples_read: reads,
                tuples_fetched: fetches,
                size_bytes: size,
            })
            .collect())
    }

    /// Get slow query statistics from pg_stat_statements.
    ///
    /// # Errors
    /// Returns [`BackendError`] on database errors.
    pub async fn slow_queries(&self, limit: usize) -> Result<Vec<SlowQueryStat>, BackendError> {
        let rows: Vec<(i64, String, i64, f64, f64, f64, i64, i64, i64, i64)> = sqlx::query_as(
            "SELECT queryid, LEFT(query, 300), calls, total_exec_time, mean_exec_time, \
             max_exec_time, rows, shared_blks_hit, shared_blks_read, shared_blks_dirtied \
             FROM pg_stat_statements \
             WHERE query NOT LIKE '%pg_stat_statements%' AND query NOT LIKE '%pg_%' \
             ORDER BY mean_exec_time DESC LIMIT $1",
        )
        .bind(limit as i64)
        .fetch_all(&self.pool)
        .await
        .map_err(|e| BackendError::Connection(e.to_string()))?;

        Ok(rows
            .into_iter()
            .map(
                |(id, query, calls, total, mean, max, rows, hit, read, dirty)| SlowQueryStat {
                    query_id: id,
                    query_preview: query,
                    calls,
                    total_exec_time_ms: total,
                    mean_exec_time_ms: mean,
                    max_exec_time_ms: max,
                    rows,
                    shared_blks_hit: hit,
                    shared_blks_read: read,
                    shared_blks_dirtied: dirty,
                },
            )
            .collect())
    }
}

/// Index usage statistics from pg_stat_user_indexes.
#[derive(Debug, Clone, serde::Serialize)]
pub struct IndexUsageStat {
    /// Schema name.
    pub schema: String,
    /// Table name.
    pub table: String,
    /// Index name.
    pub index: String,
    /// Number of index scans.
    pub scans: i64,
    /// Number of tuples read by index scans.
    pub tuples_read: i64,
    /// Number of tuples fetched by index scans.
    pub tuples_fetched: i64,
    /// Index size in bytes.
    pub size_bytes: i64,
}

/// Slow query statistics from pg_stat_statements.
#[derive(Debug, Clone, serde::Serialize)]
pub struct SlowQueryStat {
    /// Query ID.
    pub query_id: i64,
    /// First 300 characters of the query.
    pub query_preview: String,
    /// Number of times the query was called.
    pub calls: i64,
    /// Total execution time in milliseconds.
    pub total_exec_time_ms: f64,
    /// Mean execution time per call in milliseconds.
    pub mean_exec_time_ms: f64,
    /// Maximum execution time in milliseconds.
    pub max_exec_time_ms: f64,
    /// Number of rows returned.
    pub rows: i64,
    /// Shared buffer hits.
    pub shared_blks_hit: i64,
    /// Shared buffer reads (misses).
    pub shared_blks_read: i64,
    /// Shared buffer dirtied.
    pub shared_blks_dirtied: i64,
}

#[async_trait]
impl MemoryBackend for PostgresBackend {
    fn name(&self) -> &str {
        "postgres"
    }
    fn supports_vector_search(&self) -> bool {
        true
    }
    fn is_durable(&self) -> bool {
        true
    }

    async fn store(&self, object: &MemoryObject) -> Result<(), BackendError> {
        let id = object.id.0;
        let layer = format!("{:?}", object.layer).to_lowercase();
        let content = serde_json::to_value(&object.content)
            .map_err(|e| BackendError::Serialization(e.to_string()))?;
        let embedding = if object.embedding.data.is_empty() {
            None
        } else {
            Some(Self::embedding_to_pg_vector(&object.embedding))
        };
        let importance = object.importance.0;
        let confidence = object.confidence;
        let created_at = object.created_at;
        let accessed_at = object.last_accessed_at;
        let usage_count = object.usage_count as i32;
        let metadata = serde_json::to_value(&object.metadata)
            .map_err(|e| BackendError::Serialization(e.to_string()))?;

        // Extract text content for full-text search.
        let text_content = content
            .get("text")
            .and_then(|v| v.as_str())
            .or_else(|| content.get("content").and_then(|v| v.as_str()))
            .or_else(|| content.get("query").and_then(|v| v.as_str()));

        // Generate search vector for full-text search.
        let search_vector: Option<String> = text_content.map(|t| {
            // We store the tsvector as a computed expression.
            // The trigger/function handles this, but we also set it here for immediate availability.
            t.to_string()
        });

        // Use parameterized query (SQL injection safe).
        sqlx::query(
            r#"INSERT INTO qbitos_memories
                   (id, layer, content, embedding, importance, confidence,
                    created_at, accessed_at, usage_count, metadata, text_content, search_vector)
               VALUES ($1, $2, $3, $4::vector, $5, $6, $7, $8, $9, $10, $11,
                       to_tsvector('english', COALESCE($11, '')))
               ON CONFLICT (id) DO UPDATE SET
                   layer = EXCLUDED.layer,
                   content = EXCLUDED.content,
                   embedding = EXCLUDED.embedding,
                   importance = EXCLUDED.importance,
                   confidence = EXCLUDED.confidence,
                   accessed_at = EXCLUDED.accessed_at,
                   usage_count = EXCLUDED.usage_count,
                   metadata = EXCLUDED.metadata,
                   text_content = EXCLUDED.text_content,
                   search_vector = to_tsvector('english', COALESCE(EXCLUDED.text_content, ''))"#,
        )
        .bind(id)
        .bind(&layer)
        .bind(&content)
        .bind(embedding)
        .bind(importance)
        .bind(confidence)
        .bind(created_at)
        .bind(accessed_at)
        .bind(usage_count)
        .bind(&metadata)
        .bind(search_vector)
        .execute(&self.pool)
        .await
        .map_err(|e| BackendError::Connection(e.to_string()))?;
        Ok(())
    }

    async fn get(&self, id: MemoryId) -> Result<MemoryObject, BackendError> {
        let row: (
            uuid::Uuid,
            String,
            serde_json::Value,
            Option<String>,
            f32,
            f32,
            chrono::DateTime<chrono::Utc>,
            chrono::DateTime<chrono::Utc>,
            i32,
            serde_json::Value,
        ) = sqlx::query_as(
            r#"SELECT id, layer, content, embedding::text, importance, confidence,
                      created_at, accessed_at, usage_count, metadata
               FROM qbitos_memories WHERE id = $1"#,
        )
        .bind(id.0)
        .fetch_one(&self.pool)
        .await
        .map_err(|e| match e {
            sqlx::Error::RowNotFound => BackendError::NotFound(id),
            e => BackendError::Connection(e.to_string()),
        })?;

        let layer = parse_layer(&row.1);
        let embedding = row
            .3
            .as_deref()
            .map(Self::pg_vector_to_embedding)
            .unwrap_or_else(Embedding::empty);

        Ok(MemoryObject {
            id: MemoryId(row.0),
            layer,
            content: row.2,
            embedding,
            importance: Importance(row.4),
            confidence: row.5,
            created_at: row.6,
            last_accessed_at: row.7,
            usage_count: row.8 as u32,
            relationships: Vec::new(),
            causal_links: Vec::new(),
            metadata: serde_json::from_value(row.9).unwrap_or_default(),
        })
    }

    async fn delete(&self, id: MemoryId) -> Result<(), BackendError> {
        sqlx::query("DELETE FROM qbitos_memories WHERE id = $1")
            .bind(id.0)
            .execute(&self.pool)
            .await
            .map_err(|e| BackendError::Connection(e.to_string()))?;
        Ok(())
    }

    async fn count(&self) -> Result<usize, BackendError> {
        let count: (i64,) = sqlx::query_as("SELECT COUNT(*) FROM qbitos_memories")
            .fetch_one(&self.pool)
            .await
            .map_err(|e| BackendError::Connection(e.to_string()))?;
        Ok(count.0 as usize)
    }

    async fn search(&self, query: &SimilarityQuery) -> Result<Vec<SimilarityResult>, BackendError> {
        if query.embedding.data.is_empty() {
            return Err(BackendError::Serialization("empty query embedding".into()));
        }
        let query_vec = Self::embedding_to_pg_vector(&query.embedding);
        let limit = query.k as i64;
        let min_score = query.min_score;

        // Use parameterized query with the server-side function for optimal execution.
        let layer_param: Option<String> = query.layer_filter.as_ref().map(|l| {
            format!("{:?}", l).to_lowercase()
        });

        let rows: Vec<(
            uuid::Uuid,
            String,
            serde_json::Value,
            Option<String>,
            f32,
            f32,
            chrono::DateTime<chrono::Utc>,
            chrono::DateTime<chrono::Utc>,
            i32,
            serde_json::Value,
            f32,
        )> = sqlx::query_as(
            r#"SELECT id, layer, content, embedding::text, importance, confidence,
                      created_at, accessed_at, usage_count, metadata,
                      1 - (embedding <=> $1::vector) AS score
               FROM qbitos_memories
               WHERE embedding IS NOT NULL
                 AND 1 - (embedding <=> $1::vector) >= $2
                 AND ($3::text IS NULL OR layer = $3)
               ORDER BY embedding <=> $1::vector
               LIMIT $4"#,
        )
        .bind(&query_vec)
        .bind(min_score)
        .bind(&layer_param)
        .bind(limit)
        .fetch_all(&self.pool)
        .await
        .map_err(|e| BackendError::Connection(e.to_string()))?;

        Ok(rows
            .into_iter()
            .map(|row| {
                let layer = parse_layer(&row.1);
                let embedding = row
                    .3
                    .as_deref()
                    .map(Self::pg_vector_to_embedding)
                    .unwrap_or_else(Embedding::empty);
                let object = MemoryObject {
                    id: MemoryId(row.0),
                    layer,
                    content: row.2,
                    embedding,
                    importance: Importance(row.4),
                    confidence: row.5,
                    created_at: row.6,
                    last_accessed_at: row.7,
                    usage_count: row.8 as u32,
                    relationships: Vec::new(),
                    causal_links: Vec::new(),
                    metadata: serde_json::from_value(row.9).unwrap_or_default(),
                };
                SimilarityResult {
                    object,
                    score: row.10,
                }
            })
            .collect())
    }

    async fn flush(&self) -> Result<(), BackendError> {
        // Postgres is write-through — no flush needed
        Ok(())
    }
}

/// Parse a layer string into a MemoryType.
fn parse_layer(s: &str) -> MemoryType {
    match s {
        "sensory" => MemoryType::Sensory,
        "working" => MemoryType::Working,
        "episodic" => MemoryType::Episodic,
        "semantic" => MemoryType::Semantic,
        "procedural" => MemoryType::Procedural,
        "meta" => MemoryType::Meta,
        _ => MemoryType::Episodic,
    }
}

// Helper for empty embeddings.
impl Embedding {
    /// Returns an empty embedding.
    pub(crate) fn empty() -> Self {
        Self { data: Vec::new() }
    }
}
