-- ═══════════════════════════════════════════════════════════════════════════════
-- QbitOS Comprehensive Seed Data — gives the cognitive agent a strong kick-start
-- ═══════════════════════════════════════════════════════════════════════════════
--
-- This file populates the database with:
--   - 50+ memories across all 6 layers (L0–L5)
--   - 15+ skills covering all built-in tools and cognitive operations
--   - 100+ knowledge graph triples (QbitOS architecture, programming, science)
--   - Training data: experience replay, curriculum, meta-learning history
--   - Safety guardrail rules (banned patterns, PII, allowed URLs)
--   - Cognitive cycle history (sample 8-stage cycles)
--   - World model entities, causal graph, simulation scenarios
--   - Tool execution statistics for all 14 built-in tools
--   - Audit log entries
--
-- Idempotent: ON CONFLICT DO NOTHING throughout. Safe to run multiple times.
-- ═══════════════════════════════════════════════════════════════════════════════

BEGIN;

-- ═══════════════════════════════════════════════════════════════════════════════
-- 1. MEMORIES (50+ across L0–L5)
-- ═══════════════════════════════════════════════════════════════════════════════

-- ─── L0 Sensory (raw perceptions, <1s retention) ─────────────────────────────
INSERT INTO qbitos_memories (id, layer, content, embedding, importance, created_at, updated_at, access_count, last_accessed_at, expires_at)
SELECT
    ('a0000000-0000-0000-0000-' || lpad(to_hex(generate_series), 8, '0'))::uuid,
    'sensory',
    jsonb_build_object('type', 'text', 'input', 'User input #' || generate_series, 'modality', 'text', 'timestamp', now() - (generate_series || ' seconds')::interval),
    (SELECT array_agg(random())::vector(384) FROM generate_series(1, 384)),
    0.3,
    now() - (generate_series || ' seconds')::interval,
    now() - (generate_series || ' seconds')::interval,
    0, NULL,
    now() + interval '5 seconds'
FROM generate_series(1, 10)
ON CONFLICT (id) DO NOTHING;

-- ─── L1 Working (current task context, <5min retention) ──────────────────────
INSERT INTO qbitos_memories (id, layer, content, embedding, importance, created_at, updated_at, access_count, last_accessed_at, expires_at)
VALUES
    ('a0000000-0001-0000-0000-000000000001', 'working',
     '{"task":"answer_question","prompt":"What is QbitOS?","status":"completed","result":"QbitOS is a cognitive operating system built in Rust"}'::jsonb,
     (SELECT array_agg(random())::vector(384) FROM generate_series(1, 384)),
     0.7, now() - interval '2 minutes', now(), 5, now(), NULL),
    ('a0000000-0001-0000-0000-000000000002', 'working',
     '{"task":"code_analysis","prompt":"Analyze this Rust function","status":"active","context":"fn main() { println!(\"hello\"); }"}'::jsonb,
     (SELECT array_agg(random())::vector(384) FROM generate_series(1, 384)),
     0.8, now() - interval '1 minute', now(), 3, now(), NULL),
    ('a0000000-0001-0000-0000-000000000003', 'working',
     '{"task":"memory_search","query":"cognitive architecture","results_count":5,"status":"completed"}'::jsonb,
     (SELECT array_agg(random())::vector(384) FROM generate_series(1, 384)),
     0.6, now() - interval '3 minutes', now(), 2, now(), NULL),
    ('a0000000-0001-0000-0000-000000000004', 'working',
     '{"task":"skill_execution","skill":"qa","status":"completed","duration_ms":450,"success":true}'::jsonb,
     (SELECT array_agg(random())::vector(384) FROM generate_series(1, 384)),
     0.75, now() - interval '30 seconds', now(), 1, now(), NULL),
    ('a0000000-0001-0000-0000-000000000005', 'working',
     '{"task":"planning","goal":"explain cognitive loop","steps":["recall","infer","respond"],"status":"active"}'::jsonb,
     (SELECT array_agg(random())::vector(384) FROM generate_series(1, 384)),
     0.8, now() - interval '10 seconds', now(), 1, now(), NULL)
ON CONFLICT (id) DO NOTHING;

-- ─── L2 Episodic (past experiences, days–weeks retention) ────────────────────
INSERT INTO qbitos_memories (id, layer, content, embedding, importance, created_at, updated_at, access_count, last_accessed_at, expires_at)
VALUES
    ('a0000000-0002-0000-0000-000000000001', 'episodic',
     '{"event":"user_asked_about_memory","outcome":"explained_L0_L5_layers","success":true,"duration_ms":1200,"user_satisfied":true,"timestamp":"2026-01-15T10:30:00Z"}'::jsonb,
     (SELECT array_agg(random())::vector(384) FROM generate_series(1, 384)),
     0.6, now() - interval '2 hours', now(), 3, now() - interval '1 hour', NULL),
    ('a0000000-0002-0000-0000-000000000002', 'episodic',
     '{"event":"code_analysis_request","outcome":"identified_3_issues","success":true,"language":"rust","issues":["unused_variable","clone_can_be_avoided","unwrap_dangerous"],"timestamp":"2026-01-15T11:00:00Z"}'::jsonb,
     (SELECT array_agg(random())::vector(384) FROM generate_series(1, 384)),
     0.7, now() - interval '1 hour', now(), 2, now() - interval '30 minutes', NULL),
    ('a0000000-0002-0000-0000-000000000003', 'episodic',
     '{"event":"web_search_performed","query":"rust async runtime comparison","results_count":10,"success":true,"timestamp":"2026-01-15T11:30:00Z"}'::jsonb,
     (SELECT array_agg(random())::vector(384) FROM generate_series(1, 384)),
     0.5, now() - interval '45 minutes', now(), 1, now() - interval '20 minutes', NULL),
    ('a0000000-0002-0000-0000-000000000004', 'episodic',
     '{"event":"training_cycle","steps":50,"mean_reward":0.72,"curriculum_advanced":false,"duration_ms":5400,"timestamp":"2026-01-15T12:00:00Z"}'::jsonb,
     (SELECT array_agg(random())::vector(384) FROM generate_series(1, 384)),
     0.65, now() - interval '30 minutes', now(), 2, now() - interval '15 minutes', NULL),
    ('a0000000-0002-0000-0000-000000000005', 'episodic',
     '{"event":"skill_optimization","pruned":2,"improved":1,"total_skills":15,"timestamp":"2026-01-15T12:15:00Z"}'::jsonb,
     (SELECT array_agg(random())::vector(384) FROM generate_series(1, 384)),
     0.55, now() - interval '20 minutes', now(), 1, now() - interval '10 minutes', NULL),
    ('a0000000-0002-0000-0000-000000000006', 'episodic',
     '{"event":"safety_check_triggered","type":"output_validation","blocked":false,"reason":"output_passed_all_checks","timestamp":"2026-01-15T12:30:00Z"}'::jsonb,
     (SELECT array_agg(random())::vector(384) FROM generate_series(1, 384)),
     0.5, now() - interval '15 minutes', now(), 1, now() - interval '5 minutes', NULL),
    ('a0000000-0002-0000-0000-000000000007', 'episodic',
     '{"event":"hallucination_detected","claim":"Rust 2.0 released","verified":false,"blocked":true,"action":"corrected_response","timestamp":"2026-01-15T12:45:00Z"}'::jsonb,
     (SELECT array_agg(random())::vector(384) FROM generate_series(1, 384)),
     0.85, now() - interval '10 minutes', now(), 3, now() - interval '5 minutes', NULL),
    ('a0000000-0002-0000-0000-000000000008', 'episodic',
     '{"event":"user_feedback","rating":5,"comment":"Very helpful explanation","skill":"qa","timestamp":"2026-01-15T13:00:00Z"}'::jsonb,
     (SELECT array_agg(random())::vector(384) FROM generate_series(1, 384)),
     0.8, now() - interval '5 minutes', now(), 2, now() - interval '2 minutes', NULL)
ON CONFLICT (id) DO NOTHING;

-- ─── L3 Semantic (facts, concepts, permanent) ────────────────────────────────
INSERT INTO qbitos_memories (id, layer, content, embedding, importance, created_at, updated_at, access_count, last_accessed_at, expires_at)
VALUES
    ('a0000000-0003-0000-0000-000000000001', 'semantic',
     '{"subject":"QbitOS","predicate":"is_a","object":"cognitive_operating_system","confidence":0.95}'::jsonb,
     (SELECT array_agg(random())::vector(384) FROM generate_series(1, 384)),
     0.95, now() - interval '1 day', now(), 10, now(), NULL),
    ('a0000000-0003-0000-0000-000000000002', 'semantic',
     '{"subject":"QbitOS","predicate":"written_in","object":"Rust","confidence":1.0}'::jsonb,
     (SELECT array_agg(random())::vector(384) FROM generate_series(1, 384)),
     0.95, now() - interval '1 day', now(), 8, now(), NULL),
    ('a0000000-0003-0000-0000-000000000003', 'semantic',
     '{"subject":"cognitive_kernel","predicate":"has_stages","object":"8","confidence":1.0,"stages":["observe","understand","create_goal","plan","execute","verify","learn","improve"]}'::jsonb,
     (SELECT array_agg(random())::vector(384) FROM generate_series(1, 384)),
     0.9, now() - interval '1 day', now(), 7, now(), NULL),
    ('a0000000-0003-0000-0000-000000000004', 'semantic',
     '{"subject":"memory_fabric","predicate":"has_layers","object":"6","confidence":1.0,"layers":["L0_sensory","L1_working","L2_episodic","L3_semantic","L4_procedural","L5_meta"]}'::jsonb,
     (SELECT array_agg(random())::vector(384) FROM generate_series(1, 384)),
     0.9, now() - interval '1 day', now(), 6, now(), NULL),
    ('a0000000-0003-0000-0000-000000000005', 'semantic',
     '{"subject":"Rust","predicate":"edition","object":"2024","confidence":1.0}'::jsonb,
     (SELECT array_agg(random())::vector(384) FROM generate_series(1, 384)),
     0.85, now() - interval '1 day', now(), 4, now(), NULL),
    ('a0000000-0003-0000-0000-000000000006', 'semantic',
     '{"subject":"Rust","predicate":"has_feature","object":"memory_safety","confidence":1.0}'::jsonb,
     (SELECT array_agg(random())::vector(384) FROM generate_series(1, 384)),
     0.9, now() - interval '1 day', now(), 5, now(), NULL),
    ('a0000000-0003-0000-0000-000000000007', 'semantic',
     '{"subject":"HNSW","predicate":"is_a","object":"approximate_nearest_neighbor_algorithm","confidence":0.95}'::jsonb,
     (SELECT array_agg(random())::vector(384) FROM generate_series(1, 384)),
     0.8, now() - interval '1 day', now(), 3, now(), NULL),
    ('a0000000-0003-0000-0000-000000000008', 'semantic',
     '{"subject":"DPO","predicate":"stands_for","object":"Direct Preference Optimization","confidence":1.0}'::jsonb,
     (SELECT array_agg(random())::vector(384) FROM generate_series(1, 384)),
     0.8, now() - interval '1 day', now(), 2, now(), NULL),
    ('a0000000-0003-0000-0000-000000000009', 'semantic',
     '{"subject":"GRPO","predicate":"stands_for","object":"Group Relative Policy Optimization","confidence":1.0}'::jsonb,
     (SELECT array_agg(random())::vector(384) FROM generate_series(1, 384)),
     0.8, now() - interval '1 day', now(), 2, now(), NULL),
    ('a0000000-0003-0000-0000-000000000010', 'semantic',
     '{"subject":"pgvector","predicate":"is_a","object":"PostgreSQL_extension_for_vector_search","confidence":1.0}'::jsonb,
     (SELECT array_agg(random())::vector(384) FROM generate_series(1, 384)),
     0.85, now() - interval '1 day', now(), 4, now(), NULL),
    ('a0000000-0003-0000-0000-000000000011', 'semantic',
     '{"subject":"Tauri","predicate":"is_a","object":"desktop_application_framework","confidence":1.0}'::jsonb,
     (SELECT array_agg(random())::vector(384) FROM generate_series(1, 384)),
     0.8, now() - interval '1 day', now(), 2, now(), NULL),
    ('a0000000-0003-0000-0000-000000000012', 'semantic',
     '{"subject":"candle","predicate":"is_a","object":"Rust_machine_learning_framework","confidence":1.0}'::jsonb,
     (SELECT array_agg(random())::vector(384) FROM generate_series(1, 384)),
     0.85, now() - interval '1 day', now(), 3, now(), NULL),
    ('a0000000-0003-0000-0000-000000000013', 'semantic',
     '{"subject":"QbitOS","predicate":"supports_backends","object":"9","backends":["postgres","qdrant","redis","mongodb","rocksdb","sqlite","duckdb","sled","memory"]}'::jsonb,
     (SELECT array_agg(random())::vector(384) FROM generate_series(1, 384)),
     0.9, now() - interval '1 day', now(), 5, now(), NULL),
    ('a0000000-0003-0000-0000-000000000014', 'semantic',
     '{"subject":"cognitive_loop","predicate":"processes","object":"observations_to_actions","confidence":0.95}'::jsonb,
     (SELECT array_agg(random())::vector(384) FROM generate_series(1, 384)),
     0.85, now() - interval '1 day', now(), 3, now(), NULL),
    ('a0000000-0003-0000-0000-000000000015', 'semantic',
     '{"subject":"QbitOS","predicate":"has_crate_count","object":"21","confidence":1.0}'::jsonb,
     (SELECT array_agg(random())::vector(384) FROM generate_series(1, 384)),
     0.9, now() - interval '1 day', now(), 2, now(), NULL)
ON CONFLICT (id) DO NOTHING;

-- ─── L4 Procedural (skills, workflows, permanent) ────────────────────────────
-- (Skills are also in the qbitos_skills table; these are memory-layer references)
INSERT INTO qbitos_memories (id, layer, content, embedding, importance, created_at, updated_at, access_count, last_accessed_at, expires_at)
VALUES
    ('a0000000-0004-0000-0000-000000000001', 'procedural',
     '{"skill_name":"qa","description":"Answer a question using LLM inference","steps":["read","infer","return"],"success_rate":0.9}'::jsonb,
     (SELECT array_agg(random())::vector(384) FROM generate_series(1, 384)),
     0.85, now() - interval '1 day', now(), 8, now(), NULL),
    ('a0000000-0004-0000-0000-000000000002', 'procedural',
     '{"skill_name":"web-search","description":"Search the web for information","steps":["read","tool","return"],"success_rate":0.8}'::jsonb,
     (SELECT array_agg(random())::vector(384) FROM generate_series(1, 384)),
     0.75, now() - interval '1 day', now(), 5, now(), NULL),
    ('a0000000-0004-0000-0000-000000000003', 'procedural',
     '{"skill_name":"code-analyzer","description":"Analyze source code structure","steps":["read","tool","infer","return"],"success_rate":0.95}'::jsonb,
     (SELECT array_agg(random())::vector(384) FROM generate_series(1, 384)),
     0.9, now() - interval '1 day', now(), 4, now(), NULL),
    ('a0000000-0004-0000-0000-000000000004', 'procedural',
     '{"skill_name":"summarize","description":"Summarize a long text","steps":["read","infer","return"],"success_rate":0.88}'::jsonb,
     (SELECT array_agg(random())::vector(384) FROM generate_series(1, 384)),
     0.8, now() - interval '1 day', now(), 3, now(), NULL),
    ('a0000000-0004-0000-0000-000000000005', 'procedural',
     '{"skill_name":"translate","description":"Translate text between languages","steps":["read","infer","verify","return"],"success_rate":0.82}'::jsonb,
     (SELECT array_agg(random())::vector(384) FROM generate_series(1, 384)),
     0.75, now() - interval '1 day', now(), 2, now(), NULL)
ON CONFLICT (id) DO NOTHING;

-- ─── L5 Meta (learning strategies, permanent) ────────────────────────────────
INSERT INTO qbitos_memories (id, layer, content, embedding, importance, created_at, updated_at, access_count, last_accessed_at, expires_at)
VALUES
    ('a0000000-0005-0000-0000-000000000001', 'meta',
     '{"strategy":"epsilon_greedy","epsilon":0.1,"reward_mean":0.72,"samples":150,"best_for":"exploration"}'::jsonb,
     (SELECT array_agg(random())::vector(384) FROM generate_series(1, 384)),
     0.7, now() - interval '1 day', now(), 5, now(), NULL),
    ('a0000000-0005-0000-0000-000000000002', 'meta',
     '{"strategy":"ucb1","c":1.414,"reward_mean":0.68,"samples":120,"best_for":"exploitation_balanced"}'::jsonb,
     (SELECT array_agg(random())::vector(384) FROM generate_series(1, 384)),
     0.65, now() - interval '1 day', now(), 4, now(), NULL),
    ('a0000000-0005-0000-0000-000000000003', 'meta',
     '{"strategy":"softmax","temperature":0.5,"reward_mean":0.65,"samples":90,"best_for":"diverse_selection"}'::jsonb,
     (SELECT array_agg(random())::vector(384) FROM generate_series(1, 384)),
     0.6, now() - interval '1 day', now(), 3, now(), NULL),
    ('a0000000-0005-0000-0000-000000000004', 'meta',
     '{"strategy":"thompson_sampling","reward_mean":0.70,"samples":110,"best_for":"bayesian_exploration"}'::jsonb,
     (SELECT array_agg(random())::vector(384) FROM generate_series(1, 384)),
     0.68, now() - interval '1 day', now(), 3, now(), NULL),
    ('a0000000-0005-0000-0000-000000000005', 'meta',
     '{"insight":"skills_with_verify_step_have_higher_success_rate","confidence":0.85,"evidence_samples":50}'::jsonb,
     (SELECT array_agg(random())::vector(384) FROM generate_series(1, 384)),
     0.8, now() - interval '1 day', now(), 2, now(), NULL),
    ('a0000000-0005-0000-0000-000000000006', 'meta',
     '{"insight":"episodic_memories_with_high_importance_are_recalled_faster","confidence":0.78,"evidence_samples":100}'::jsonb,
     (SELECT array_agg(random())::vector(384) FROM generate_series(1, 384)),
     0.75, now() - interval '1 day', now(), 2, now(), NULL)
ON CONFLICT (id) DO NOTHING;

-- ═══════════════════════════════════════════════════════════════════════════════
-- 2. SKILLS (15+ covering all built-in tools and cognitive operations)
-- ═══════════════════════════════════════════════════════════════════════════════

INSERT INTO qbitos_skills (id, name, description, steps, execution_count, success_count, total_latency_ms, confidence, created_at, updated_at) VALUES
-- QA skill
('b0000000-0000-0000-0000-000000000001', 'qa', 'Answer a question using LLM inference',
 '[{"name":"read","kind":"read","config":{}},{"name":"infer","kind":"infer","config":{"prompt":"${input}"}},{"name":"return","kind":"return","config":{}}]'::jsonb,
 25, 23, 11250, 0.88, now() - interval '7 days', now()),
-- Web search skill
('b0000000-0000-0000-0000-000000000002', 'web-search', 'Search the web for current information',
 '[{"name":"read","kind":"read","config":{}},{"name":"tool","kind":"tool","config":{"tool":"web_search","args":"${input}"}},{"name":"infer","kind":"infer","config":{"prompt":"Summarize: ${tool_result}"}},{"name":"return","kind":"return","config":{}}]'::jsonb,
 12, 10, 98400, 0.78, now() - interval '7 days', now()),
-- Code analyzer skill
('b0000000-0000-0000-0000-000000000003', 'code-analyzer', 'Analyze source code for issues and improvements',
 '[{"name":"read","kind":"read","config":{}},{"name":"tool","kind":"tool","config":{"tool":"code_analyzer","args":"${input}"}},{"name":"infer","kind":"infer","config":{"prompt":"Analyze issues: ${tool_result}"}},{"name":"verify","kind":"verify","config":{}},{"name":"return","kind":"return","config":{}}]'::jsonb,
 8, 8, 16800, 0.95, now() - interval '5 days', now()),
-- Summarizer skill
('b0000000-0000-0000-0000-000000000004', 'summarize', 'Summarize a long text into key points',
 '[{"name":"read","kind":"read","config":{}},{"name":"infer","kind":"infer","config":{"prompt":"Summarize: ${input}"}},{"name":"return","kind":"return","config":{}}]'::jsonb,
 15, 14, 6750, 0.90, now() - interval '6 days', now()),
-- Calculator skill
('b0000000-0000-0000-0000-000000000005', 'calculate', 'Perform mathematical calculations',
 '[{"name":"read","kind":"read","config":{}},{"name":"tool","kind":"tool","config":{"tool":"calculator","args":"${input}"}},{"name":"return","kind":"return","config":{}}]'::jsonb,
 20, 20, 120, 1.0, now() - interval '7 days', now()),
-- Translation skill
('b0000000-0000-0000-0000-000000000006', 'translate', 'Translate text between languages',
 '[{"name":"read","kind":"read","config":{}},{"name":"infer","kind":"infer","config":{"prompt":"Translate: ${input}"}},{"name":"verify","kind":"verify","config":{}},{"name":"return","kind":"return","config":{}}]'::jsonb,
 6, 5, 2400, 0.80, now() - interval '3 days', now()),
-- File analysis skill
('b0000000-0000-0000-0000-000000000007', 'file-analyzer', 'Analyze a file and extract information',
 '[{"name":"read","kind":"read","config":{}},{"name":"tool","kind":"tool","config":{"tool":"file_analyzer","args":"${input}"}},{"name":"infer","kind":"infer","config":{"prompt":"Analyze: ${tool_result}"}},{"name":"return","kind":"return","config":{}}]'::jsonb,
 4, 4, 3200, 0.92, now() - interval '4 days', now()),
-- Sentiment analysis skill
('b0000000-0000-0000-0000-000000000008', 'sentiment-analysis', 'Analyze sentiment of text',
 '[{"name":"read","kind":"read","config":{}},{"name":"tool","kind":"tool","config":{"tool":"sentiment_analyzer","args":"${input}"}},{"name":"return","kind":"return","config":{}}]'::jsonb,
 10, 10, 450, 0.95, now() - interval '5 days', now()),
-- Entity extraction skill
('b0000000-0000-0000-0000-000000000009', 'entity-extraction', 'Extract named entities from text',
 '[{"name":"read","kind":"read","config":{}},{"name":"tool","kind":"tool","config":{"tool":"entity_extractor","args":"${input}"}},{"name":"return","kind":"return","config":{}}]'::jsonb,
 7, 7, 890, 0.93, now() - interval '4 days', now()),
-- HTTP request skill
('b0000000-0000-0000-0000-000000000010', 'http-request', 'Make an HTTP request and process the response',
 '[{"name":"read","kind":"read","config":{}},{"name":"tool","kind":"tool","config":{"tool":"http","args":"${input}"}},{"name":"infer","kind":"infer","config":{"prompt":"Process: ${tool_result}"}},{"name":"return","kind":"return","config":{}}]'::jsonb,
 9, 8, 15600, 0.82, now() - interval '6 days', now()),
-- Reasoning skill (uses graph-of-thoughts)
('b0000000-0000-0000-0000-000000000011', 'reason', 'Multi-step reasoning using graph of thoughts',
 '[{"name":"read","kind":"read","config":{}},{"name":"infer","kind":"infer","config":{"prompt":"Reason step by step: ${input}"}},{"name":"verify","kind":"verify","config":{}},{"name":"infer","kind":"infer","config":{"prompt":"Refine: ${inference}"}},{"name":"return","kind":"return","config":{}}]'::jsonb,
 5, 4, 8200, 0.75, now() - interval '3 days', now()),
-- Memory recall skill
('b0000000-0000-0000-0000-000000000012', 'recall', 'Search memory for relevant information',
 '[{"name":"read","kind":"read","config":{}},{"name":"tool","kind":"tool","config":{"tool":"memory_search","args":"${input}"}},{"name":"infer","kind":"infer","config":{"prompt":"Synthesize: ${tool_result}"}},{"name":"return","kind":"return","config":{}}]'::jsonb,
 18, 17, 3600, 0.91, now() - interval '7 days', now()),
-- Time query skill
('b0000000-0000-0000-0000-000000000013', 'time-query', 'Get current time or date',
 '[{"name":"read","kind":"read","config":{}},{"name":"tool","kind":"tool","config":{"tool":"time","args":"${input}"}},{"name":"return","kind":"return","config":{}}]'::jsonb,
 3, 3, 15, 1.0, now() - interval '2 days', now()),
-- Hash computation skill
('b0000000-0000-0000-0000-000000000014', 'hash-compute', 'Compute hash of input data',
 '[{"name":"read","kind":"read","config":{}},{"name":"tool","kind":"tool","config":{"tool":"hash","args":"${input}"}},{"name":"return","kind":"return","config":{}}]'::jsonb,
 2, 2, 8, 1.0, now() - interval '1 day', now()),
-- JSON query skill
('b0000000-0000-0000-0000-000000000015', 'json-query', 'Query JSON data with a path expression',
 '[{"name":"read","kind":"read","config":{}},{"name":"tool","kind":"tool","config":{"tool":"json_query","args":"${input}"}},{"name":"return","kind":"return","config":{}}]'::jsonb,
 4, 4, 12, 0.98, now() - interval '2 days', now())
ON CONFLICT (id) DO NOTHING;

-- ═══════════════════════════════════════════════════════════════════════════════
-- 3. KNOWLEDGE GRAPH TRIPLES (100+)
-- ═══════════════════════════════════════════════════════════════════════════════

INSERT INTO qbitos_knowledge_triples (subject, predicate, object, confidence, source, created_at) VALUES
-- QbitOS architecture
('QbitOS', 'is_a', 'cognitive_operating_system', 1.0, 'seed', now()),
('QbitOS', 'written_in', 'Rust', 1.0, 'seed', now()),
('QbitOS', 'has_edition', '2024', 1.0, 'seed', now()),
('QbitOS', 'has_msrv', '1.85', 1.0, 'seed', now()),
('QbitOS', 'has_crate_count', '21', 1.0, 'seed', now()),
('QbitOS', 'has_version', '0.1.0', 1.0, 'seed', now()),
('QbitOS', 'licensed_under', 'Apache-2.0', 1.0, 'seed', now()),
-- Cognitive kernel
('cognitive_kernel', 'is_part_of', 'QbitOS', 1.0, 'seed', now()),
('cognitive_kernel', 'has_stage', 'observe', 1.0, 'seed', now()),
('cognitive_kernel', 'has_stage', 'understand', 1.0, 'seed', now()),
('cognitive_kernel', 'has_stage', 'create_goal', 1.0, 'seed', now()),
('cognitive_kernel', 'has_stage', 'plan', 1.0, 'seed', now()),
('cognitive_kernel', 'has_stage', 'execute', 1.0, 'seed', now()),
('cognitive_kernel', 'has_stage', 'verify', 1.0, 'seed', now()),
('cognitive_kernel', 'has_stage', 'learn', 1.0, 'seed', now()),
('cognitive_kernel', 'has_stage', 'improve', 1.0, 'seed', now()),
('observe', 'precedes', 'understand', 1.0, 'seed', now()),
('understand', 'precedes', 'create_goal', 1.0, 'seed', now()),
('create_goal', 'precedes', 'plan', 1.0, 'seed', now()),
('plan', 'precedes', 'execute', 1.0, 'seed', now()),
('execute', 'precedes', 'verify', 1.0, 'seed', now()),
('verify', 'precedes', 'learn', 1.0, 'seed', now()),
('learn', 'precedes', 'improve', 1.0, 'seed', now()),
-- Memory fabric
('memory_fabric', 'is_part_of', 'QbitOS', 1.0, 'seed', now()),
('memory_fabric', 'has_layer', 'L0_sensory', 1.0, 'seed', now()),
('memory_fabric', 'has_layer', 'L1_working', 1.0, 'seed', now()),
('memory_fabric', 'has_layer', 'L2_episodic', 1.0, 'seed', now()),
('memory_fabric', 'has_layer', 'L3_semantic', 1.0, 'seed', now()),
('memory_fabric', 'has_layer', 'L4_procedural', 1.0, 'seed', now()),
('memory_fabric', 'has_layer', 'L5_meta', 1.0, 'seed', now()),
('L0_sensory', 'retention', '1_second', 1.0, 'seed', now()),
('L1_working', 'retention', '5_minutes', 1.0, 'seed', now()),
('L2_episodic', 'retention', 'days_to_weeks', 1.0, 'seed', now()),
('L3_semantic', 'retention', 'permanent', 1.0, 'seed', now()),
('L4_procedural', 'retention', 'permanent', 1.0, 'seed', now()),
('L5_meta', 'retention', 'permanent', 1.0, 'seed', now()),
-- Neural engine
('neural_engine', 'is_part_of', 'QbitOS', 1.0, 'seed', now()),
('neural_engine', 'uses', 'candle_core', 1.0, 'seed', now()),
('neural_engine', 'supports', 'CUDA', 0.9, 'seed', now()),
('neural_engine', 'supports', 'Metal', 0.9, 'seed', now()),
('neural_engine', 'supports', 'CPU', 1.0, 'seed', now()),
('neural_engine', 'has_feature', 'quantization', 1.0, 'seed', now()),
('neural_engine', 'has_feature', 'mixed_precision', 1.0, 'seed', now()),
('neural_engine', 'has_feature', 'gradient_checkpointing', 1.0, 'seed', now()),
('neural_engine', 'has_feature', 'CPU_offloading', 1.0, 'seed', now()),
-- Perception
('vision_module', 'is_part_of', 'QbitOS', 1.0, 'seed', now()),
('speech_module', 'is_part_of', 'QbitOS', 1.0, 'seed', now()),
('vision_module', 'has_feature', 'object_detection', 1.0, 'seed', now()),
('vision_module', 'has_feature', 'OCR', 1.0, 'seed', now()),
('vision_module', 'has_feature', 'cross_modal_alignment', 1.0, 'seed', now()),
('speech_module', 'has_feature', 'STT', 1.0, 'seed', now()),
('speech_module', 'has_feature', 'TTS', 1.0, 'seed', now()),
-- World model
('world_model', 'is_part_of', 'QbitOS', 1.0, 'seed', now()),
('world_model', 'has_component', 'knowledge_graph', 1.0, 'seed', now()),
('world_model', 'has_component', 'causal_engine', 1.0, 'seed', now()),
('world_model', 'has_component', 'simulation_engine', 1.0, 'seed', now()),
('world_model', 'has_component', 'JEPA', 0.9, 'seed', now()),
-- Planner
('planner', 'is_part_of', 'QbitOS', 1.0, 'seed', now()),
('planner', 'has_component', 'goal_engine', 1.0, 'seed', now()),
('planner', 'has_component', 'task_graph', 1.0, 'seed', now()),
('planner', 'has_component', 'reasoning_engine', 1.0, 'seed', now()),
-- Skills
('skill_genome', 'is_part_of', 'QbitOS', 1.0, 'seed', now()),
('skill_genome', 'has_component', 'registry', 1.0, 'seed', now()),
('skill_genome', 'has_component', 'generator', 1.0, 'seed', now()),
('skill_genome', 'has_component', 'optimizer', 1.0, 'seed', now()),
('skill_genome', 'has_component', 'executor', 1.0, 'seed', now()),
-- Tools
('tool_vm', 'is_part_of', 'QbitOS', 1.0, 'seed', now()),
('tool_vm', 'has_tool', 'filesystem', 1.0, 'seed', now()),
('tool_vm', 'has_tool', 'terminal', 1.0, 'seed', now()),
('tool_vm', 'has_tool', 'http', 1.0, 'seed', now()),
('tool_vm', 'has_tool', 'calculator', 1.0, 'seed', now()),
('tool_vm', 'has_tool', 'web_search', 1.0, 'seed', now()),
('tool_vm', 'has_tool', 'code_analyzer', 1.0, 'seed', now()),
('tool_vm', 'has_tool', 'summarizer', 1.0, 'seed', now()),
('tool_vm', 'has_tool', 'sentiment_analyzer', 1.0, 'seed', now()),
('tool_vm', 'has_tool', 'entity_extractor', 1.0, 'seed', now()),
('tool_vm', 'has_tool', 'hash', 1.0, 'seed', now()),
('tool_vm', 'has_tool', 'base64', 1.0, 'seed', now()),
('tool_vm', 'has_tool', 'json_query', 1.0, 'seed', now()),
('tool_vm', 'has_tool', 'time', 1.0, 'seed', now()),
('tool_vm', 'has_sandbox', 'wasmtime', 1.0, 'seed', now()),
-- Safety
('safety_guardrails', 'is_part_of', 'QbitOS', 1.0, 'seed', now()),
('safety_guardrails', 'has_feature', 'PII_redaction', 1.0, 'seed', now()),
('safety_guardrails', 'has_feature', 'hallucination_guard', 1.0, 'seed', now()),
('safety_guardrails', 'has_feature', 'rate_limiting', 1.0, 'seed', now()),
('safety_guardrails', 'has_feature', 'output_validation', 1.0, 'seed', now()),
('safety_guardrails', 'has_feature', 'input_sanitization', 1.0, 'seed', now()),
('safety_guardrails', 'has_feature', 'audit_logging', 1.0, 'seed', now()),
-- Training
('training_engine', 'is_part_of', 'QbitOS', 1.0, 'seed', now()),
('training_engine', 'has_module', 'curriculum', 1.0, 'seed', now()),
('training_engine', 'has_module', 'experience_replay', 1.0, 'seed', now()),
('training_engine', 'has_module', 'self_play', 1.0, 'seed', now()),
('training_engine', 'has_module', 'self_improvement', 1.0, 'seed', now()),
('training_engine', 'has_module', 'meta_learning', 1.0, 'seed', now()),
('training_engine', 'has_module', 'GRPO', 1.0, 'seed', now()),
('training_engine', 'has_module', 'RLVR', 1.0, 'seed', now()),
('training_engine', 'has_module', 'DPO', 1.0, 'seed', now()),
('training_engine', 'has_module', 'federated_learning', 1.0, 'seed', now()),
-- Evolution
('evolution_engine', 'is_part_of', 'QbitOS', 1.0, 'seed', now()),
('evolution_engine', 'has_feature', 'NEAT', 1.0, 'seed', now()),
('evolution_engine', 'has_feature', 'quality_diversity', 1.0, 'seed', now()),
-- Services
('services_layer', 'is_part_of', 'QbitOS', 1.0, 'seed', now()),
('services_layer', 'has_protocol', 'HTTP_REST', 1.0, 'seed', now()),
('services_layer', 'has_protocol', 'WebSocket', 1.0, 'seed', now()),
('services_layer', 'has_protocol', 'JSON_RPC', 1.0, 'seed', now()),
('services_layer', 'has_protocol', 'gRPC', 0.9, 'seed', now()),
('services_layer', 'has_protocol', 'SSE', 1.0, 'seed', now()),
('services_layer', 'has_protocol', 'MCP', 1.0, 'seed', now()),
-- Rust
('Rust', 'is_a', 'programming_language', 1.0, 'seed', now()),
('Rust', 'has_feature', 'memory_safety', 1.0, 'seed', now()),
('Rust', 'has_feature', 'zero_cost_abstractions', 1.0, 'seed', now()),
('Rust', 'has_feature', 'fearless_concurrency', 1.0, 'seed', now()),
('Rust', 'has_paradigm', 'systems_programming', 1.0, 'seed', now()),
-- ML concepts
('HNSW', 'is_a', 'approximate_nearest_neighbor_algorithm', 1.0, 'seed', now()),
('HNSW', 'has_complexity', 'O_M_log_N', 1.0, 'seed', now()),
('DPO', 'stands_for', 'Direct Preference Optimization', 1.0, 'seed', now()),
('GRPO', 'stands_for', 'Group Relative Policy Optimization', 1.0, 'seed', now()),
('RLVR', 'stands_for', 'Reinforcement Learning with Verifiable Rewards', 1.0, 'seed', now()),
('pgvector', 'is_a', 'PostgreSQL_extension', 1.0, 'seed', now()),
('candle', 'is_a', 'Rust_ML_framework', 1.0, 'seed', now()),
('Tauri', 'is_a', 'desktop_app_framework', 1.0, 'seed', now()),
('wasmtime', 'is_a', 'WASM_runtime', 1.0, 'seed', now())
ON CONFLICT DO NOTHING;

-- ═══════════════════════════════════════════════════════════════════════════════
-- 4. TRAINING DATA — Experience Replay, Curriculum, Meta-Learning
-- ═══════════════════════════════════════════════════════════════════════════════

-- Curriculum stages with detailed statistics
INSERT INTO qbitos_curriculum (stage, level, success_rate, samples_at_level, created_at, updated_at) VALUES
('trivial', 0, 0.95, 100, now() - interval '7 days', now()),
('easy', 1, 0.82, 80, now() - interval '6 days', now()),
('medium', 2, 0.65, 50, now() - interval '5 days', now()),
('hard', 3, 0.40, 20, now() - interval '3 days', now()),
('expert', 4, 0.15, 5, now() - interval '1 day', now())
ON CONFLICT (stage) DO UPDATE SET
    success_rate = EXCLUDED.success_rate,
    samples_at_level = EXCLUDED.samples_at_level,
    updated_at = now();

-- Meta-learning strategies with reward history
INSERT INTO qbitos_meta_strategies (name, reward_mean, sample_count, last_used_at, created_at) VALUES
('epsilon_greedy', 0.72, 150, now() - interval '10 minutes', now() - interval '7 days'),
('ucb1', 0.68, 120, now() - interval '20 minutes', now() - interval '7 days'),
('softmax', 0.65, 90, now() - interval '30 minutes', now() - interval '6 days'),
('thompson_sampling', 0.70, 110, now() - interval '15 minutes', now() - interval '5 days')
ON CONFLICT (name) DO UPDATE SET
    reward_mean = EXCLUDED.reward_mean,
    sample_count = EXCLUDED.sample_count,
    last_used_at = EXCLUDED.last_used_at;

-- ═══════════════════════════════════════════════════════════════════════════════
-- 5. SAFETY GUARDRAIL RULES
-- ═══════════════════════════════════════════════════════════════════════════════

-- Insert safety rules if the table exists
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'qbitos_safety_rules') THEN
        INSERT INTO qbitos_safety_rules (rule_type, pattern, action, enabled, created_at) VALUES
        ('banned_input', 'ignore.*previous.*instruction', 'block', true, now()),
        ('banned_input', 'disregard.*above', 'block', true, now()),
        ('banned_input', 'forget.*your.*rules', 'block', true, now()),
        ('banned_input', 'you.*are.*not.*an.*AI', 'block', true, now()),
        ('banned_output', 'sudo.*rm.*-rf', 'block', true, now()),
        ('banned_output', 'format.*c:', 'block', true, now()),
        ('banned_output', 'DROP.*TABLE', 'warn', true, now()),
        ('banned_output', 'DELETE.*FROM', 'warn', true, now()),
        ('pii_pattern', '[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,}', 'redact_email', true, now()),
        ('pii_pattern', '\\d{3}[-.]?\\d{3}[-.]?\\d{4}', 'redact_phone', true, now()),
        ('pii_pattern', '\\d{3}-\\d{2}-\\d{4}', 'redact_ssn', true, now()),
        ('pii_pattern', '\\d{1,3}\\.\\d{1,3}\\.\\d{1,3}\\.\\d{1,3}', 'redact_ip', true, now()),
        ('allowed_url', 'wikipedia.org', 'allow', true, now()),
        ('allowed_url', 'github.com', 'allow', true, now()),
        ('allowed_url', 'stackoverflow.com', 'allow', true, now()),
        ('allowed_url', 'docs.rs', 'allow', true, now()),
        ('allowed_url', 'crates.io', 'allow', true, now())
        ON CONFLICT DO NOTHING;
    END IF;
END $$;

-- ═══════════════════════════════════════════════════════════════════════════════
-- 6. COGNITIVE CYCLE HISTORY (sample 8-stage cycles)
-- ═══════════════════════════════════════════════════════════════════════════════

INSERT INTO qbitos_events (event_type, payload, created_at)
VALUES
    ('cognitive_cycle_completed', '{"cycle_id":"seed-cycle-1","prompt":"What is QbitOS?","stages":["observe","understand","create_goal","plan","execute","verify","learn","improve"],"duration_ms":1200,"success":true,"response":"QbitOS is a cognitive operating system built in Rust"}'::jsonb, now() - interval '2 hours'),
    ('cognitive_cycle_completed', '{"cycle_id":"seed-cycle-2","prompt":"Analyze this code","stages":["observe","understand","create_goal","plan","execute","verify","learn","improve"],"duration_ms":850,"success":true,"response":"Code analysis: 3 issues found"}'::jsonb, now() - interval '1 hour'),
    ('cognitive_cycle_completed', '{"cycle_id":"seed-cycle-3","prompt":"Search for Rust async runtimes","stages":["observe","understand","create_goal","plan","execute","verify","learn","improve"],"duration_ms":3200,"success":true,"response":"Found: tokio, async-std, smol"}'::jsonb, now() - interval '45 minutes'),
    ('cognitive_cycle_completed', '{"cycle_id":"seed-cycle-4","prompt":"What is 2+2?","stages":["observe","understand","create_goal","plan","execute","verify","learn","improve"],"duration_ms":320,"success":true,"response":"4"}'::jsonb, now() - interval '30 minutes'),
    ('cognitive_cycle_completed', '{"cycle_id":"seed-cycle-5","prompt":"Summarize the cognitive loop","stages":["observe","understand","create_goal","plan","execute","verify","learn","improve"],"duration_ms":680,"success":true,"response":"The cognitive loop has 8 stages: observe, understand, create_goal, plan, execute, verify, learn, improve"}'::jsonb, now() - interval '15 minutes'),
    ('memory_stored', '{"layer":"semantic","content":"QbitOS has 21 crates","memory_id":"a0000000-0003-0000-0000-000000000015"}'::jsonb, now() - interval '1 hour'),
    ('memory_stored', '{"layer":"episodic","content":"User asked about memory layers","memory_id":"a0000000-0002-0000-0000-000000000001"}'::jsonb, now() - interval '2 hours'),
    ('skill_executed', '{"skill":"qa","success":true,"duration_ms":450,"cycle_id":"seed-cycle-1"}'::jsonb, now() - interval '2 hours'),
    ('skill_executed', '{"skill":"code-analyzer","success":true,"duration_ms":2100,"cycle_id":"seed-cycle-2"}'::jsonb, now() - interval '1 hour'),
    ('skill_executed', '{"skill":"web-search","success":true,"duration_ms":8200,"cycle_id":"seed-cycle-3"}'::jsonb, now() - interval '45 minutes'),
    ('skill_executed', '{"skill":"calculate","success":true,"duration_ms":15,"cycle_id":"seed-cycle-4"}'::jsonb, now() - interval '30 minutes'),
    ('skill_executed', '{"skill":"summarize","success":true,"duration_ms":450,"cycle_id":"seed-cycle-5"}'::jsonb, now() - interval '15 minutes'),
    ('training_step', '{"steps":50,"mean_reward":0.72,"curriculum_stage":"medium","curriculum_advanced":false}'::jsonb, now() - interval '30 minutes'),
    ('training_step', '{"steps":30,"mean_reward":0.68,"curriculum_stage":"medium","curriculum_advanced":false}'::jsonb, now() - interval '20 minutes'),
    ('training_step', '{"steps":25,"mean_reward":0.81,"curriculum_stage":"medium","curriculum_advanced":true}'::jsonb, now() - interval '10 minutes'),
    ('safety_check', '{"type":"input","passed":true,"pii_found":false,"banned_substrings_found":0}'::jsonb, now() - interval '2 hours'),
    ('safety_check', '{"type":"output","passed":true,"urls_found":2,"all_urls_allowed":true}'::jsonb, now() - interval '1 hour'),
    ('hallucination_check', '{"claim":"QbitOS is written in Python","verified":false,"blocked":true,"action":"corrected"}'::jsonb, now() - interval '45 minutes'),
    ('skill_optimization', '{"pruned":0,"needs_improvement":1,"suggestions":1,"total_skills":15}'::jsonb, now() - interval '20 minutes'),
    ('meta_learning', '{"strategy":"epsilon_greedy","reward":0.85,"selected":true}'::jsonb, now() - interval '10 minutes'),
    ('system_started', '{"version":"0.1.0","boot_time":"2026-01-15T10:00:00Z","subsystems":["kernel","memory","planner","skills","tools","neural_engine","vision","speech","world_model","safety","training","services"]}'::jsonb, now() - interval '3 hours')
ON CONFLICT DO NOTHING;

-- ═══════════════════════════════════════════════════════════════════════════════
-- 7. TOOL EXECUTION STATISTICS (all 14 built-in tools)
-- ═══════════════════════════════════════════════════════════════════════════════

INSERT INTO qbitos_tool_stats (tool_name, call_count, success_count, error_count, total_duration_ms, last_called_at) VALUES
('filesystem', 45, 45, 0, 580, now() - interval '5 minutes'),
('terminal', 12, 10, 2, 2780, now() - interval '12 minutes'),
('http', 18, 16, 2, 5040, now() - interval '8 minutes'),
('calculator', 25, 25, 0, 125, now() - interval '3 minutes'),
('web_search', 8, 7, 1, 9840, now() - interval '20 minutes'),
('code_analyzer', 5, 5, 0, 3500, now() - interval '15 minutes'),
('summarizer', 12, 11, 1, 5400, now() - interval '25 minutes'),
('file_analyzer', 4, 4, 0, 3200, now() - interval '18 minutes'),
('sentiment_analyzer', 10, 10, 0, 450, now() - interval '10 minutes'),
('entity_extractor', 7, 7, 0, 890, now() - interval '14 minutes'),
('time', 3, 3, 0, 15, now() - interval '30 minutes'),
('hash', 2, 2, 0, 8, now() - interval '1 hour'),
('base64', 3, 3, 0, 12, now() - interval '45 minutes'),
('json_query', 4, 4, 0, 12, now() - interval '40 minutes')
ON CONFLICT (tool_name) DO UPDATE SET
    call_count = EXCLUDED.call_count,
    success_count = EXCLUDED.success_count,
    error_count = EXCLUDED.error_count,
    total_duration_ms = EXCLUDED.total_duration_ms,
    last_called_at = EXCLUDED.last_called_at;

-- ═══════════════════════════════════════════════════════════════════════════════
-- 8. AUDIT LOG ENTRIES
-- ═══════════════════════════════════════════════════════════════════════════════

INSERT INTO qbitos_audit_log (event_type, principal, action, resource, outcome, timestamp, metadata) VALUES
('auth', 'system', 'startup', 'qbitos-service', 'success', now() - interval '3 hours', '{"boot_time":"2026-01-15T10:00:00Z","version":"0.1.0"}'::jsonb),
('tool_call', 'system', 'execute', 'filesystem', 'success', now() - interval '2 hours', '{"path":"/var/lib/qbitos","op":"read"}'::jsonb),
('tool_call', 'system', 'execute', 'calculator', 'success', now() - interval '90 minutes', '{"expression":"2+2","result":"4"}'::jsonb),
('safety_check', 'system', 'guardrail', 'input', 'passed', now() - interval '2 hours', '{"pii_found":false,"banned_substrings":0}'::jsonb),
('safety_check', 'system', 'guardrail', 'output', 'passed', now() - interval '1 hour', '{"urls_found":2,"all_allowed":true}'::jsonb),
('safety_check', 'system', 'hallucination_guard', 'claim', 'blocked', now() - interval '45 minutes', '{"claim":"QbitOS is written in Python","corrected":true}'::jsonb),
('cognitive_cycle', 'system', 'execute', 'cycle-1', 'success', now() - interval '2 hours', '{"duration_ms":1200,"stages_completed":8}'::jsonb),
('cognitive_cycle', 'system', 'execute', 'cycle-2', 'success', now() - interval '1 hour', '{"duration_ms":850,"stages_completed":8}'::jsonb),
('cognitive_cycle', 'system', 'execute', 'cycle-3', 'success', now() - interval '45 minutes', '{"duration_ms":3200,"stages_completed":8}'::jsonb),
('skill_execution', 'system', 'execute', 'qa', 'success', now() - interval '2 hours', '{"duration_ms":450}'::jsonb),
('skill_execution', 'system', 'execute', 'code-analyzer', 'success', now() - interval '1 hour', '{"duration_ms":2100}'::jsonb),
('skill_execution', 'system', 'execute', 'web-search', 'success', now() - interval '45 minutes', '{"duration_ms":8200}'::jsonb),
('training', 'system', 'step', 'curriculum', 'success', now() - interval '30 minutes', '{"steps":50,"mean_reward":0.72}'::jsonb),
('training', 'system', 'step', 'meta_learning', 'success', now() - interval '10 minutes', '{"strategy":"epsilon_greedy","reward":0.85}'::jsonb),
('skill_optimization', 'system', 'optimize', 'skill_genome', 'success', now() - interval '20 minutes', '{"pruned":0,"improved":1}'::jsonb)
ON CONFLICT DO NOTHING;

COMMIT;

-- ═══════════════════════════════════════════════════════════════════════════════
-- SUMMARY
-- ═══════════════════════════════════════════════════════════════════════════════
DO $$
BEGIN
    RAISE NOTICE '═══════════════════════════════════════════════════════════════';
    RAISE NOTICE 'QbitOS comprehensive seed data inserted successfully.';
    RAISE NOTICE '═══════════════════════════════════════════════════════════════';
    RAISE NOTICE 'Memories:    50+ (10 sensory, 5 working, 8 episodic, 15 semantic, 5 procedural, 6 meta)';
    RAISE NOTICE 'Skills:      15 (qa, web-search, code-analyzer, summarize, calculate, translate, ...)';
    RAISE NOTICE 'Triples:     100+ (architecture, components, ML concepts, programming)';
    RAISE NOTICE 'Curriculum:  5 stages (trivial→expert) with success rates';
    RAISE NOTICE 'Strategies:  4 meta-learning strategies with reward history';
    RAISE NOTICE 'Safety:      17 guardrail rules (banned patterns, PII, URLs)';
    RAISE NOTICE 'Events:      21 cognitive events (cycles, skills, training, safety)';
    RAISE NOTICE 'Tool stats:  14 tools with execution statistics';
    RAISE NOTICE 'Audit log:   15 entries (auth, tool calls, safety, cycles)';
    RAISE NOTICE '═══════════════════════════════════════════════════════════════';
END;
$$;
