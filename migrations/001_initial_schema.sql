-- Migration: 001_initial_schema
-- Description: Initial database schema for Conduit routing system
-- Source: Alembic migration 9a6c8a59cb30
-- Apply: Run this SQL in Supabase SQL Editor

-- ============================================================================
-- QUERIES TABLE
-- ============================================================================

CREATE TABLE queries (
    id TEXT PRIMARY KEY,
    text TEXT NOT NULL CHECK (length(trim(text)) > 0),
    user_id TEXT,
    context JSONB,
    constraints JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_queries_user_id ON queries(user_id);
CREATE INDEX idx_queries_created_at ON queries(created_at DESC);

-- ============================================================================
-- ROUTING_DECISIONS TABLE
-- ============================================================================

CREATE TABLE routing_decisions (
    id TEXT PRIMARY KEY,
    query_id TEXT NOT NULL REFERENCES queries(id) ON DELETE CASCADE,
    selected_model TEXT NOT NULL,
    confidence NUMERIC(3,2) NOT NULL CHECK (confidence >= 0.0 AND confidence <= 1.0),
    features JSONB NOT NULL,
    reasoning TEXT NOT NULL,
    metadata JSONB DEFAULT '{}'::jsonb NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_routing_decisions_query_id ON routing_decisions(query_id);
CREATE INDEX idx_routing_decisions_model ON routing_decisions(selected_model);
CREATE INDEX idx_routing_decisions_created_at ON routing_decisions(created_at DESC);
CREATE INDEX idx_routing_decisions_confidence ON routing_decisions(confidence DESC);

-- ============================================================================
-- RESPONSES TABLE
-- ============================================================================

CREATE TABLE responses (
    id TEXT PRIMARY KEY,
    query_id TEXT NOT NULL REFERENCES queries(id) ON DELETE CASCADE,
    model TEXT NOT NULL,
    text TEXT NOT NULL,
    cost NUMERIC(10,6) NOT NULL CHECK (cost >= 0.0),
    latency NUMERIC(10,3) NOT NULL CHECK (latency >= 0.0),
    tokens INTEGER NOT NULL CHECK (tokens >= 0),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_responses_query_id ON responses(query_id);
CREATE INDEX idx_responses_model ON responses(model);
CREATE INDEX idx_responses_created_at ON responses(created_at DESC);
CREATE INDEX idx_responses_cost ON responses(cost DESC);
CREATE INDEX idx_responses_latency ON responses(latency DESC);

-- ============================================================================
-- FEEDBACK TABLE
-- ============================================================================

CREATE TABLE feedback (
    id TEXT PRIMARY KEY,
    response_id TEXT NOT NULL REFERENCES responses(id) ON DELETE CASCADE,
    quality_score NUMERIC(3,2) NOT NULL CHECK (quality_score >= 0.0 AND quality_score <= 1.0),
    user_rating INTEGER CHECK (user_rating IS NULL OR (user_rating >= 1 AND user_rating <= 5)),
    met_expectations BOOLEAN NOT NULL,
    comments TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_feedback_response_id ON feedback(response_id);
CREATE INDEX idx_feedback_created_at ON feedback(created_at DESC);
CREATE INDEX idx_feedback_quality_score ON feedback(quality_score DESC);
CREATE INDEX idx_feedback_met_expectations ON feedback(met_expectations);

-- ============================================================================
-- IMPLICIT_FEEDBACK TABLE (Phase 2+)
-- ============================================================================

CREATE TABLE implicit_feedback (
    id TEXT PRIMARY KEY,
    response_id TEXT NOT NULL REFERENCES responses(id) ON DELETE CASCADE,
    query_id TEXT NOT NULL REFERENCES queries(id) ON DELETE CASCADE,
    retry_detected BOOLEAN DEFAULT FALSE NOT NULL,
    retry_delay_seconds NUMERIC(10,3) CHECK (retry_delay_seconds IS NULL OR retry_delay_seconds >= 0.0),
    task_abandoned BOOLEAN DEFAULT FALSE NOT NULL,
    latency_accepted BOOLEAN DEFAULT TRUE NOT NULL,
    error_occurred BOOLEAN DEFAULT FALSE NOT NULL,
    error_type TEXT,
    response_used BOOLEAN DEFAULT TRUE NOT NULL,
    followup_queries INTEGER DEFAULT 0 NOT NULL CHECK (followup_queries >= 0),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_implicit_feedback_response_id ON implicit_feedback(response_id);
CREATE INDEX idx_implicit_feedback_query_id ON implicit_feedback(query_id);
CREATE INDEX idx_implicit_feedback_created_at ON implicit_feedback(created_at DESC);
CREATE INDEX idx_implicit_feedback_retry_detected ON implicit_feedback(retry_detected);
CREATE INDEX idx_implicit_feedback_error_occurred ON implicit_feedback(error_occurred);

-- ============================================================================
-- MODEL_STATES TABLE
-- ============================================================================

CREATE TABLE model_states (
    model_id TEXT PRIMARY KEY,
    alpha NUMERIC(20,10) NOT NULL CHECK (alpha > 0),
    beta NUMERIC(20,10) NOT NULL CHECK (beta > 0),
    total_requests INTEGER DEFAULT 0 NOT NULL CHECK (total_requests >= 0),
    total_cost NUMERIC(10,6) DEFAULT 0.0 NOT NULL CHECK (total_cost >= 0.0),
    avg_quality NUMERIC(3,2) DEFAULT 0.0 NOT NULL CHECK (avg_quality >= 0.0 AND avg_quality <= 1.0),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_model_states_updated_at ON model_states(updated_at DESC);
CREATE INDEX idx_model_states_avg_quality ON model_states(avg_quality DESC);

-- ============================================================================
-- VIEWS
-- ============================================================================

CREATE OR REPLACE VIEW recent_routing_performance AS
SELECT
    rd.selected_model,
    COUNT(*) as total_routes,
    AVG(rd.confidence) as avg_confidence,
    AVG(r.cost) as avg_cost,
    AVG(r.latency) as avg_latency,
    AVG(CASE WHEN f.met_expectations THEN 1.0 ELSE 0.0 END) as success_rate
FROM routing_decisions rd
LEFT JOIN responses r ON r.query_id = rd.query_id
LEFT JOIN feedback f ON f.response_id = r.id
WHERE rd.created_at > NOW() - INTERVAL '7 days'
GROUP BY rd.selected_model
ORDER BY total_routes DESC;

-- ============================================================================
-- ALEMBIC VERSION TRACKING (if using Alembic)
-- ============================================================================

CREATE TABLE IF NOT EXISTS alembic_version (
    version_num VARCHAR(32) NOT NULL,
    CONSTRAINT alembic_version_pkc PRIMARY KEY (version_num)
);

INSERT INTO alembic_version (version_num) VALUES ('9a6c8a59cb30');

-- ============================================================================
-- MIGRATION COMPLETE
-- ============================================================================

-- Verify tables
SELECT tablename
FROM pg_tables
WHERE schemaname = 'public'
AND tablename IN ('queries', 'routing_decisions', 'responses', 'feedback', 'implicit_feedback', 'model_states')
ORDER BY tablename;
