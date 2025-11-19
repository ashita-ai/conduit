-- Conduit Initial Database Schema
-- Supabase PostgreSQL Migration
-- Version: 001
-- Date: 2025-11-18

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Queries table
CREATE TABLE queries (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    text TEXT NOT NULL,
    user_id TEXT,
    context JSONB,
    constraints JSONB,
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_queries_user_id ON queries(user_id);
CREATE INDEX idx_queries_created_at ON queries(created_at DESC);

-- Routing decisions table
CREATE TABLE routing_decisions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    query_id UUID NOT NULL REFERENCES queries(id) ON DELETE CASCADE,
    selected_model TEXT NOT NULL,
    confidence FLOAT NOT NULL CHECK (confidence >= 0.0 AND confidence <= 1.0),
    features JSONB NOT NULL,
    reasoning TEXT NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_routing_decisions_query_id ON routing_decisions(query_id);
CREATE INDEX idx_routing_decisions_model ON routing_decisions(selected_model);
CREATE INDEX idx_routing_decisions_created_at ON routing_decisions(created_at DESC);

-- Responses table
CREATE TABLE responses (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    query_id UUID NOT NULL REFERENCES queries(id) ON DELETE CASCADE,
    model TEXT NOT NULL,
    text TEXT NOT NULL,
    cost FLOAT NOT NULL CHECK (cost >= 0.0),
    latency FLOAT NOT NULL CHECK (latency >= 0.0),
    tokens INT NOT NULL CHECK (tokens >= 0),
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_responses_query_id ON responses(query_id);
CREATE INDEX idx_responses_model ON responses(model);
CREATE INDEX idx_responses_created_at ON responses(created_at DESC);

-- Feedback table
CREATE TABLE feedback (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    response_id UUID NOT NULL REFERENCES responses(id) ON DELETE CASCADE,
    quality_score FLOAT NOT NULL CHECK (quality_score >= 0.0 AND quality_score <= 1.0),
    user_rating INT CHECK (user_rating IS NULL OR (user_rating >= 1 AND user_rating <= 5)),
    met_expectations BOOLEAN NOT NULL,
    comments TEXT,
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_feedback_response_id ON feedback(response_id);
CREATE INDEX idx_feedback_created_at ON feedback(created_at DESC);

-- Model states table (Thompson Sampling Beta parameters)
CREATE TABLE model_states (
    model_id TEXT PRIMARY KEY,
    alpha FLOAT NOT NULL DEFAULT 1.0 CHECK (alpha > 0.0),
    beta FLOAT NOT NULL DEFAULT 1.0 CHECK (beta > 0.0),
    total_requests INT NOT NULL DEFAULT 0 CHECK (total_requests >= 0),
    total_cost FLOAT NOT NULL DEFAULT 0.0 CHECK (total_cost >= 0.0),
    avg_quality FLOAT NOT NULL DEFAULT 0.0 CHECK (avg_quality >= 0.0 AND avg_quality <= 1.0),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_model_states_updated_at ON model_states(updated_at DESC);

-- Initialize default model states
INSERT INTO model_states (model_id, alpha, beta, total_requests, total_cost, avg_quality)
VALUES
    ('gpt-4o-mini', 1.0, 1.0, 0, 0.0, 0.0),
    ('gpt-4o', 1.0, 1.0, 0, 0.0, 0.0),
    ('claude-sonnet-4', 1.0, 1.0, 0, 0.0, 0.0),
    ('claude-opus-4', 1.0, 1.0, 0, 0.0, 0.0)
ON CONFLICT (model_id) DO NOTHING;

-- Create view for analytics
CREATE VIEW routing_analytics AS
SELECT
    r.model,
    COUNT(*) as total_requests,
    AVG(r.cost) as avg_cost,
    AVG(r.latency) as avg_latency,
    AVG(r.tokens) as avg_tokens,
    AVG(COALESCE(f.quality_score, 0.0)) as avg_quality,
    COUNT(f.id) as feedback_count
FROM responses r
LEFT JOIN feedback f ON r.id = f.response_id
GROUP BY r.model;

-- Comments for documentation
COMMENT ON TABLE queries IS 'User queries submitted for LLM routing';
COMMENT ON TABLE routing_decisions IS 'ML-powered routing decisions (Thompson Sampling)';
COMMENT ON TABLE responses IS 'LLM responses with cost and latency tracking';
COMMENT ON TABLE feedback IS 'User feedback for quality scoring and model updates';
COMMENT ON TABLE model_states IS 'Thompson Sampling Beta distribution parameters per model';
COMMENT ON VIEW routing_analytics IS 'Aggregated analytics for routing performance';
