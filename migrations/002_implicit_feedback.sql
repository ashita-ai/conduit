-- Conduit Implicit Feedback Schema
-- Adds implicit behavioral signal tracking for ML improvements
-- Version: 002
-- Date: 2025-11-19
-- Dependencies: 001_initial_schema.sql

-- Implicit feedback table (behavioral signals)
CREATE TABLE implicit_feedback (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    query_id UUID NOT NULL REFERENCES queries(id) ON DELETE CASCADE,
    model_id TEXT NOT NULL,
    timestamp TIMESTAMP NOT NULL DEFAULT NOW(),

    -- Error signals
    error_occurred BOOLEAN NOT NULL DEFAULT FALSE,
    error_type TEXT,

    -- Latency signals
    latency_seconds FLOAT NOT NULL CHECK (latency_seconds >= 0.0),
    latency_accepted BOOLEAN NOT NULL DEFAULT TRUE,
    latency_tolerance TEXT CHECK (latency_tolerance IN ('high', 'medium', 'low')),

    -- Retry signals
    retry_detected BOOLEAN NOT NULL DEFAULT FALSE,
    retry_delay_seconds FLOAT CHECK (retry_delay_seconds IS NULL OR retry_delay_seconds >= 0.0),
    similarity_score FLOAT CHECK (similarity_score IS NULL OR (similarity_score >= 0.0 AND similarity_score <= 1.0)),
    original_query_id UUID REFERENCES queries(id),

    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

-- Indexes for efficient queries
CREATE INDEX idx_implicit_feedback_query_id ON implicit_feedback(query_id);
CREATE INDEX idx_implicit_feedback_model_id ON implicit_feedback(model_id);
CREATE INDEX idx_implicit_feedback_timestamp ON implicit_feedback(timestamp DESC);
CREATE INDEX idx_implicit_feedback_error_occurred ON implicit_feedback(error_occurred) WHERE error_occurred = TRUE;
CREATE INDEX idx_implicit_feedback_retry_detected ON implicit_feedback(retry_detected) WHERE retry_detected = TRUE;

-- View for implicit feedback analytics
CREATE VIEW implicit_feedback_analytics AS
SELECT
    if.model_id,
    COUNT(*) as total_signals,

    -- Error metrics
    COUNT(*) FILTER (WHERE if.error_occurred) as error_count,
    ROUND(100.0 * COUNT(*) FILTER (WHERE if.error_occurred) / COUNT(*), 2) as error_rate_pct,

    -- Retry metrics
    COUNT(*) FILTER (WHERE if.retry_detected) as retry_count,
    ROUND(100.0 * COUNT(*) FILTER (WHERE if.retry_detected) / COUNT(*), 2) as retry_rate_pct,
    AVG(if.retry_delay_seconds) FILTER (WHERE if.retry_detected) as avg_retry_delay_seconds,

    -- Latency metrics
    AVG(if.latency_seconds) as avg_latency_seconds,
    COUNT(*) FILTER (WHERE if.latency_tolerance = 'high') as high_tolerance_count,
    COUNT(*) FILTER (WHERE if.latency_tolerance = 'medium') as medium_tolerance_count,
    COUNT(*) FILTER (WHERE if.latency_tolerance = 'low') as low_tolerance_count,

    -- Time range
    MIN(if.timestamp) as first_signal,
    MAX(if.timestamp) as last_signal

FROM implicit_feedback if
GROUP BY if.model_id;

-- View combining explicit and implicit feedback
CREATE VIEW combined_feedback_analytics AS
SELECT
    r.model,

    -- Request metrics
    COUNT(DISTINCT r.id) as total_requests,
    AVG(r.cost) as avg_cost,
    AVG(r.latency) as avg_latency,

    -- Explicit feedback metrics
    COUNT(DISTINCT f.id) as explicit_feedback_count,
    AVG(f.quality_score) FILTER (WHERE f.quality_score IS NOT NULL) as avg_explicit_quality,
    COUNT(*) FILTER (WHERE f.met_expectations = TRUE) as expectations_met_count,

    -- Implicit feedback metrics
    COUNT(DISTINCT if.id) as implicit_feedback_count,
    COUNT(*) FILTER (WHERE if.error_occurred = TRUE) as implicit_error_count,
    COUNT(*) FILTER (WHERE if.retry_detected = TRUE) as implicit_retry_count,
    AVG(if.latency_seconds) as implicit_avg_latency,

    -- Combined quality score (weighted: explicit 0.7, implicit 0.3)
    CASE
        WHEN COUNT(DISTINCT f.id) > 0 OR COUNT(DISTINCT if.id) > 0 THEN
            COALESCE(
                (AVG(f.quality_score) * 0.7), 0.0
            ) + COALESCE(
                (1.0 -
                    (COUNT(*) FILTER (WHERE if.error_occurred = TRUE)::FLOAT / NULLIF(COUNT(DISTINCT if.id), 0)) * 0.3
                ), 0.0
            )
        ELSE NULL
    END as combined_quality_score

FROM responses r
LEFT JOIN feedback f ON r.id = f.response_id
LEFT JOIN implicit_feedback if ON r.query_id = if.query_id AND r.model = if.model_id
GROUP BY r.model;

-- Comments for documentation
COMMENT ON TABLE implicit_feedback IS 'Implicit behavioral signals (errors, latency, retries) for ML learning';
COMMENT ON COLUMN implicit_feedback.error_occurred IS 'Whether model produced an error or low-quality response';
COMMENT ON COLUMN implicit_feedback.error_type IS 'Classification of error (api_error, timeout, empty_response, etc.)';
COMMENT ON COLUMN implicit_feedback.latency_seconds IS 'Actual response time in seconds';
COMMENT ON COLUMN implicit_feedback.latency_accepted IS 'User waited for response (did not timeout)';
COMMENT ON COLUMN implicit_feedback.latency_tolerance IS 'Categorized user patience (high/medium/low)';
COMMENT ON COLUMN implicit_feedback.retry_detected IS 'User re-submitted semantically similar query';
COMMENT ON COLUMN implicit_feedback.retry_delay_seconds IS 'Time between original and retry query';
COMMENT ON COLUMN implicit_feedback.similarity_score IS 'Cosine similarity to previous query (0-1)';
COMMENT ON COLUMN implicit_feedback.original_query_id IS 'ID of query being retried (if retry detected)';
COMMENT ON VIEW implicit_feedback_analytics IS 'Aggregated implicit signal metrics per model';
COMMENT ON VIEW combined_feedback_analytics IS 'Combined explicit and implicit feedback analytics with weighted quality scores';
