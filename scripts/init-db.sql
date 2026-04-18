-- JarvisAI Database Initialization Script
-- Run this when setting up the database for the first time

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";

-- Create indexes for better performance
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_users_username ON users(username);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_users_email ON users(email);

-- Create audit table if it doesn't exist
CREATE TABLE IF NOT EXISTS audit_log (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    user_id VARCHAR(255),
    action VARCHAR(255) NOT NULL,
    resource VARCHAR(255),
    details TEXT,
    ip_address INET,
    user_agent TEXT
);

-- Create indexes for audit log
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_audit_log_timestamp ON audit_log(timestamp);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_audit_log_user_id ON audit_log(user_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_audit_log_action ON audit_log(action);

-- Create performance monitoring table
CREATE TABLE IF NOT EXISTS system_metrics (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    cpu_usage DECIMAL(5,2),
    memory_usage DECIMAL(5,2),
    disk_usage DECIMAL(5,2),
    active_connections INTEGER,
    response_time_avg DECIMAL(10,3),
    error_rate DECIMAL(5,2)
);

-- Create indexes for metrics
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_system_metrics_timestamp ON system_metrics(timestamp);

-- Create API usage tracking table
CREATE TABLE IF NOT EXISTS api_usage (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    endpoint VARCHAR(500),
    method VARCHAR(10),
    status_code INTEGER,
    response_time DECIMAL(10,3),
    user_id VARCHAR(255),
    ip_address INET
);

-- Create indexes for API usage
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_api_usage_timestamp ON api_usage(timestamp);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_api_usage_endpoint ON api_usage(endpoint);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_api_usage_user_id ON api_usage(user_id);

-- Insert initial data
INSERT INTO audit_log (user_id, action, resource, details)
VALUES ('system', 'database_init', 'database', 'Initial database setup completed')
ON CONFLICT DO NOTHING;