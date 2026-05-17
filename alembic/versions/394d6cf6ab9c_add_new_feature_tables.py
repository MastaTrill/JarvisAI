"""add_new_feature_tables

Revision ID: 394d6cf6ab9c
Revises: 6944a00a722b
Create Date: 2026-05-16 19:01:37.643374

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = '394d6cf6ab9c'
down_revision: Union[str, None] = '6944a00a722b'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _table_exists(table_name: str) -> bool:
    bind = op.get_bind()
    from sqlalchemy import inspect
    return table_name in inspect(bind).get_table_names()


def upgrade() -> None:
    # RAG documents
    if not _table_exists("rag_documents"):
        op.create_table(
            "rag_documents",
            sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column("filename", sa.String(500), nullable=False),
            sa.Column("file_type", sa.String(20), nullable=False),
            sa.Column("file_size", sa.Integer(), default=0),
            sa.Column("chunk_count", sa.Integer(), default=0),
            sa.Column("status", sa.String(20), default="processing"),
            sa.Column("error_message", sa.String(1000), nullable=True),
            sa.Column("created_at", sa.DateTime(), nullable=True),
            sa.Column("created_by", sa.String(100)),
            sa.Column("tags", sa.String(1000), default=""),
        )
    if not _table_exists("rag_chunks"):
        op.create_table(
            "rag_chunks",
            sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column("document_id", sa.Integer(), nullable=False, index=True),
            sa.Column("chunk_index", sa.Integer(), nullable=False),
            sa.Column("content", sa.Text(), nullable=False),
            sa.Column("word_count", sa.Integer(), default=0),
            sa.Column("created_at", sa.DateTime(), nullable=True),
        )

    # Code sandbox
    if not _table_exists("code_executions"):
        op.create_table(
            "code_executions",
            sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column("language", sa.String(20), default="python"),
            sa.Column("code", sa.Text(), nullable=False),
            sa.Column("output", sa.Text(), default=""),
            sa.Column("error", sa.Text(), nullable=True),
            sa.Column("exit_code", sa.Integer(), nullable=True),
            sa.Column("duration_ms", sa.Integer(), default=0),
            sa.Column("status", sa.String(20), default="pending"),
            sa.Column("created_at", sa.DateTime(), nullable=True),
            sa.Column("created_by", sa.String(100)),
        )

    # A/B testing
    if not _table_exists("ab_experiments"):
        op.create_table(
            "ab_experiments",
            sa.Column("id", sa.String(36), primary_key=True),
            sa.Column("name", sa.String(200), nullable=False, index=True),
            sa.Column("description", sa.Text()),
            sa.Column("status", sa.String(20), default="draft"),
            sa.Column("variants", sa.JSON(), nullable=False),
            sa.Column("traffic_allocation", sa.Float(), default=1.0),
            sa.Column("primary_metric", sa.String(100), default="conversion"),
            sa.Column("created_at", sa.DateTime(), nullable=True),
            sa.Column("started_at", sa.DateTime(), nullable=True),
            sa.Column("ended_at", sa.DateTime(), nullable=True),
            sa.Column("created_by", sa.String(100)),
            sa.Column("winner", sa.String(100), nullable=True),
        )
    if not _table_exists("ab_events"):
        op.create_table(
            "ab_events",
            sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column("experiment_id", sa.String(36), nullable=False, index=True),
            sa.Column("user_id", sa.String(100), nullable=False, index=True),
            sa.Column("variant", sa.String(100), nullable=False),
            sa.Column("event_type", sa.String(50), nullable=False),
            sa.Column("event_name", sa.String(100), default=""),
            sa.Column("event_value", sa.Float(), nullable=True),
            sa.Column("metadata_json", sa.JSON(), nullable=True),
            sa.Column("timestamp", sa.DateTime(), nullable=True),
        )

    # Benchmarking
    if not _table_exists("benchmark_runs"):
        op.create_table(
            "benchmark_runs",
            sa.Column("id", sa.String(36), primary_key=True),
            sa.Column("name", sa.String(200), nullable=False),
            sa.Column("benchmark_type", sa.String(50)),
            sa.Column("target", sa.String(500)),
            sa.Column("iterations", sa.Integer(), default=100),
            sa.Column("avg_latency_ms", sa.Float()),
            sa.Column("min_latency_ms", sa.Float()),
            sa.Column("max_latency_ms", sa.Float()),
            sa.Column("median_latency_ms", sa.Float()),
            sa.Column("p95_latency_ms", sa.Float()),
            sa.Column("p99_latency_ms", sa.Float()),
            sa.Column("std_dev_ms", sa.Float()),
            sa.Column("requests_per_second", sa.Float()),
            sa.Column("error_rate", sa.Float(), default=0.0),
            sa.Column("system_snapshot", sa.JSON()),
            sa.Column("created_at", sa.DateTime(), nullable=True),
            sa.Column("created_by", sa.String(100)),
        )

    # Model comparison
    if not _table_exists("model_comparisons"):
        op.create_table(
            "model_comparisons",
            sa.Column("id", sa.String(36), primary_key=True),
            sa.Column("name", sa.String(200), nullable=False),
            sa.Column("model_names", sa.JSON(), nullable=False),
            sa.Column("dataset", sa.String(500)),
            sa.Column("metrics", sa.JSON()),
            sa.Column("winner", sa.String(200), nullable=True),
            sa.Column("notes", sa.String(1000), nullable=True),
            sa.Column("created_at", sa.DateTime(), nullable=True),
            sa.Column("created_by", sa.String(100)),
        )

    # Agent personality
    if not _table_exists("agent_personalities"):
        op.create_table(
            "agent_personalities",
            sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column("name", sa.String(100), nullable=False, unique=True),
            sa.Column("display_name", sa.String(200), default="Jarvis"),
            sa.Column("system_prompt", sa.Text(), default=""),
            sa.Column("tone", sa.String(50), default="professional"),
            sa.Column("expertise", sa.String(500), default=""),
            sa.Column("language", sa.String(20), default="en"),
            sa.Column("avatar_style", sa.String(50), default="default"),
            sa.Column("is_default", sa.Integer(), default=0),
            sa.Column("created_at", sa.DateTime(), nullable=True),
            sa.Column("created_by", sa.String(100)),
        )

    # Model routes
    if not _table_exists("model_routes"):
        op.create_table(
            "model_routes",
            sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column("name", sa.String(100), nullable=False),
            sa.Column("description", sa.String(500), default=""),
            sa.Column("task_type", sa.String(50), nullable=False),
            sa.Column("provider", sa.String(50), default="ollama"),
            sa.Column("model_name", sa.String(200), default=""),
            sa.Column("priority", sa.Integer(), default=0),
            sa.Column("enabled", sa.Integer(), default=1),
            sa.Column("created_at", sa.DateTime(), nullable=True),
        )


def downgrade() -> None:
    op.drop_table("model_routes")
    op.drop_table("agent_personalities")
    op.drop_table("model_comparisons")
    op.drop_table("benchmark_runs")
    op.drop_table("ab_events")
    op.drop_table("ab_experiments")
    op.drop_table("code_executions")
    op.drop_table("rag_chunks")
    op.drop_table("rag_documents")
