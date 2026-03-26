"""initial schema

Revision ID: 6944a00a722b
Revises:
Create Date: 2026-03-09 02:20:20.564439

Baseline migration capturing all tables managed by db_config.Base:
  - model_registry (models_registry.py)
  - jobs (jobs_persistent.py)
  - audit_trail (audit_trail.py)
  - model_versions (models_versioning.py)
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect


# revision identifiers, used by Alembic.
revision: str = "6944a00a722b"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _table_exists(table_name: str) -> bool:
    """Check if a table already exists (for baseline safety on existing DBs)."""
    bind = op.get_bind()
    insp = inspect(bind)
    return table_name in insp.get_table_names()


def upgrade() -> None:
    if not _table_exists("model_registry"):
        op.create_table(
            "model_registry",
            sa.Column("id", sa.Integer(), nullable=False),
            sa.Column("name", sa.String(), nullable=True),
            sa.Column("version", sa.String(), nullable=True),
            sa.Column("description", sa.String(), nullable=True),
            sa.Column("accuracy", sa.Float(), nullable=True),
            sa.Column("registered_at", sa.DateTime(), nullable=True),
            sa.Column("active", sa.Boolean(), nullable=True),
            sa.Column("device", sa.String(), nullable=True),
            sa.Column("external_endpoint", sa.String(), nullable=True),
            sa.Column("drift_score", sa.Float(), nullable=True),
            sa.Column("audit_log", sa.String(), nullable=True),
            sa.Column(
                "parent_id",
                sa.Integer(),
                sa.ForeignKey("model_registry.id"),
                nullable=True,
            ),
            sa.PrimaryKeyConstraint("id"),
        )
        op.create_index(
            op.f("ix_model_registry_id"), "model_registry", ["id"], unique=False
        )
        op.create_index(
            op.f("ix_model_registry_name"), "model_registry", ["name"], unique=False
        )

    if not _table_exists("jobs"):
        op.create_table(
            "jobs",
            sa.Column("id", sa.Integer(), nullable=False),
            sa.Column("job_id", sa.String(), nullable=True),
            sa.Column("status", sa.String(), nullable=True),
            sa.Column("created_at", sa.DateTime(), nullable=True),
            sa.Column("completed_at", sa.DateTime(), nullable=True),
            sa.Column("result", sa.JSON(), nullable=True),
            sa.Column("cancelled", sa.Boolean(), nullable=True),
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint("job_id"),
        )
        op.create_index(op.f("ix_jobs_id"), "jobs", ["id"], unique=False)
        op.create_index(op.f("ix_jobs_job_id"), "jobs", ["job_id"], unique=True)

    if not _table_exists("audit_trail"):
        op.create_table(
            "audit_trail",
            sa.Column("id", sa.Integer(), nullable=False),
            sa.Column("user", sa.String(), nullable=True),
            sa.Column("action", sa.String(), nullable=True),
            sa.Column("target", sa.String(), nullable=True),
            sa.Column("timestamp", sa.DateTime(), nullable=True),
            sa.Column("details", sa.String(), nullable=True),
            sa.PrimaryKeyConstraint("id"),
        )
        op.create_index(op.f("ix_audit_trail_id"), "audit_trail", ["id"], unique=False)

    if not _table_exists("model_versions"):
        op.create_table(
            "model_versions",
            sa.Column("id", sa.Integer(), nullable=False),
            sa.Column("model_name", sa.String(), nullable=True),
            sa.Column("version", sa.String(), nullable=True),
            sa.Column("path", sa.String(), nullable=True),
            sa.Column("accuracy", sa.Float(), nullable=True),
            sa.Column("created_at", sa.DateTime(), nullable=True),
            sa.Column("is_active", sa.Boolean(), nullable=True),
            sa.PrimaryKeyConstraint("id"),
        )
        op.create_index(
            op.f("ix_model_versions_id"), "model_versions", ["id"], unique=False
        )
        op.create_index(
            op.f("ix_model_versions_model_name"),
            "model_versions",
            ["model_name"],
            unique=False,
        )
        op.create_index(
            op.f("ix_model_versions_version"),
            "model_versions",
            ["version"],
            unique=False,
        )


def downgrade() -> None:
    op.drop_table("model_versions")
    op.drop_table("audit_trail")
    op.drop_table("jobs")
    op.drop_table("model_registry")
