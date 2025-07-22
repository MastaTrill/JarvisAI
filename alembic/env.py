"""
Alembic environment configuration for migrations.
Refactored for clarity and best practices.
"""


import os
import sys
from logging.config import fileConfig
from sqlalchemy import engine_from_config, pool
from alembic.context import (
    config as alembic_config,
    configure as alembic_configure,
    begin_transaction as alembic_begin_transaction,
    run_migrations as alembic_run_migrations,
    is_offline_mode as alembic_is_offline_mode,
)

# Ensure project root is in sys.path before importing db_config
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from db_config import Base
import logging


# Support for environment variable override of DB URL
config = alembic_config
db_url_env = os.getenv("ALEMBIC_DATABASE_URL")
if db_url_env:
    config.set_main_option("sqlalchemy.url", db_url_env)

# Set up logger
logger = logging.getLogger("alembic.env")

# Interpret the config file for Python logging.
if config.config_file_name:
    fileConfig(config.config_file_name)
else:
    raise RuntimeError("Alembic config_file_name is not set.")

# Set target metadata for 'autogenerate' support
target_metadata = Base.metadata

def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode."""
    url = config.get_main_option("sqlalchemy.url")
    if not url:
        raise RuntimeError("sqlalchemy.url is not set in Alembic config.")
    logger.info("Running Alembic migrations in offline mode.")
    pre_migration_hook()
    alembic_configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
    )
    with alembic_begin_transaction():
        alembic_run_migrations()
    post_migration_hook()

def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""
    section = config.get_section(config.config_ini_section)
    if section is None:
        raise RuntimeError(
            "Alembic config section is missing. "
            "Please ensure your alembic.ini file is present and properly configured."
        )
    connectable = engine_from_config(
        section,
        prefix='sqlalchemy.',
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        logger.info("Running Alembic migrations in online mode.")
        pre_migration_hook()
        alembic_configure(
            connection=connection,
            target_metadata=target_metadata,
        )
        with alembic_begin_transaction():
            alembic_run_migrations()
        post_migration_hook()

def pre_migration_hook() -> None:
    """Custom logic to run before migrations (optional)."""
    logger.info("Pre-migration hook executed.")

def post_migration_hook() -> None:
    """Custom logic to run after migrations (optional)."""
    logger.info("Post-migration hook executed.")


# Migration consistency check (CI/CD):
def check_migration_consistency() -> None:
    """Check if models and migrations are in sync (for CI/CD)."""
    from alembic.autogenerate import compare_metadata
    from alembic.script import ScriptDirectory
    from alembic.runtime.environment import EnvironmentContext
    script = ScriptDirectory.from_config(config)
    def process_revision_directives(context, revision, directives):
        if getattr(config.cmd_opts, 'autogenerate', False):
            diffs = compare_metadata(context, target_metadata)
            if diffs:
                raise RuntimeError("Model and migration are out of sync! Run 'alembic revision --autogenerate'.")
    with EnvironmentContext(
        config,
        script,
        fn=run_migrations_online,
        as_sql=False,
        process_revision_directives=process_revision_directives,
    ):
        pass

if __name__ == "__main__":
    # For CI/CD: python alembic/env.py check
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "check":
        check_migration_consistency()
    else:
        if alembic_is_offline_mode():
            run_migrations_offline()
        else:
            run_migrations_online()
else:
    if alembic_is_offline_mode():
        run_migrations_offline()
    else:
        run_migrations_online()