"""
Alembic environment configuration for migrations.
Refactored for clarity and best practices.
"""


import os
import sys
from logging.config import fileConfig
from sqlalchemy import engine_from_config, pool
from alembic.config import Config
from alembic.context import (
    configure as alembic_configure,
    begin_transaction as alembic_begin_transaction,
    run_migrations as alembic_run_migrations,
    is_offline_mode as alembic_is_offline_mode,
)

# Ensure project root is in sys.path before importing db_config
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from db_config import Base
import logging



# Alembic Config object setup
config = Config(os.path.join(os.path.dirname(__file__), '..', 'alembic.ini'))
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
        logger.error("sqlalchemy.url is not set in Alembic config.")
        raise RuntimeError("sqlalchemy.url is not set in Alembic config.")
    logger.info("Running Alembic migrations in offline mode.")
    try:
        pre_migration_hook()
        alembic_configure(
            url=url,
            target_metadata=target_metadata,
            literal_binds=True,
        )
        with alembic_begin_transaction():
            alembic_run_migrations()
        post_migration_hook()
        logger.info("Alembic offline migrations completed successfully.")
    except Exception as e:
        logger.exception(f"Error during offline migrations: {e}")
        raise

def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""
    section = config.get_section(config.config_ini_section)
    if section is None:
        logger.error("Alembic config section is missing. Please ensure your alembic.ini file is present and properly configured.")
        raise RuntimeError(
            "Alembic config section is missing. "
            "Please ensure your alembic.ini file is present and properly configured."
        )
    connectable = engine_from_config(
        section,
        prefix='sqlalchemy.',
        poolclass=pool.NullPool,
    )
    try:
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
            logger.info("Alembic online migrations completed successfully.")
    except Exception as e:
        logger.exception(f"Error during online migrations: {e}")
        raise

def pre_migration_hook() -> None:
    """Custom logic to run before migrations (optional)."""
    logger.info("Pre-migration hook executed.")
    # Add custom pre-migration logic here (e.g., health checks, backups)

def post_migration_hook() -> None:
    """Custom logic to run after migrations (optional)."""
    logger.info("Post-migration hook executed.")
    # Add custom post-migration logic here (e.g., notifications, cleanup)


def check_migration_consistency() -> None:
    """Check if models and migrations are in sync (for CI/CD). Raises if out of sync."""
    from alembic.autogenerate import compare_metadata
    from alembic.script import ScriptDirectory
    from alembic.runtime.environment import EnvironmentContext
    script = ScriptDirectory.from_config(config)
    def process_revision_directives(context, revision, directives):
        if getattr(config.cmd_opts, 'autogenerate', False):
            diffs = compare_metadata(context, target_metadata)
            if diffs:
                logger.error("Model and migration are out of sync! Run 'alembic revision --autogenerate'.")
                raise RuntimeError("Model and migration are out of sync! Run 'alembic revision --autogenerate'.")
    try:
        with EnvironmentContext(
            config,
            script,
            fn=run_migrations_online,
            as_sql=False,
            process_revision_directives=process_revision_directives,
        ):
            logger.info("Migration consistency check completed successfully.")
    except Exception as e:
        logger.exception(f"Migration consistency check failed: {e}")
        raise

# --- Migration utility functions ---
def alembic_stamp(revision: str = "head") -> None:
    """Stamp the database with the given revision (default: head)."""
    from alembic.command import stamp
    try:
        stamp(config, revision)
        logger.info(f"Database stamped with revision: {revision}")
    except Exception as e:
        logger.exception(f"Error during alembic stamp: {e}")
        raise

def alembic_upgrade(revision: str = "head") -> None:
    """Upgrade the database to the given revision (default: head)."""
    from alembic.command import upgrade
    try:
        upgrade(config, revision)
        logger.info(f"Database upgraded to revision: {revision}")
    except Exception as e:
        logger.exception(f"Error during alembic upgrade: {e}")
        raise

def alembic_downgrade(revision: str) -> None:
    """Downgrade the database to the given revision."""
    from alembic.command import downgrade
    try:
        downgrade(config, revision)
        logger.info(f"Database downgraded to revision: {revision}")
    except Exception as e:
        logger.exception(f"Error during alembic downgrade: {e}")
        raise

def main() -> None:
    """CLI entry point for Alembic migration utilities."""
    import sys
    args = sys.argv[1:]
    if not args:
        # Default: run migrations
        if alembic_is_offline_mode():
            run_migrations_offline()
        else:
            run_migrations_online()
        return
    cmd = args[0].lower()
    if cmd == "check":
        check_migration_consistency()
    elif cmd == "stamp":
        revision = args[1] if len(args) > 1 else "head"
        alembic_stamp(revision)
    elif cmd == "upgrade":
        revision = args[1] if len(args) > 1 else "head"
        alembic_upgrade(revision)
    elif cmd == "downgrade":
        if len(args) < 2:
            logger.error("Please specify a revision for downgrade.")
            print("Usage: python alembic/env.py downgrade <revision>")
            sys.exit(1)
        alembic_downgrade(args[1])
    else:
        logger.error(f"Unknown command: {cmd}")
        print("Usage: python alembic/env.py [check|stamp|upgrade|downgrade]")
        sys.exit(1)

if __name__ == "__main__":
    main()
else:
    if alembic_is_offline_mode():
        run_migrations_offline()
    else:
        run_migrations_online()