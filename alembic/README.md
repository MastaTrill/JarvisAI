# Alembic Migrations - Project README

## How to Create a New Migration

1. Make changes to your SQLAlchemy models in the codebase.
2. Run:
   ```sh
   alembic revision --autogenerate -m "describe your change"
   ```
3. Review the generated migration script in `alembic/versions/` and edit if needed.

## How to Apply Migrations

To upgrade your database to the latest schema:

```sh
alembic upgrade head
```text


## How to Check Model/Migration Sync (CI/CD)

To ensure your models and migrations are in sync, run:

```sh
python alembic/env.py check
```

This will fail if there are differences between your models and the migration history.

## Supported Python Versions for CI/CD

Your workflows and CI/CD should use only supported Python versions:

```yaml
python-version: [3.9, 3.10, 3.11]
```

Do not use Python 3.1 or any unsupported version, as they are not available on Ubuntu 24.04 runners.


### Troubleshooting

If you see an error like:

```
The version '3.1' with architecture 'x64' was not found for Ubuntu 24.04.
```

Check all workflow files for any reference to `python-version: 3.1` and change it to a supported version as shown above.

## How to Use a Custom Database URL

Set the environment variable `ALEMBIC_DATABASE_URL` before running Alembic commands:

```sh
export ALEMBIC_DATABASE_URL=postgresql://user:pass@host/dbname
alembic upgrade head
```


## Pre/Post Migration Hooks

You can add custom logic to the `pre_migration_hook` and `post_migration_hook` functions in `alembic/env.py`.

## Best Practices

- Use descriptive messages for migration scripts.
- Test migrations on a fresh database occasionally.
- Regularly review and prune old migrations if the project grows large.

## Alembic Migrations for Jarvis AI

- Use `alembic revision --autogenerate -m "message"` to create migrations.
- Use `alembic upgrade head` to apply migrations.
- Edit `db_config.py` and models to define your database schema.
