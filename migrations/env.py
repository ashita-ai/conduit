import os
from logging.config import fileConfig

from sqlalchemy import engine_from_config
from sqlalchemy import pool

from alembic import context

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Override sqlalchemy.url with DIRECT_URL (migrations) or DATABASE_URL (runtime)
# DIRECT_URL uses port 5432 for migrations (Alembic requirement)
# DATABASE_URL uses port 6543 for pooled connections (runtime)
database_url = os.getenv("DIRECT_URL") or os.getenv("DATABASE_URL")
if database_url:
    # Double the % signs to escape them for ConfigParser
    database_url = database_url.replace('%', '%%')
    config.set_main_option("sqlalchemy.url", database_url)

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# add your model's MetaData object here
# for 'autogenerate' support
# from myapp import mymodel
# target_metadata = mymodel.Base.metadata
target_metadata = None

# other values from the config, defined by the needs of env.py,
# can be acquired:
# my_important_option = config.get_main_option("my_important_option")
# ... etc.


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.

    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    Uses DIRECT_URL for direct connection (migrations)
    or DATABASE_URL as fallback.
    """
    from sqlalchemy import create_engine

    # Get database URL (DIRECT_URL prioritized for migrations)
    database_url = os.getenv("DIRECT_URL") or os.getenv("DATABASE_URL")

    if not database_url:
        database_url = config.get_main_option("sqlalchemy.url")

    # Create sync engine with psycopg2 (Alembic standard)
    connectable = create_engine(database_url, poolclass=pool.NullPool)

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
