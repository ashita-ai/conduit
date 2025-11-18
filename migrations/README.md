# Database Migrations

This directory contains Alembic database migrations for the Conduit routing system.

## Why Alembic?

- **Industry Standard**: Python's de facto migration tool
- **Version Control**: Track schema changes over time
- **Rollback Support**: Safely revert changes if needed
- **Team Collaboration**: Share schema changes via git

## Structure

```
migrations/
├── README.md                  # This file
├── env.py                     # Alembic environment config
├── script.py.mako            # Migration template
└── versions/                  # Migration files
    └── 9a6c8a59cb30_initial_schema.py
```

## Quick Start

### Apply Migrations

```bash
# Load environment variables
export $(cat .env | xargs)

# Run migrations
alembic upgrade head
```

### Check Current Version

```bash
alembic current
```

### View Migration History

```bash
alembic history
```

## Creating New Migrations

### Manual Migration

```bash
alembic revision -m "add new column to queries"
# Edit the generated file in migrations/versions/
alembic upgrade head
```

### Autogenerate (if using SQLAlchemy models)

```bash
alembic revision --autogenerate -m "add new model"
alembic upgrade head
```

## Rollback

```bash
# Rollback one migration
alembic downgrade -1

# Rollback to specific version
alembic downgrade <revision>

# Rollback all
alembic downgrade base
```

## Configuration

Database connection is configured via `DATABASE_URL` environment variable.

See `alembic.ini` and `env.py` for details.

## Tables Created

- **queries**: User queries to route
- **routing_decisions**: ML routing decisions with Thompson Sampling
- **responses**: LLM responses with metrics
- **feedback**: Explicit user feedback
- **implicit_feedback**: Behavioral signals (Phase 2+)
- **model_states**: Thompson Sampling Beta distribution parameters

## Views Created

- **recent_routing_performance**: 7-day routing performance by model

## See Also

- Alembic docs: https://alembic.sqlalchemy.org/
- Models: `conduit/core/models.py`
- Database layer: `conduit/core/database.py`
