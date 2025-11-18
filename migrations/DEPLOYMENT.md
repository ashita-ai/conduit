# Database Migration Deployment Guide

## Context

We're using **Alembic** for database migrations (Python industry standard), but Supabase free tier blocks direct PostgreSQL connections. Here are your deployment options:

## ✅ Option 1: Supabase SQL Editor (Recommended for Free Tier)

**Steps:**
1. Open Supabase dashboard: https://dzlxobcplcaiijkddrvd.supabase.co
2. Navigate to SQL Editor
3. Copy the migration SQL from `migrations/001_initial_schema.sql`
4. Paste and run in SQL Editor

**Pros:**
- Works on free tier
- Visual confirmation in dashboard
- No connection issues

**Cons:**
- Manual step (not automated)
- Alembic won't track version internally

---

## ✅ Option 2: Alembic via Pooler Connection (When Available)

**Prerequisites:**
- Supabase connection pooler enabled
- Pooler connection string from dashboard

**Steps:**
1. Get pooler connection string from Supabase:
   ```
   Project Settings → Database → Connection Pooling
   ```
2. Update `.env` with pooler URL:
   ```
   DATABASE_URL=postgresql://postgres.dzlxobcplcaiijkddrvd:[password]@aws-0-us-west-1.pooler.supabase.com:6543/postgres
   ```
3. Run migrations:
   ```bash
   ./migrate.sh
   # OR
   export $(cat .env | xargs) && alembic upgrade head
   ```

**Pros:**
- Automated
- Alembic version tracking
- Rollback support

**Cons:**
- Requires pooler access (may not be on free tier)

---

## ✅ Option 3: Direct PostgreSQL (Paid/Production)

**Prerequisites:**
- Paid Supabase tier OR local PostgreSQL

**Steps:**
1. Ensure `DATABASE_URL` in `.env` points to accessible PostgreSQL
2. Run migrations:
   ```bash
   ./migrate.sh
   ```

**Pros:**
- Full Alembic features
- Automated deployments
- CI/CD integration

---

## Migration Files

### Alembic Format (Version Controlled)
- `migrations/versions/9a6c8a59cb30_initial_schema.py` - Python migration
- `migrations/env.py` - Environment configuration
- `alembic.ini` - Alembic configuration

### SQL Format (Manual Execution)
- `migrations/001_initial_schema.sql` - Generated SQL for manual execution

---

## Verification

After applying migrations, verify tables exist:

```sql
-- Run in Supabase SQL Editor
SELECT tablename
FROM pg_tables
WHERE schemaname = 'public'
ORDER BY tablename;

-- Expected tables:
-- - queries
-- - routing_decisions
-- - responses
-- - feedback
-- - implicit_feedback
-- - model_states
```

---

## Future Migrations

### Creating New Migrations

```bash
# Create new migration
alembic revision -m "add new column"

# Edit generated file in migrations/versions/

# Apply migration (when pooler available)
alembic upgrade head

# OR generate SQL for manual execution
alembic upgrade head --sql > migrations/002_new_migration.sql
```

### Rollback (Alembic only)

```bash
# Rollback last migration
alembic downgrade -1

# Rollback to specific version
alembic downgrade <revision>

# Rollback all
alembic downgrade base
```

---

## CI/CD Integration (Future)

When you have pooler access or paid tier:

```yaml
# .github/workflows/deploy.yml
- name: Run Migrations
  env:
    DATABASE_URL: ${{ secrets.DATABASE_URL }}
  run: |
    alembic upgrade head
```

---

## Why Both Formats?

1. **Alembic** (`.py` files):
   - Industry standard
   - Version control
   - Team collaboration
   - Rollback support
   - Future-proof for when pooler is available

2. **Raw SQL** (`.sql` files):
   - Works on free tier
   - Manual execution fallback
   - Can be run in Supabase SQL Editor

Both are kept in version control. Choose deployment method based on your access level.
