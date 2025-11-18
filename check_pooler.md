# Check for Supabase Connection Pooler

## Steps:

1. Open Supabase Dashboard: https://dzlxobcplcaiijkddrvd.supabase.co
2. Navigate to: **Project Settings** (gear icon) → **Database**
3. Scroll down to find **Connection Pooling** or **Connection String** section
4. Look for these options:

### Session Mode
```
postgresql://postgres.dzlxobcplcaiijkddrvd:[YOUR-PASSWORD]@aws-0-us-west-1.pooler.supabase.com:6543/postgres
```

### Transaction Mode (preferred for serverless)
```
postgresql://postgres.dzlxobcplcaiijkddrvd:[YOUR-PASSWORD]@aws-0-us-west-1.pooler.supabase.com:6543/postgres?pgbouncer=true
```

## If You Find a Pooler Connection String:

Update `.env`:
```bash
# Replace the current DATABASE_URL with the pooler connection
DATABASE_URL=postgresql://postgres.dzlxobcplcaiijkddrvd:[YOUR-PASSWORD]@aws-0-us-west-1.pooler.supabase.com:6543/postgres
```

Then run:
```bash
./migrate.sh
```

**Alembic will work perfectly!** ✅

## If No Pooler Connection Available:

That's totally fine! We have two options:

### Option 1: Use the SQL file (5 minutes)
```bash
# Copy the SQL file content
cat migrations/001_initial_schema.sql

# Paste into Supabase SQL Editor and run
```

### Option 2: Try direct connection one more time
Sometimes the connection works from certain networks. Let's test:
```bash
export DATABASE_URL="postgresql://postgres:LMd%5Eo0rNDU6r7sgP@db.dzlxobcplcaiijkddrvd.supabase.co:5432/postgres"
.venv/bin/alembic upgrade head
```

If this works, Alembic runs immediately! If not, use Option 1.
