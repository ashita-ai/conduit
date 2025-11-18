#!/bin/bash
# migrate.sh - Run Alembic migrations for Conduit

set -e  # Exit on error

# Colors for output
GREEN='\033[0.32m'
RED='\033[0;31m'
NC='\033[0m'  # No Color

echo "🗄️  Conduit Database Migration"
echo "================================"

# Check if DATABASE_URL is set
if [ -z "$DATABASE_URL" ]; then
    echo -e "${RED}❌ DATABASE_URL environment variable not set${NC}"
    echo "Please set DATABASE_URL in .env or export it:"
    echo "  export DATABASE_URL='postgresql://...'"
    exit 1
fi

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    echo "📦 Activating virtual environment..."
    source .venv/bin/activate
fi

# Check current migration status
echo ""
echo "📋 Current migration status:"
alembic current || true

# Show pending migrations
echo ""
echo "⏳ Pending migrations:"
alembic history --verbose | head -20

# Run migrations
echo ""
echo "🚀 Running migrations..."
alembic upgrade head

# Show final status
echo ""
echo -e "${GREEN}✅ Migrations complete!${NC}"
echo ""
echo "📊 Current schema version:"
alembic current

echo ""
echo "================================"
echo "Migration completed successfully!"
