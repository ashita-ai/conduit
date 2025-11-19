# LLM Pricing Updates

This document describes how to update LLM pricing data in Conduit.

## Overview

Conduit uses pricing data from the community-maintained [llm-prices.com](https://www.llm-prices.com/) project to calculate accurate LLM costs. This data is stored in the `model_prices` table and can be updated as providers change their pricing.

## Quick Update

To sync the latest pricing from llm-prices.com:

```bash
python scripts/sync_pricing.py
```

This will:
- Fetch current pricing for ~90 models from llm-prices.com API
- Update existing model prices or insert new models
- Preserve source attribution and snapshot timestamps

## Pricing Sync Script

### Usage

```bash
# Sync pricing (recommended: run weekly or when pricing changes)
python scripts/sync_pricing.py

# Dry run (preview changes without updating database)
python scripts/sync_pricing.py --dry-run

# Verbose logging
python scripts/sync_pricing.py --verbose
```

### What it Does

1. **Fetches Data**: Retrieves `current-v1.json` from llm-prices.com
2. **Deduplicates**: Handles any duplicate model IDs in source data
3. **Upserts**: Updates existing prices or inserts new models
4. **Tracks Snapshots**: Records the snapshot date for data lineage

### Supported Models

The sync script pulls pricing for all models in llm-prices.com, including:

- **OpenAI**: GPT-4o, GPT-4o-mini, O1, O3, O4 variants
- **Anthropic**: Claude 3, Claude 3.5, Claude 4.5 variants
- **Google**: Gemini Pro, Flash, and experimental models
- **xAI**: Grok variants
- **Amazon**: Nova models
- **Mistral**: Mistral, Ministral, Codestral variants
- **Others**: DeepSeek, Kimi, OpenChat, etc.

## Data Schema

The `model_prices` table stores:

```sql
CREATE TABLE model_prices (
    model_id TEXT PRIMARY KEY,
    input_cost_per_million NUMERIC(12,8),
    output_cost_per_million NUMERIC(12,8),
    cached_input_cost_per_million NUMERIC(12,8),
    source TEXT,
    snapshot_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ
);
```

- **model_id**: Provider-specific model identifier (e.g., `gpt-4o-mini`)
- **Costs**: Per-million-token pricing in USD
- **cached_input_cost**: Cached/prompt caching pricing (if supported)
- **source**: Data source (`llm-prices.com`)
- **snapshot_at**: When this pricing was valid
- **created_at**: When record was first inserted

## Manual Updates

If you need to add custom pricing for a model not in llm-prices.com:

```sql
INSERT INTO model_prices (
    model_id,
    input_cost_per_million,
    output_cost_per_million,
    cached_input_cost_per_million,
    source,
    snapshot_at
) VALUES (
    'custom-model-id',
    1.50,  -- $1.50 per 1M input tokens
    6.00,  -- $6.00 per 1M output tokens
    0.75,  -- $0.75 per 1M cached input tokens
    'manual',
    NOW()
) ON CONFLICT (model_id) DO UPDATE SET
    input_cost_per_million = EXCLUDED.input_cost_per_million,
    output_cost_per_million = EXCLUDED.output_cost_per_million,
    cached_input_cost_per_million = EXCLUDED.cached_input_cost_per_million,
    snapshot_at = EXCLUDED.snapshot_at;
```

## Automation

### Recommended: Weekly Cron Job

```bash
# Add to crontab (runs every Sunday at 2 AM)
0 2 * * 0 cd /path/to/conduit && /path/to/.venv/bin/python scripts/sync_pricing.py
```

### GitHub Actions (Future)

```yaml
name: Update LLM Pricing
on:
  schedule:
    - cron: '0 2 * * 0'  # Weekly on Sunday
  workflow_dispatch:  # Manual trigger

jobs:
  sync-pricing:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
      - run: pip install -r requirements.txt
      - run: python scripts/sync_pricing.py
        env:
          SUPABASE_URL: ${{ secrets.SUPABASE_URL }}
          SUPABASE_ANON_KEY: ${{ secrets.SUPABASE_ANON_KEY }}
```

## Model ID Conventions

Model IDs must match exactly between config.py and the pricing database:
- Use the exact ID from llm-prices.com (e.g., `claude-3.5-sonnet`, not `claude-sonnet-4`)
- Verify new models exist: `SELECT model_id FROM model_prices WHERE model_id = 'your-model';`
- Update config.py default_models to match available pricing

## Data Source

### llm-prices.com API

- **Endpoint**: `https://www.llm-prices.com/current-v1.json`
- **Format**: JSON with `updated_at` timestamp and `prices` array
- **Maintainer**: Community-maintained (simonw/llm-prices)
- **License**: Open source
- **Update Frequency**: Community contributions (variable)

### Data Format

```json
{
  "updated_at": "2025-11-18",
  "prices": [
    {
      "id": "gpt-4o-mini",
      "vendor": "openai",
      "name": "GPT-4o mini",
      "input": 0.150,
      "output": 0.600,
      "input_cached": 0.075
    }
  ]
}
```

## Verification

After syncing, verify the data:

```python
from conduit.core.database import Database

db = Database()
await db.connect()

# Check total count
pricing = await db.get_model_prices()
print(f"Total models: {len(pricing)}")

# Check specific model
model_price = pricing.get("gpt-4o-mini")
if model_price:
    print(f"GPT-4o mini input: ${model_price.input_cost_per_token:.10f}/token")
    print(f"GPT-4o mini output: ${model_price.output_cost_per_token:.10f}/token")
```

## Troubleshooting

### Error: Duplicate model IDs

The sync script automatically deduplicates based on model_id, keeping the last occurrence.

### Error: Connection failed

Ensure `SUPABASE_URL` and `SUPABASE_ANON_KEY` are set:

```bash
export SUPABASE_URL="https://your-project.supabase.co"
export SUPABASE_ANON_KEY="your-anon-key"
```

### Error: API rate limit

The llm-prices.com API is static JSON served via CDN and should not rate limit. If issues occur, add retries:

```python
# The script will be updated to handle transient failures
```

## Cost Calculation

See `conduit/core/pricing.py` for how per-million costs are converted to per-token:

```python
@property
def input_cost_per_token(self) -> float:
    """Cost per single input token in dollars."""
    return self.input_cost_per_million / 1_000_000.0
```

## Future Improvements

- [ ] Add retry logic for transient API failures
- [ ] Support historical pricing snapshots
- [ ] Add pricing change notifications
- [ ] Create pricing analytics dashboard
- [ ] Support custom pricing overrides per customer
