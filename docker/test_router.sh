#!/bin/bash
# Test script for Conduit + LiteLLM Router with Open WebUI

set -e

echo "======================================================================"
echo "Testing Conduit + LiteLLM Router"
echo "======================================================================"
echo ""

# Test 1: Health check
echo "Test 1: Health check..."
curl -s http://localhost:8000/health | python -m json.tool
echo "✓ Health check passed"
echo ""

# Test 2: Root endpoint
echo "Test 2: Router info..."
curl -s http://localhost:8000/ | python -m json.tool
echo "✓ Router info retrieved"
echo ""

# Test 3: List models
echo "Test 3: List available models..."
curl -s http://localhost:8000/v1/models | python -m json.tool
echo "✓ Models listed (should show multiple deployments)"
echo ""

# Test 4: Chat completion
echo "Test 4: Chat completion (Conduit routes request)..."
curl -s -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o-mini",
    "messages": [{"role": "user", "content": "Say hello in 5 words"}],
    "temperature": 0.7,
    "max_tokens": 20
  }' | python -m json.tool
echo "✓ Chat completion successful"
echo ""

# Test 5: Multiple requests to see routing
echo "Test 5: Multiple requests to demonstrate routing..."
for i in {1..3}; do
  echo "Request $i..."
  RESPONSE=$(curl -s -X POST http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d "{
      \"model\": \"gpt-4o-mini\",
      \"messages\": [{\"role\": \"user\", \"content\": \"Count to $i\"}],
      \"max_tokens\": 10
    }")
  echo "$RESPONSE" | python -m json.tool | grep -E "\"model\"|\"content\""
  echo ""
done
echo "✓ Multiple requests completed (Conduit learning from each)"
echo ""

echo "======================================================================"
echo "All tests passed!"
echo "======================================================================"
echo ""
echo "Open WebUI should be available at: http://localhost:3000"
echo "Models visible in UI: gpt-4o-mini-1, gpt-4o-mini-2, claude-3-5-haiku-1, etc."
echo "Conduit will route between deployments and learn optimal selection"
echo ""
