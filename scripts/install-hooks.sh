#!/bin/bash
# Install git hooks for local CI checks before push
#
# Run this once after cloning the repo:
#   bash scripts/install-hooks.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
HOOKS_DIR="$REPO_ROOT/.git/hooks"

echo "📦 Installing git hooks..."

# Install pre-push hook
if [ -f "$HOOKS_DIR/pre-push" ]; then
    echo "⚠️  Pre-push hook already exists. Backing up to pre-push.backup"
    cp "$HOOKS_DIR/pre-push" "$HOOKS_DIR/pre-push.backup"
fi

cp "$SCRIPT_DIR/git-hooks/pre-push" "$HOOKS_DIR/pre-push"
chmod +x "$HOOKS_DIR/pre-push"

echo "✅ Installed pre-push hook"
echo ""
echo "The hook will run these checks before every push:"
echo "  • Ruff linting"
echo "  • Black formatting"
echo "  • Mypy type checking (non-blocking)"
echo "  • Unit tests"
echo ""
echo "To skip the hook for a single push, use:"
echo "  git push --no-verify"
echo ""
echo "Done! 🎉"
