# Release Process

## Pre-Release Checklist

Before creating a release, verify:

- [ ] All tests pass: `uv run pytest`
- [ ] Coverage meets threshold: `uv run pytest --cov=conduit --cov-fail-under=80`
- [ ] Type checking clean: `uv run mypy conduit/`
- [ ] Linting passes: `uv run ruff check conduit/`
- [ ] Formatting correct: `uv run black conduit/ --check`
- [ ] CHANGELOG.md updated with release notes
- [ ] No TODO/FIXME comments: `grep -r "TODO\|FIXME" conduit/`
- [ ] No open critical issues blocking release

## Version Bumping

1. Update version in `pyproject.toml`:
   ```toml
   version = "X.Y.Z"
   ```

2. Version is referenced in these files (should match):
   - `conduit/__init__.py` - imports from pyproject.toml
   - `conduit/cli/main.py` - CLI version display
   - `conduit/api/app.py` - API version header

## Creating a Release

1. Ensure main branch is up to date:
   ```bash
   git checkout main
   git pull origin main
   ```

2. Run full validation:
   ```bash
   uv run pytest --cov=conduit --cov-fail-under=80
   uv run mypy conduit/
   uv run ruff check conduit/
   ```

3. Update CHANGELOG.md:
   - Move items from [Unreleased] to new version section
   - Add release date

4. Commit version bump:
   ```bash
   git add pyproject.toml CHANGELOG.md
   git commit -m "Release vX.Y.Z"
   ```

5. Create and push tag:
   ```bash
   git tag -a vX.Y.Z -m "Release vX.Y.Z"
   git push origin main --tags
   ```

6. Build package:
   ```bash
   uv build
   ```

7. Publish to PyPI:
   ```bash
   uv publish
   ```

8. Create GitHub Release:
   - Go to Releases > Draft new release
   - Select tag vX.Y.Z
   - Title: "Conduit vX.Y.Z"
   - Copy relevant CHANGELOG section as description
   - Attach wheel and sdist from `dist/`
   - Publish release

## Post-Release

- [ ] Verify package installable: `pip install conduit-router==X.Y.Z`
- [ ] Update documentation if needed
- [ ] Announce release (if significant)
- [ ] Close relevant GitHub milestone

## Versioning Policy

Conduit follows [Semantic Versioning](https://semver.org/):

- **MAJOR** (X.0.0): Breaking API changes
- **MINOR** (0.X.0): New features, backward compatible
- **PATCH** (0.0.X): Bug fixes, backward compatible

### Pre-1.0 Policy

While version is 0.x.y:
- API may change between minor versions
- Patch versions are backward compatible
- Document breaking changes in CHANGELOG

### Post-1.0 Policy

After 1.0.0:
- Breaking changes require major version bump
- Deprecated features get at least one minor version warning
- Security fixes may break compatibility if necessary
