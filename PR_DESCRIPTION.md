# Update REPL to Default to Production Environment

## Summary

Updates the interactive REPL to default to production (`app.freeplay.ai`) with SSL verification enabled, making it ready for open-source users out of the box. Adds a `--local` flag for Freeplay engineers doing local development.

## Motivation

Now that the SDK is open source, external users don't have a local Freeplay server to connect to. The REPL should default to the production environment so users can start using it immediately without configuration changes.

## Changes

### New REPL Commands

**Production Mode (Default)**
```bash
make repl
```
- Connects to `https://app.freeplay.ai`
- SSL verification enabled
- Ready for open-source users out of the box

**Local Development Mode**
```bash
make repl-local
```
- Connects to `http://localhost:8000`
- SSL verification disabled (for self-signed certificates)
- For Freeplay engineers running the app locally

### What Changed

**Before:**
- REPL always disabled SSL verification
- Assumed local development environment

**After:**
- `make repl` → Production (SSL enabled, app.freeplay.ai)
- `make repl-local` → Local development (SSL disabled, localhost:8000)

## Files Modified

- `scripts/repl_setup.py` - Added `--local` flag detection and conditional SSL/URL configuration
- `Makefile` - Added `repl-local` target, updated comments for both commands
- `CHANGELOG.md` - Documented both REPL modes

## Implementation Details

The REPL script now checks for the `--local` flag in `sys.argv`:

```python
# Default to production URL and enabled SSL verification
FREEPLAY_API_URL = os.getenv("FREEPLAY_API_URL", "https://app.freeplay.ai")
disable_ssl = False

if "--local" in sys.argv:
    disable_ssl = True
    FREEPLAY_API_URL = "http://localhost:8000"
    # Apply SSL patches for self-signed certs
    # ... SSL patching code ...
    print("\n✨ SSL warnings disabled and requests patched for local development")
else:
    print("\n✅ SSL verification enabled (default for production)")
```

Makefile provides convenient targets:
```makefile
# Production mode (default)
.PHONY: repl
repl:
	set -a; source .env 2>/dev/null || true; set +a; uv run python -i scripts/repl_setup.py

# Local development mode
.PHONY: repl-local
repl-local:
	set -a; source .env 2>/dev/null || true; set +a; uv run python -i scripts/repl_setup.py --local
```

## Testing

Verified both modes work correctly:
- ✅ `make repl` connects to production with SSL enabled
- ✅ `make repl-local` connects to localhost with SSL disabled
- ✅ Client initialization works in both modes
- ✅ Environment variables load correctly
- ✅ Appropriate console messages displayed

## Usage Examples

**For Open-Source Users:**
```bash
# Set your API key in .env
echo "FREEPLAY_API_KEY=your-key-here" > .env
echo "FREEPLAY_PROJECT_ID=your-project-id" >> .env

# Start REPL (connects to production)
make repl

# You're ready to go!
>>> client.recordings.create(...)
```

**For Freeplay Engineers:**
```bash
# Run the app locally
cd ../freeplay-app
make run

# In another terminal, start local REPL
cd ../freeplay-python
make repl-local

# Test against local backend
>>> client.recordings.create(...)
```

## Breaking Changes

None. This is a non-breaking change:
- Existing users can continue using `make repl` (now connects to production)
- Local development workflow is still supported via `make repl-local`
- All existing REPL functionality remains the same
- Environment variables work the same way
