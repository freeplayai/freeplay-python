#!/usr/bin/env python
# ruff: noqa: E402
"""
Interactive REPL setup script for Freeplay development.

By default, connects to production (app.freeplay.ai).
Use --local flag to connect to localhost with SSL bypass.

Usage:
    python -i scripts/repl_setup.py           # Production (default)
    python -i scripts/repl_setup.py --local   # Local development
"""

import os
import sys

# Check if running in local mode
is_local = "--local" in sys.argv

if is_local:
    print("🔧 Local mode: Disabling SSL verification for localhost...")

    # SSL bypass patches for local development only
    import urllib3

    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    import requests
    from requests.adapters import HTTPAdapter
    import ssl

    class NoSSLAdapter(HTTPAdapter):
        def init_poolmanager(self, *args, **kwargs):
            kwargs["ssl_context"] = ssl._create_unverified_context()
            return super().init_poolmanager(*args, **kwargs)

    # Create a session with the adapter
    session = requests.Session()
    session.mount("https://", NoSSLAdapter())

    # Monkey patch requests to use this session
    original_patch = requests.patch

    def patch_no_ssl(*args, **kwargs):
        kwargs["verify"] = False
        return original_patch(*args, **kwargs)

    requests.patch = patch_no_ssl

    # Additional monkey patch for all request methods
    original_request = requests.Session.request

    def no_ssl_verify(self, method, url, **kwargs):
        kwargs["verify"] = False
        return original_request(self, method, url, **kwargs)

    requests.Session.request = no_ssl_verify

# Import Freeplay after patches (if local mode)
from freeplay import Freeplay

# Load environment variables
freeplay_api_key = os.environ.get("FREEPLAY_API_KEY")
project_id = os.environ.get("FREEPLAY_PROJECT_ID")
session_id = os.environ.get("FREEPLAY_SESSION_ID")
dataset_id = os.environ.get("FREEPLAY_DATASET_ID")

# Set API base URL
if is_local:
    # Local development - default to localhost
    api_base = os.environ.get("FREEPLAY_API_URL", "http://localhost:8000")
else:
    # Production - default to app.freeplay.ai
    api_base = os.environ.get("FREEPLAY_API_URL", "https://app.freeplay.ai")

# Initialize client
if not freeplay_api_key:
    print("⚠️  Warning: FREEPLAY_API_KEY not set in .env")
    client = None
else:
    client = Freeplay(
        freeplay_api_key=freeplay_api_key,
        api_base=f"{api_base}/api",
    )
    print("✅ Freeplay client initialized as 'client'")

# Print available variables
print("\n" + "=" * 60)
print("🎮 Freeplay Interactive REPL")
print("=" * 60)
print("\nMode:", "🔧 Local Development" if is_local else "🌐 Production")
print("\nAvailable variables:")
print("  • client       : Freeplay client instance")
print(f"  • project_id   : {project_id if project_id else '(not set)'}")
print(f"  • session_id   : {session_id if session_id else '(not set)'}")
print(f"  • dataset_id   : {dataset_id if dataset_id else '(not set)'}")
print(f"  • api_base     : {api_base}")

if is_local:
    print("\n⚠️  SSL verification disabled for local development")
else:
    print("\n🔒 SSL verification enabled (production mode)")
    print("    Use --local flag to connect to localhost")

print("\nExample commands:")
print("  client.metadata.update_session(")
print("      project_id=project_id,")
print("      session_id=session_id,")
print("      metadata={'test_key': 'Hello from Python!'}")
print("  )")
print("\n" + "=" * 60 + "\n")

# Import code module for interactive console
import code

# Prepare the local namespace with all our variables
local_vars = {
    "client": client,
    "project_id": project_id,
    "session_id": session_id,
    "dataset_id": dataset_id,
    "api_base": api_base,
    "Freeplay": Freeplay,
    "os": os,
}

# Start interactive console
code.interact(local=local_vars, banner="")
