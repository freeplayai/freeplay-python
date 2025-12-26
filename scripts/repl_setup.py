#!/usr/bin/env python
# ruff: noqa: E402
"""
Interactive REPL setup script for Freeplay development.
This script sets up SSL bypass patches and initializes the Freeplay client
with environment variables.
"""

import os

# CORRECT monkey patch that avoids recursion
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Patch at the adapter level instead
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

# Import Freeplay after patches
from freeplay import Freeplay

# Load environment variables
freeplay_api_key = os.environ.get("FREEPLAY_API_KEY")
api_base = os.environ.get("FREEPLAY_API_URL")
project_id = os.environ.get("FREEPLAY_PROJECT_ID")
session_id = os.environ.get("FREEPLAY_SESSION_ID")
dataset_id = os.environ.get("FREEPLAY_DATASET_ID")

# Initialize client
if not freeplay_api_key:
    print("⚠️  Warning: FREEPLAY_API_KEY not set in .env")
    client = None
else:
    client = Freeplay(
        freeplay_api_key=freeplay_api_key,
        api_base=f"{api_base}/api" if api_base else None,
    )
    print("✅ Freeplay client initialized as 'client'")

# Print available variables
print("\n" + "=" * 60)
print("🎮 Freeplay Interactive REPL")
print("=" * 60)
print("\nAvailable variables:")
print("  • client       : Freeplay client instance")
print(f"  • project_id   : {project_id if project_id else '(not set)'}")
print(f"  • session_id   : {session_id if session_id else '(not set)'}")
print(f"  • dataset_id   : {dataset_id if dataset_id else '(not set)'}")
print(f"  • api_base     : {api_base if api_base else '(not set)'}")
print("\n✨ SSL warnings disabled and requests patched for local development")
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
    "requests": requests,
}

# Start interactive console
code.interact(local=local_vars, banner="")
