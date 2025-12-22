"""
Shared Base Class for Freeplay SDK Tests

This module provides a reusable test base class for all SDK resource tests.

Why this was created:
- Eliminates code duplication across test files
- Provides consistent test fixtures (API keys, project IDs, client instances)
- Makes it easy to add new resource test files with minimal boilerplate
- Follows unittest.TestCase patterns for consistency

What it provides:
- Common test fixtures: freeplay_api_key, api_base, project_id, client
- Shared setUp() and tearDown() methods
- Extensible design - subclasses can add resource-specific fixtures

Usage:
    class TestMyResource(FreeplayTestBase):
        def setUp(self):
            super().setUp()
            # Add resource-specific fixtures here

        def test_my_feature(self):
            # Use self.client, self.project_id, etc.
            pass
"""

from unittest import TestCase
from uuid import uuid4

from freeplay import Freeplay


class FreeplayTestBase(TestCase):
    """
    Base class for Freeplay SDK tests.

    Provides common test fixtures:
    - freeplay_api_key: Test API key
    - api_base: Test API base URL
    - project_id: Test project UUID
    - client: Configured Freeplay client instance

    Subclasses can extend setUp() to add resource-specific fixtures.
    """

    def setUp(self) -> None:
        """Set up common test fixtures."""
        super().setUp()
        self.maxDiff = None

        # Common test configuration
        self.freeplay_api_key = "test_freeplay_api_key"
        self.api_base = "http://localhost:9091/api"

        # Common test IDs
        self.project_id = str(uuid4())

        # Create client instance
        self.client = Freeplay(
            freeplay_api_key=self.freeplay_api_key, api_base=self.api_base
        )

    def tearDown(self) -> None:
        """Clean up after tests."""
        super().tearDown()
