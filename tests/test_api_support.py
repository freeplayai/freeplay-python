from unittest import TestCase

from freeplay.api_support import _retry


class TestApiSupport(TestCase):
    def test_retry_includes_read_failures(self) -> None:
        self.assertEqual(3, _retry.connect)
        self.assertEqual(3, _retry.read)
