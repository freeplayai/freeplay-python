import logging
import os
from unittest import TestResult

original_test_run = TestResult.startTestRun


def start_test_run_with_log_level(self):  # type: ignore
    log_level = os.getenv("TEST_LOG_LEVEL", "CRITICAL")
    logging.basicConfig(level=log_level)
    original_test_run(self)


TestResult.startTestRun = start_test_run_with_log_level  # type: ignore
