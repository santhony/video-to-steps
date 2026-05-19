"""Pytest configuration — interprets RUN_CLOUD_TESTS to enable @pytest.mark.cloud.

By default, cloud tests are skipped (marked with -m 'not cloud' in pyproject.toml).
Set RUN_CLOUD_TESTS=1 to override and run cloud tests.
"""

import os


def pytest_collection_modifyitems(config, items):
    """Remove the default 'not cloud' filter when RUN_CLOUD_TESTS=1."""
    if os.getenv("RUN_CLOUD_TESTS") == "1":
        # Override the default marker expression to allow cloud tests
        config.option.markexpr = ""
