"""Root conftest.py — excludes SDK tests from backend collection.

The SDK has its own pyproject.toml and dependency set (pytest-httpx).
Running ``pytest`` from the project root should only collect backend tests.
SDK tests should be run separately via ``cd sdk && pytest``.
"""

collect_ignore_glob = ["sdk/*"]
