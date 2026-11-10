"""
Shared test setup: make the whole test run use a temporary SQLite database.

Why this file exists: app/main.py calls load_dotenv(), which would read the real
(Neon) DATABASE_URL from .env, and the app's startup runs create_all() on that
database. pytest loads conftest.py BEFORE the test files import the app, so
setting DATABASE_URL here wins: load_dotenv() never overwrites a variable that
is already set.
"""
import os
import shutil
import tempfile

import pytest

# 1. Point the app at a brand-new SQLite file in a temporary folder.
#    This must happen at import time (not inside a fixture), because test files
#    import the app while pytest is still collecting tests.
TEST_DB_DIR = tempfile.mkdtemp(prefix="pcb_test_db_")
os.environ["DATABASE_URL"] = "sqlite:///" + os.path.join(TEST_DB_DIR, "test.db").replace("\\", "/")


@pytest.fixture(scope="session", autouse=True)
def check_test_database():
    """Safety net: stop immediately if the app is somehow pointed at a real database."""
    from app.db.database import engine

    assert engine.url.get_backend_name() == "sqlite", "Tests must never run against a real database"


def pytest_sessionfinish(session, exitstatus):
    """Delete the temporary database. pytest calls this at the very end of every run,
    even if tests fail or only collection ran (fixtures don't run in that case)."""
    import sys

    database = sys.modules.get("app.db.database")
    if database is not None:
        database.engine.dispose()  # close open connections first, or Windows can't delete the file
    shutil.rmtree(TEST_DB_DIR, ignore_errors=True)
