# artifacts/tests/conftest.py

"""
Pytest configuration file for the Employee Onboarding System.

This file provides reusable fixtures for setting up an isolated test environment,
including a temporary in-memory database and a FastAPI TestClient.

Fixtures Provided:
- `db_session`: Creates and tears down an isolated in-memory SQLite database for
                each test function, ensuring complete test isolation.
- `client`: Provides a FastAPI `TestClient` instance that is configured to use
            the isolated test database provided by the `db_session` fixture.
"""

# 1. Path Setup Section (MANDATORY)
# This section ensures that the test runner can find and import the main application module.
# -----------------------------------------------------------------------------------------
import os
import sys

# Add the 'app' directory to the Python path to enable imports from 'main'
current_dir = os.path.dirname(os.path.abspath(__file__))
app_dir = os.path.join(os.path.dirname(current_dir), 'app')
if app_dir not in sys.path:
    sys.path.insert(0, app_dir)

# 2. Standard and Third-Party Imports
# -----------------------------------
import pytest
from typing import Generator
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool

# 3. Application Imports
# These imports are now possible due to the sys.path modification above.
# We import the FastAPI app instance, the base for SQLAlchemy models, and the
# production database dependency function which we will override.
# ---------------------------------------------------------------------------
from main import app, get_db, Base


# 4. Test Database Configuration
# ------------------------------
# Use an in-memory SQLite database for fast, isolated tests.
TEST_DATABASE_URL = "sqlite:///:memory:"

# Create a new SQLAlchemy engine for the test database.
# `connect_args` is required for SQLite to allow it to be used across different threads,
# which is a scenario that can occur during testing.
test_engine = create_engine(
    TEST_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)

# Create a sessionmaker factory that will be used to create sessions for the test database.
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)


# 5. Database Fixture Section
# ---------------------------
@pytest.fixture(scope="function")
def db_session() -> Generator[Session, None, None]:
    """
    Pytest fixture to provide an isolated database session for each test function.

    This fixture handles the complete lifecycle of a test database transaction:
    1.  Creates all database tables defined in the SQLAlchemy models before a test runs.
    2.  Yields a new database session for the test to use.
    3.  Closes the session after the test is complete.
    4.  Drops all database tables after the test, ensuring no state is carried over
        to subsequent tests. This guarantees 100% test isolation.
    """
    # Create tables in the test database
    Base.metadata.create_all(bind=test_engine)
    db = TestingSessionLocal()
    try:
        # Yield the session to the test function
        yield db
    finally:
        # Ensure the session is closed after the test
        db.close()
        # Drop all tables to clean up the database for the next test
        Base.metadata.drop_all(bind=test_engine)


# 6. Client Fixture Section
# -------------------------
@pytest.fixture(scope="function")
def client(db_session: Session) -> Generator[TestClient, None, None]:
    """
    Pytest fixture to provide a FastAPI TestClient that uses the isolated test database.

    This fixture performs the following steps:
    1.  Depends on the `db_session` fixture to get an isolated database session.
    2.  Defines a dependency override for `get_db` to ensure the app uses the
        test session instead of the production database connection.
    3.  Applies this override to the FastAPI application instance.
    4.  Yields a `TestClient` configured to communicate with the in-memory app.
    5.  Crucially, it cleans up the dependency override after the test is finished,
        restoring the application to its original state.
    """
    def override_get_db() -> Generator[Session, None, None]:
        """Dependency override that yields the test database session."""
        try:
            yield db_session
        finally:
            # The session lifecycle is managed by the `db_session` fixture,
            # so we don't need to close it here.
            pass

    # Apply the dependency override to the app
    app.dependency_overrides[get_db] = override_get_db

    try:
        # Use a `with` statement for the TestClient to handle startup/shutdown events
        with TestClient(app) as test_client:
            yield test_client
    finally:
        # Clean up the dependency override to prevent test pollution
        app.dependency_overrides.clear()