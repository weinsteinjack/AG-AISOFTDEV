import os
import importlib.util
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from fastapi.testclient import TestClient

# Import the FastAPI app and DB objects from the project without relying on package layout
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
_APP_MAIN_PATH = os.path.join(_ROOT, "app", "main.py")
_SPEC = importlib.util.spec_from_file_location("app_main_module", _APP_MAIN_PATH)
_MOD = importlib.util.module_from_spec(_SPEC)
assert _SPEC and _SPEC.loader
_SPEC.loader.exec_module(_MOD)  # type: ignore[attr-defined]

Base = _MOD.Base
get_db = _MOD.get_db
app = _MOD.app

# Create a new SQLAlchemy engine for an in-memory SQLite database
SQLALCHEMY_DATABASE_URL = "sqlite://"

# Fixture to create and destroy the database schema
@pytest.fixture(scope="function")
def db_engine():
    engine = create_engine(
        SQLALCHEMY_DATABASE_URL,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool
    )
    Base.metadata.create_all(bind=engine)
    yield engine
    Base.metadata.drop_all(bind=engine)

# Fixture to create a new database session for each test
@pytest.fixture(scope="function")
def db_session(db_engine):
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=db_engine)
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Fixture to override the get_db dependency
@pytest.fixture(scope="function")
def client(db_session):
    def override_get_db():
        try:
            yield db_session
        finally:
            db_session.rollback()
    
    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as client:
        yield client
    # Clean up override after each test function
    app.dependency_overrides.pop(get_db, None)
