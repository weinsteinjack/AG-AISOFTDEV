import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from fastapi.testclient import TestClient

# Import the FastAPI app and DB objects from the project
# This repo's app lives in app/main.py
from app.main import Base, get_db, app

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
