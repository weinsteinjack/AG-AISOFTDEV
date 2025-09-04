import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.pool import StaticPool
from fastapi.testclient import TestClient

from app.main import app, get_db
from app.db_models import Base, User, OnboardingTask  # Import Base and models to ensure they are registered

@pytest.fixture(scope='function')
def test_engine():
    # Create a test engine with StaticPool to ensure the same in-memory DB is shared
    engine = create_engine(
        'sqlite:///:memory:', 
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,  # Use StaticPool to ensure connection reuse
        echo=False
    )
    Base.metadata.create_all(engine)
    yield engine
    Base.metadata.drop_all(engine)


@pytest.fixture(scope='function')
def db_session(test_engine):
    # Create a session using the shared test engine
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)
    session = SessionLocal()
    yield session
    session.close()


@pytest.fixture(scope='function')
def client(test_engine):
    # Create a SessionLocal bound to the same test engine
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)

    def _get_test_db():
        db = SessionLocal()
        try:
            # Ensure tables exist in this session's connection
            Base.metadata.create_all(db.get_bind())
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = _get_test_db

    client = TestClient(app)
    try:
        yield client
    finally:
        app.dependency_overrides.clear()