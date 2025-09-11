from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from app.main import app, get_db
from app.db_models import Base, User
from utils.logging import get_logger

logger = get_logger()

def test_dependency_override():
    """Test that dependency override works"""
    # Create test database
    engine = create_engine('sqlite:///:memory:', connect_args={"check_same_thread": False})
    
    # Debug: Check what tables are registered with Base
    logger.debug("Base metadata tables: %s", list(Base.metadata.tables.keys()))
    
    # Create all tables
    Base.metadata.create_all(engine)
    
    # Debug: Check what tables were actually created
    with engine.connect() as conn:
        result = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table';"))
        tables = [row[0] for row in result]
        logger.debug("Created tables in test DB: %s", tables)
    
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    # Test the database session directly first
    test_session = SessionLocal()
    try:
        # Try to query directly using our test session
        users = test_session.query(User).all()
        logger.debug(
            "Direct query on test session successful: %d users found", len(users)
        )
    except Exception as e:
        logger.error("Direct query failed: %s", e)
    finally:
        test_session.close()

    def _get_test_db():
        logger.debug("_get_test_db called - dependency override is working!")
        db = SessionLocal()
        try:
            # Debug: Check which database this session is connected to
            with db.get_bind().connect() as conn:
                result = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table';"))
                tables = [row[0] for row in result]
                logger.debug("Tables in dependency override session: %s", tables)
            
            # If no tables, create them on this engine
            if not tables:
                logger.info("No tables found, creating them...")
                Base.metadata.create_all(db.get_bind())
                # Check again
                with db.get_bind().connect() as conn:
                    result = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table';"))
                    tables = [row[0] for row in result]
                    logger.debug("Tables after creation: %s", tables)
            
            yield db
        finally:
            db.close()

    # Override dependency
    logger.debug("Setting dependency override...")
    app.dependency_overrides[get_db] = _get_test_db
    
    # Create test client
    client = TestClient(app)
    
    try:
        # Test the endpoint
        logger.debug("Making POST request...")
        response = client.post(
            "/users/",
            json={"email": "test@example.com", "password": "testpassword"},
        )
        logger.debug("Response status: %s", response.status_code)
        logger.debug("Response body: %s", response.text)

        if response.status_code == 200:
            logger.info("SUCCESS: User created successfully!")
        else:
            logger.error("FAILED: Status %s", response.status_code)

    except Exception as e:
        logger.error("ERROR: %s", e)
        
    finally:
        app.dependency_overrides.clear()

if __name__ == "__main__":
    test_dependency_override()
