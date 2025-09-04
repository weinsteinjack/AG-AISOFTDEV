import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from app.db_models import Base, User, OnboardingTask

def test_database_setup():
    """Test that the database setup works correctly"""
    # Create an in-memory database
    engine = create_engine('sqlite:///:memory:', connect_args={"check_same_thread": False})
    
    # Create all tables
    Base.metadata.create_all(engine)
    
    # Check that tables were created
    with engine.connect() as conn:
        result = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table';"))
        tables = [row[0] for row in result]
        print(f"Created tables: {tables}")
        assert 'users' in tables
        assert 'onboarding_tasks' in tables
    
    # Test creating a user
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = SessionLocal()
    
    new_user = User(
        name="test",
        email="test@example.com",
        role="New Hire",
        hashed_password="hashed",
        is_active=True
    )
    session.add(new_user)
    session.commit()
    
    # Query the user back
    db_user = session.query(User).filter(User.email == "test@example.com").first()
    assert db_user is not None
    assert db_user.email == "test@example.com"
    
    session.close()
    print("Database setup test passed!")

if __name__ == "__main__":
    test_database_setup()
