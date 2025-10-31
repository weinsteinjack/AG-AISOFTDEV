# test_users_happy_path.py

"""
Pytest test suite for the "happy path" scenarios of the User CRUD endpoints.

This suite covers:
- Successful creation of a user (POST /users/).
- Successful retrieval of all users (GET /users/).
- Successful retrieval of a specific user by ID (GET /users/{user_id}).

Best Practices Followed:
- **Test Database Isolation**: A separate SQLite database (`test_onboarding.db`) is used for testing
  to ensure tests are independent and do not interfere with development data.
- **Dependency Injection Override**: FastAPI's dependency injection system is used to override
  the `get_db` dependency, pointing it to the test database during test execution.
- **Fixtures for Setup/Teardown**: A pytest fixture (`client`) handles the creation and cleanup
  of the database tables and the TestClient instance for each test function, ensuring a clean state.
- **Arrange-Act-Assert Pattern**: Each test is clearly structured for readability and maintainability.
- **Descriptive Naming and Docstrings**: Test functions and variables have clear names, and
  docstrings explain the purpose of each test.
"""

import os
import sys
from datetime import date, datetime

# Add the app directory to Python path to enable imports
current_dir = os.path.dirname(os.path.abspath(__file__))
app_dir = os.path.join(os.path.dirname(current_dir), 'app')
if app_dir not in sys.path:
    sys.path.insert(0, app_dir)

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Import the FastAPI app and other necessary components from the main application file
from main import Base, UserRole, app, get_db

# --- 1. Test Database Configuration ---

# Define the URL for the test database.
# Using a file-based SQLite DB for simplicity and speed.
TEST_DATABASE_URL = "sqlite:///./test_onboarding.db"

# Create the SQLAlchemy engine for the test database.
# `connect_args` is necessary for SQLite to allow multi-threaded access.
engine = create_engine(
    TEST_DATABASE_URL, connect_args={"check_same_thread": False}
)

# Create a session maker for the test database. This will be used to create sessions.
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


# --- 2. Dependency Override ---

def override_get_db():
    """
    A dependency override for get_db that provides a session to the test database.
    This function replaces the production `get_db` dependency during testing.
    It ensures that all API calls within a test use an isolated database session.
    """
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()

# Apply the dependency override to the FastAPI app.
# This tells the app to use `override_get_db` whenever `get_db` is requested.
app.dependency_overrides[get_db] = override_get_db


# --- 3. Pytest Fixture for Test Client and DB Management ---

@pytest.fixture(scope="function")
def client():
    """
    Pytest fixture to set up and tear down the test environment for each test function.

    Yields:
        TestClient: An instance of FastAPI's TestClient configured for the app.
    """
    # Arrange: Create all database tables before running a test.
    Base.metadata.create_all(bind=engine)
    
    # Yield the test client to the test function.
    with TestClient(app) as test_client:
        yield test_client
    
    # Teardown: Drop all database tables after the test is complete.
    # This ensures a clean state for the next test.
    Base.metadata.drop_all(bind=engine)
    # Optional: remove the test database file after all tests are done (can be done in a session-scoped fixture)
    # if os.path.exists("./test_onboarding.db"):
    #     os.remove("./test_onboarding.db")


# --- 4. Test Functions for Happy Path Scenarios ---

def test_create_user_success(client: TestClient):
    """
    Tests the successful creation of a new user via the POST /users/ endpoint.

    Scenario:
    - A valid user payload is sent to the API.

    Expected Outcome:
    - The API returns a 201 Created status code.
    - The response body contains the data of the newly created user.
    - Auto-generated fields like `user_id` and `created_at` are present and valid.
    """
    # Arrange: Define the valid payload for the new user.
    new_user_data = {
        "first_name": "Jane",
        "last_name": "Smith",
        "email": "jane.smith@example.com",
        "job_title": "Product Manager",
        "department": "Product",
        "start_date": "2024-02-01",
        "role": UserRole.MANAGER.value, # Use the enum value for correctness
        "mentor_id": None,
        "onboarding_path_id": None,
    }

    # Act: Make the POST request to the /users/ endpoint.
    response = client.post("/users/", json=new_user_data)

    # Assert: Verify the response.
    assert response.status_code == 201, f"Expected status 201, but got {response.status_code}"
    
    data = response.json()
    
    # Verify the structure and auto-generated fields
    assert "user_id" in data
    assert isinstance(data["user_id"], int)
    assert data["user_id"] > 0
    
    assert "created_at" in data
    # Attempt to parse the datetime string to ensure it's a valid format
    try:
        datetime.fromisoformat(data["created_at"].replace('Z', '+00:00'))
    except ValueError:
        pytest.fail("`created_at` is not a valid ISO 8601 datetime string.")

    # Verify the returned data matches the input data
    assert data["first_name"] == new_user_data["first_name"]
    assert data["last_name"] == new_user_data["last_name"]
    assert data["email"] == new_user_data["email"]
    assert data["job_title"] == new_user_data["job_title"]
    assert data["start_date"] == new_user_data["start_date"]
    assert data["role"] == new_user_data["role"]


def test_list_users_success(client: TestClient):
    """
    Tests the successful retrieval of a list of users via the GET /users/ endpoint.

    Scenario:
    - Create two users to populate the database.
    - Make a GET request to retrieve all users.

    Expected Outcome:
    - The API returns a 200 OK status code.
    - The response body is a list containing the two created users.
    - The structure of each user object in the list is correct.
    """
    # Arrange: Create two users to ensure the database is not empty.
    user1_data = {
        "first_name": "Alice",
        "last_name": "Wonderland",
        "email": "alice.w@example.com",
        "start_date": "2023-10-01",
        "role": "New Hire",
    }
    user2_data = {
        "first_name": "Bob",
        "last_name": "Builder",
        "email": "bob.b@example.com",
        "start_date": "2023-11-15",
        "role": "Mentor",
    }
    client.post("/users/", json=user1_data)
    client.post("/users/", json=user2_data)

    # Act: Make the GET request to the /users/ endpoint.
    response = client.get("/users/")

    # Assert: Verify the response.
    assert response.status_code == 200, f"Expected status 200, but got {response.status_code}"
    
    data = response.json()
    
    assert isinstance(data, list), "Response should be a list of users."
    assert len(data) == 2, "Expected to retrieve 2 users."
    
    # Verify the content and structure of the returned users
    assert data[0]["email"] == user1_data["email"]
    assert data[1]["email"] == user2_data["email"]
    assert "user_id" in data[0]
    assert "created_at" in data[1]


def test_get_specific_user_success(client: TestClient):
    """
    Tests the successful retrieval of a single user by ID via GET /users/{user_id}.

    Scenario:
    - Create a new user.
    - Use the ID from the creation response to fetch that specific user.

    Expected Outcome:
    - The API returns a 200 OK status code.
    - The response body contains the correct details for the requested user.
    """
    # Arrange: Create a user to fetch.
    user_to_create = {
        "first_name": "Charlie",
        "last_name": "Chocolate",
        "email": "charlie.c@example.com",
        "job_title": "Confectioner",
        "department": "Sweets",
        "start_date": "2024-03-10",
        "role": UserRole.CONTENT_CREATOR.value,
    }
    create_response = client.post("/users/", json=user_to_create)
    assert create_response.status_code == 201
    created_user_id = create_response.json()["user_id"]

    # Act: Make the GET request to fetch the newly created user by their ID.
    response = client.get(f"/users/{created_user_id}")

    # Assert: Verify the response.
    assert response.status_code == 200, f"Expected status 200, but got {response.status_code}"
    
    data = response.json()
    
    # Verify that the retrieved user's data matches the data used for creation.
    assert data["user_id"] == created_user_id
    assert data["email"] == user_to_create["email"]
    assert data["first_name"] == user_to_create["first_name"]
    assert data["last_name"] == user_to_create["last_name"]
    assert data["job_title"] == user_to_create["job_title"]
    assert data["department"] == user_to_create["department"]
    assert data["start_date"] == user_to_create["start_date"]
    assert data["role"] == user_to_create["role"]