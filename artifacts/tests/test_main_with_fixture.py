# artifacts/tests/test_main_with_fixture.py

"""
Pytest integration tests for the FastAPI Employee Onboarding System.

This test suite validates the CRUD functionality of the /users/ endpoint.
It is specifically designed to use pytest fixtures defined in a conftest.py
file, ensuring proper database isolation for each test function.

Key Features of this Test Suite:
- Utilizes the `client` fixture for making API requests to a test instance
  of the FastAPI application.
- Relies on fixtures for automatic setup and teardown of a clean, in-memory
  test database for each test, guaranteeing test isolation.
- Follows the Arrange-Act-Assert pattern for clear and maintainable tests.
- Includes comprehensive docstrings and assertion messages for clarity.
- Adheres to pytest best practices for testing FastAPI applications.
"""

# --- 1. Path Setup Section (MANDATORY) ---
# This block ensures that the main application code in the 'app' directory
# can be imported correctly by the test suite located in the 'tests' directory.
# This setup is crucial for the test runner to find the application modules.
import os
import sys

# Add the app directory to Python path to enable imports
current_dir = os.path.dirname(os.path.abspath(__file__))
app_dir = os.path.join(os.path.dirname(current_dir), 'app')
if app_dir not in sys.path:
    sys.path.insert(0, app_dir)

# --- 2. Imports Section ---
from datetime import date
import pytest
from fastapi.testclient import TestClient

# Imports from the application code (main.py)
# Note: We only import necessary models/enums, not the `app` or `get_db`
# objects, as those are managed by the test fixtures.
from main import UserRole


# --- 3. Test Functions Section ---

def test_create_user_success(client: TestClient):
    """
    Test successful user creation via POST /users/.

    - Arrange: Define valid user data payload.
    - Act: Send a POST request to the /users/ endpoint.
    - Assert:
        - The HTTP status code is 201 (Created).
        - The response body contains the user data sent in the request.
        - The response body includes auto-generated fields 'user_id' and 'created_at'.
    """
    # Arrange: Define valid data for a new user
    user_data = {
        "first_name": "Jane",
        "last_name": "Smith",
        "email": "jane.smith@example.com",
        "job_title": "Product Manager",
        "department": "Product",
        "start_date": "2024-03-01",
        "role": UserRole.MANAGER.value, # Use enum value for correctness
        "mentor_id": None,
        "onboarding_path_id": None,
    }

    # Act: Send the POST request to the API
    response = client.post("/users/", json=user_data)

    # Assert: Verify the response
    assert response.status_code == 201, f"Expected status 201, but got {response.status_code}"
    response_data = response.json()

    # Verify auto-generated fields are present
    assert "user_id" in response_data, "Response should contain 'user_id'"
    assert "created_at" in response_data, "Response should contain 'created_at'"
    assert isinstance(response_data["user_id"], int), "'user_id' should be an integer"

    # Verify the returned data matches the input data
    for key, value in user_data.items():
        assert response_data[key] == value, f"Field '{key}' did not match"


def test_list_users_success(client: TestClient):
    """
    Test retrieving a list of users via GET /users/.

    - Arrange: Create a new user to ensure the database is not empty.
    - Act: Send a GET request to the /users/ endpoint.
    - Assert:
        - The HTTP status code is 200 (OK).
        - The response body is a list.
        - The created user is present in the returned list.
    """
    # Arrange: Create a user so the list is not empty
    user_to_create = {
        "first_name": "Alex",
        "last_name": "Ray",
        "email": "alex.ray@example.com",
        "job_title": "Data Scientist",
        "department": "Analytics",
        "start_date": "2024-05-10",
        "role": UserRole.NEW_HIRE.value,
    }
    create_response = client.post("/users/", json=user_to_create)
    assert create_response.status_code == 201, "Setup failed: Could not create user"

    # Act: Retrieve the list of all users
    response = client.get("/users/")

    # Assert: Verify the response
    assert response.status_code == 200, f"Expected status 200, but got {response.status_code}"
    users_list = response.json()
    assert isinstance(users_list, list), "The response should be a list of users"
    assert len(users_list) > 0, "The users list should not be empty"

    # Check if the created user is in the list
    emails_in_response = [user["email"] for user in users_list]
    assert user_to_create["email"] in emails_in_response, \
        "The newly created user was not found in the list"


def test_get_specific_user_success(client: TestClient):
    """
    Test retrieving a single, specific user by ID via GET /users/{user_id}.

    - Arrange: Create a new user and extract their 'user_id' from the response.
    - Act: Send a GET request to /users/{user_id} using the extracted ID.
    - Assert:
        - The HTTP status code is 200 (OK).
        - The response body contains the correct data for the requested user.
        - The 'user_id' in the response matches the requested ID.
    """
    # Arrange: Create a user and get their ID
    user_to_create = {
        "first_name": "Sam",
        "last_name": "Jones",
        "email": "sam.jones@example.com",
        "job_title": "DevOps Engineer",
        "department": "Engineering",
        "start_date": str(date(2024, 6, 20)), # Using datetime for robustness
        "role": UserRole.MENTOR.value,
    }
    create_response = client.post("/users/", json=user_to_create)
    assert create_response.status_code == 201, "Setup failed: Could not create user"
    created_user_data = create_response.json()
    user_id = created_user_data["user_id"]

    # Act: Request the specific user by their ID
    response = client.get(f"/users/{user_id}")

    # Assert: Verify the response
    assert response.status_code == 200, f"Expected status 200, but got {response.status_code}"
    retrieved_user = response.json()

    # Verify that the retrieved user's ID and email match the created user
    assert retrieved_user["user_id"] == user_id, "User ID in response does not match requested ID"
    assert retrieved_user["email"] == user_to_create["email"], "User email does not match"
    assert retrieved_user["first_name"] == user_to_create["first_name"], "User first name does not match"
    assert retrieved_user["start_date"] == user_to_create["start_date"], "User start date does not match"