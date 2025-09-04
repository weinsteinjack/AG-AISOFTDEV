import time
from fastapi.testclient import TestClient
from app.main import app  # Use absolute import so pytest can find the package

client = TestClient(app)

def test_create_user():
    # Use a unique email to avoid conflicts with existing data
    unique_email = f"test_{int(time.time())}@example.com"
    # Send a request to create a user
    response = client.post("/users/", json={"email": unique_email, "password": "testpassword"})
    # Assert that the response status code is 200
    assert response.status_code == 200
    # Assert that the response email matches the input email
    assert response.json()["email"] == unique_email

def test_read_users():
    # Create a user first with unique email
    unique_email = f"test_read_{int(time.time())}@example.com"
    client.post("/users/", json={"email": unique_email, "password": "testpassword"})
    
    # Send a request to retrieve the list of users
    response = client.get("/users/")
    # Assert that the response status code is 200
    assert response.status_code == 200
    # Assert that the response contains a list with at least one user
    assert len(response.json()) > 0