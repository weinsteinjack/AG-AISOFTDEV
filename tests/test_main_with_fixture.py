def test_create_user(client, db_session):
    user_data = {
        "email": "test@example.com",
        "name": "Test User",
        "role": "Developer"
    }
    response = client.post("/users/", json=user_data)
    assert response.status_code == 200
    assert response.json()["email"] == user_data["email"]

def test_read_users(client, db_session):
    user_data = {
        "email": "sample@example.com",
        "name": "Sample User",
        "role": "Manager"
    }
    # Create a user
    client.post("/users/", json=user_data)
    
    # Read users
    response = client.get("/users/")
    assert response.status_code == 200
    assert isinstance(response.json(), list)
    assert len(response.json()) > 0