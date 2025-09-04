def test_create_user(client, db_session):
    response = client.post("/users/", json={"email": "test@example.com", "password": "testpassword"})
    assert response.status_code == 200
    assert response.json()["email"] == "test@example.com"

def test_get_user(client, db_session):
    # First, create a user
    create_response = client.post("/users/", json={"email": "test@example.com", "password": "testpassword"})
    assert create_response.status_code == 200

    # Now, retrieve the user
    user_id = create_response.json()["id"]
    response = client.get(f"/users/{user_id}")
    assert response.status_code == 200
    assert response.json()["email"] == "test@example.com"

def test_get_nonexistent_user(client, db_session):
    response = client.get("/users/99999")  # Assuming 99999 is a non-existent ID
    assert response.status_code == 404