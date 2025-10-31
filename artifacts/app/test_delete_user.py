"""Test script to create and delete a temporary user"""
import urllib.request
import urllib.parse
import json
import time
import random

# Wait for server to start
print("Waiting for server to start...")
time.sleep(3)

BASE_URL = "http://127.0.0.1:8000"

# Create a temporary user
test_email = f"testuser_temp_{random.randint(1000, 9999)}@example.com"
user_data = {
    "first_name": "Test",
    "last_name": "User",
    "email": test_email,
    "job_title": "QA Tester",
    "department": "Testing",
    "start_date": "2024-01-15",
    "role": "New Hire"
}

print(f"\n1. Creating temporary user with email: {test_email}")
data = json.dumps(user_data).encode('utf-8')
req = urllib.request.Request(
    f"{BASE_URL}/users/",
    data=data,
    headers={'Content-Type': 'application/json'},
    method='POST'
)

try:
    with urllib.request.urlopen(req) as response:
        if response.status == 201:
            created_user = json.loads(response.read().decode('utf-8'))
            user_id = created_user["user_id"]
            print(f"✓ User created successfully!")
            print(f"  User ID: {user_id}")
            print(f"  Name: {created_user['first_name']} {created_user['last_name']}")
            print(f"  Email: {created_user['email']}")
            
            # Delete the user
            print(f"\n2. Deleting user with ID: {user_id}")
            delete_req = urllib.request.Request(
                f"{BASE_URL}/users/{user_id}",
                method='DELETE'
            )
            
            with urllib.request.urlopen(delete_req) as delete_response:
                if delete_response.status == 200:
                    delete_message = json.loads(delete_response.read().decode('utf-8'))
                    print(f"✓ User deleted successfully!")
                    print(f"  Response: {json.dumps(delete_message, indent=2)}")
                    
                    # Verify user is actually deleted
                    print(f"\n3. Verifying user is deleted...")
                    verify_req = urllib.request.Request(f"{BASE_URL}/users/{user_id}")
                    try:
                        with urllib.request.urlopen(verify_req) as verify_response:
                            print(f"✗ WARNING: User still exists! Status: {verify_response.status}")
                    except urllib.error.HTTPError as e:
                        if e.code == 404:
                            print(f"✓ Verification successful - user not found (404)")
                        else:
                            print(f"✗ Unexpected error: {e.code}")
                else:
                    print(f"✗ Failed to delete user. Status: {delete_response.status}")
        else:
            print(f"✗ Failed to create user. Status: {response.status}")
except urllib.error.HTTPError as e:
    print(f"✗ Error: {e.code} - {e.read().decode('utf-8')}")
except Exception as e:
    print(f"✗ Error: {e}")

