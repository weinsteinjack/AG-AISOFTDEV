# main_in_memory.py

"""
A complete FastAPI application for an employee onboarding system.

This application provides a RESTful API for managing users within the system.
It uses an in-memory dictionary to simulate a database, making it a self-contained
and runnable example.

Features:
- Pydantic models for robust data validation and serialization.
- Full CRUD (Create, Read, Update, Delete) functionality for users.
- In-memory data storage that persists for the application's lifecycle.
- Detailed OpenAPI/Swagger documentation generated automatically by FastAPI.
- Best practices including type hints, dependency injection, and proper error handling.

To run this application:
1. Make sure you have the required packages installed:
   pip install "fastapi[all]"
2. Run the script from your terminal:
   python main_in_memory.py
3. The API will be available at http://127.0.0.1:8000
4. Interactive API documentation (Swagger UI) will be at http://127.0.0.1:8000/docs
"""

import uvicorn
from datetime import date, datetime
from enum import Enum
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException, status, Query
from pydantic import BaseModel, EmailStr, Field

# --- 1. Application Metadata and Initialization ---

app = FastAPI(
    title="Employee Onboarding System API",
    description="API for managing users and their onboarding process. This version uses an in-memory database.",
    version="1.0.0",
    contact={
        "name": "Senior Python Developer",
        "url": "https://github.com/your-profile",
        "email": "dev@example.com",
    },
)

# --- 2. In-Memory Database Simulation ---

# A simple dictionary to store our user data. The key is the user_id.
db_users: Dict[int, Dict[str, Any]] = {}
# A counter to simulate auto-incrementing primary keys
user_id_counter = 0

# Let's add some seed data to make the API explorable from the start
def seed_data():
    """Initializes the in-memory database with some sample data."""
    global user_id_counter
    
    # Reset database on each run
    db_users.clear()
    user_id_counter = 0

    # Sample users
    sample_users = [
        {
            "first_name": "Alice", "last_name": "Wonderland", "email": "alice@example.com",
            "job_title": "Software Engineer", "department": "Engineering",
            "start_date": date(2023, 8, 15), "role": "New Hire",
            "mentor_id": 2, "onboarding_path_id": 1
        },
        {
            "first_name": "Bob", "last_name": "Builder", "email": "bob@example.com",
            "job_title": "Engineering Manager", "department": "Engineering",
            "start_date": date(2022, 5, 20), "role": "Manager",
            "mentor_id": None, "onboarding_path_id": None
        },
        {
            "first_name": "Charlie", "last_name": "Chocolate", "email": "charlie@hr.com",
            "job_title": "HR Specialist", "department": "Human Resources",
            "start_date": date(2021, 1, 10), "role": "HR Admin",
            "mentor_id": None, "onboarding_path_id": None
        }
    ]

    for user_data in sample_users:
        user_id_counter += 1
        db_users[user_id_counter] = {
            "user_id": user_id_counter,
            "created_at": datetime.now(),
            **user_data
        }

# --- 3. Pydantic Model Definitions ---

class UserRole(str, Enum):
    """Enumeration for user roles to enforce valid values."""
    NEW_HIRE = "New Hire"
    HR_ADMIN = "HR Admin"
    CONTENT_CREATOR = "Content Creator"
    MANAGER = "Manager"
    MENTOR = "Mentor"

class UserBase(BaseModel):
    """Base model with common user attributes."""
    first_name: str = Field(..., min_length=1, max_length=50, example="John")
    last_name: str = Field(..., min_length=1, max_length=50, example="Doe")
    email: EmailStr = Field(..., example="john.doe@example.com")
    job_title: Optional[str] = Field(None, max_length=100, example="Software Developer")
    department: Optional[str] = Field(None, max_length=100, example="Technology")
    start_date: date = Field(..., example="2024-01-15")
    role: UserRole = Field(..., example=UserRole.NEW_HIRE)
    mentor_id: Optional[int] = Field(None, gt=0, example=2)
    onboarding_path_id: Optional[int] = Field(None, gt=0, example=1)

class UserCreate(UserBase):
    """
    Model for creating a new user.
    All fields from UserBase are required for creation.
    """
    pass

class UserUpdate(UserBase):
    """
    Model for updating a user with a full replacement (PUT).
    All fields from UserBase must be provided.
    """
    pass

class UserPartialUpdate(BaseModel):
    """
    Model for partially updating a user (PATCH).
    All fields are optional.
    """
    first_name: Optional[str] = Field(None, min_length=1, max_length=50, example="John")
    last_name: Optional[str] = Field(None, min_length=1, max_length=50, example="Doe")
    email: Optional[EmailStr] = Field(None, example="john.doe@example.com")
    job_title: Optional[str] = Field(None, max_length=100, example="Senior Software Developer")
    department: Optional[str] = Field(None, max_length=100, example="Technology")
    start_date: Optional[date] = Field(None, example="2024-01-15")
    role: Optional[UserRole] = Field(None, example=UserRole.MENTOR)
    mentor_id: Optional[int] = Field(None, gt=0, example=2)
    onboarding_path_id: Optional[int] = Field(None, gt=0, example=1)

class UserResponse(UserBase):
    """
    Model for API responses when returning user data.
    Includes auto-generated fields like `user_id` and `created_at`.
    """
    user_id: int = Field(..., gt=0, example=1)
    created_at: datetime = Field(..., example="2023-08-15T14:30:00Z")

    class Config:
        """Pydantic config to enable ORM mode (works with dicts too)."""
        from_attributes = True

# --- 4. Helper Functions ---

def find_user_by_id(user_id: int) -> Optional[Dict[str, Any]]:
    """Fetches a user from the in-memory DB by their ID."""
    return db_users.get(user_id)

def find_user_by_email(email: str) -> Optional[Dict[str, Any]]:
    """Fetches a user from the in-memory DB by their email."""
    for user in db_users.values():
        if user["email"] == email:
            return user
    return None

# --- 5. FastAPI Endpoints ---

@app.post(
    "/users/",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Users"],
    summary="Create a new user",
    description="Adds a new user to the system. The email address must be unique.",
)
def create_user(user_in: UserCreate) -> Any:
    """
    Create a new user record.

    - **first_name**: User's first name.
    - **last_name**: User's last name.
    - **email**: User's unique email address.
    - **job_title**: User's job position.
    - **department**: The department the user belongs to.
    - **start_date**: The official start date for the user.
    - **role**: The user's role in the onboarding system.
    - **mentor_id**: (Optional) The ID of the user's assigned mentor.
    - **onboarding_path_id**: (Optional) The ID of the assigned onboarding path.
    """
    if find_user_by_email(user_in.email):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"A user with the email '{user_in.email}' already exists.",
        )
    
    global user_id_counter
    user_id_counter += 1
    
    new_user = {
        "user_id": user_id_counter,
        "created_at": datetime.now(),
        **user_in.model_dump()
    }
    
    db_users[user_id_counter] = new_user
    return new_user

@app.get(
    "/users/",
    response_model=List[UserResponse],
    tags=["Users"],
    summary="List all users",
    description="Retrieves a list of all users with optional pagination.",
)
def list_users(
    skip: int = Query(0, ge=0, description="Number of records to skip for pagination"),
    limit: int = Query(10, ge=1, le=100, description="Maximum number of records to return")
) -> List[UserResponse]:
    """
    Retrieve a paginated list of all users.
    """
    users_list = list(db_users.values())
    return users_list[skip : skip + limit]

@app.get(
    "/users/{user_id}",
    response_model=UserResponse,
    tags=["Users"],
    summary="Get a specific user by ID",
    description="Retrieves the full details of a single user by their unique ID.",
    responses={404: {"description": "User not found"}}
)
def get_user(user_id: int) -> Any:
    """
    Get details for a specific user.
    """
    user = find_user_by_id(user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User with ID {user_id} not found.",
        )
    return user

@app.put(
    "/users/{user_id}",
    response_model=UserResponse,
    tags=["Users"],
    summary="Update a user (full update)",
    description="Replaces all data for a specific user. All fields in the request body are required.",
    responses={404: {"description": "User not found"}, 409: {"description": "Email already in use"}}
)
def update_user(user_id: int, user_in: UserUpdate) -> Any:
    """
    Perform a full update on a user's record.
    """
    user_to_update = find_user_by_id(user_id)
    if not user_to_update:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User with ID {user_id} not found.",
        )
        
    # Check if the new email is already taken by another user
    existing_user_with_email = find_user_by_email(user_in.email)
    if existing_user_with_email and existing_user_with_email["user_id"] != user_id:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"The email '{user_in.email}' is already associated with another account.",
        )

    # Convert Pydantic model to a dictionary
    update_data = user_in.model_dump()
    
    # Update the user in the database
    user_to_update.update(update_data)
    
    return user_to_update


@app.patch(
    "/users/{user_id}",
    response_model=UserResponse,
    tags=["Users"],
    summary="Partially update a user",
    description="Updates one or more fields for a specific user. Only include the fields you want to change.",
    responses={404: {"description": "User not found"}, 409: {"description": "Email already in use"}}
)
def partial_update_user(user_id: int, user_in: UserPartialUpdate) -> Any:
    """
    Perform a partial update on a user's record.
    """
    user_to_update = find_user_by_id(user_id)
    if not user_to_update:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User with ID {user_id} not found.",
        )
    
    # Get only the fields that were actually provided in the request
    update_data = user_in.model_dump(exclude_unset=True)

    if not update_data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No fields provided for update.",
        )

    # If email is being updated, check for conflicts
    if "email" in update_data:
        new_email = update_data["email"]
        existing_user_with_email = find_user_by_email(new_email)
        if existing_user_with_email and existing_user_with_email["user_id"] != user_id:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"The email '{new_email}' is already associated with another account.",
            )
    
    # Update the user dictionary with the new data
    user_to_update.update(update_data)
    
    return user_to_update


@app.delete(
    "/users/{user_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    tags=["Users"],
    summary="Delete a user",
    description="Permanently deletes a user from the system.",
    responses={404: {"description": "User not found"}}
)
def delete_user(user_id: int):
    """
    Delete a user by their ID.
    """
    if user_id not in db_users:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User with ID {user_id} not found.",
        )
    
    del db_users[user_id]
    # No content is returned for a 204 response
    return


# --- 6. Application Startup Logic ---

@app.on_event("startup")
async def startup_event():
    """
    This function runs when the application starts.
    It's a good place to initialize resources, like our seed data.
    """
    print("Starting up and seeding in-memory database...")
    seed_data()
    print(f"Database seeded with {len(db_users)} users.")


# --- 7. Main Block for Running the Application ---

if __name__ == "__main__":
    """
    This block allows the script to be run directly using `python main_in_memory.py`.
    It starts the Uvicorn server, which serves the FastAPI application.
    """
    uvicorn.run(
        "main_in_memory:app",
        host="127.0.0.1",
        port=8000,
        reload=True,  # `reload=True` is great for development
        log_level="info"
    )