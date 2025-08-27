from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

app = FastAPI()

class UserBase(BaseModel):
    name: str
    email: str
    role: str

class UserCreate(UserBase):
    pass

class User(UserBase):
    id: int

# Fake in-memory database
users_db = [
    User(id=1, name="John Doe", email="john.doe@example.com", role="New Hire"),
    User(id=2, name="Jane Smith", email="jane.smith@example.com", role="Manager"),
]

# Get all users
@app.get("/users/", response_model=List[User])
async def read_users():
    """
    Retrieves all users from the in-memory database.
    
    This endpoint returns a complete list of all users currently stored in the
    in-memory database. Unlike the SQLAlchemy version, this implementation doesn't
    support pagination and returns all users at once. It's suitable for small
    datasets and development/testing purposes.
    
    Args:
        None
    
    Returns:
        List[User]: A list of all user objects in the database, each containing:
            - id (int): Unique user identifier
            - name (str): User's full name
            - email (str): User's email address
            - role (str): User's role in the organization
            
            Returns empty list if no users exist.
    
    Raises:
        None: This endpoint doesn't raise exceptions as it simply returns
            the current state of the in-memory list.
    
    Notes:
        - No pagination support (returns all users)
        - Data is stored in memory and will be lost on server restart
        - Includes two pre-populated example users on startup
        - Async function for compatibility with FastAPI's async support
        - Consider adding pagination for production use with larger datasets
    
    Example:
        >>> # GET /users/
        >>> # Response (200 OK):
        >>> [
        ...     {
        ...         "id": 1,
        ...         "name": "John Doe",
        ...         "email": "john.doe@example.com",
        ...         "role": "New Hire"
        ...     },
        ...     {
        ...         "id": 2,
        ...         "name": "Jane Smith",
        ...         "email": "jane.smith@example.com",
        ...         "role": "Manager"
        ...     }
        ... ]
    
    Dependencies:
        - users_db: Global in-memory list storing User objects
        - User: Pydantic model defining user structure
    """
    return users_db

# Get user by ID
@app.get("/users/{user_id}", response_model=User)
async def read_user(user_id: int):
    """
    Retrieves a single user by their ID from the in-memory database.
    
    This endpoint searches through the in-memory user list to find and return
    a user with the specified ID. It performs a linear search through the list,
    which is acceptable for small datasets but may be slow for large ones.
    
    Args:
        user_id (int): The unique identifier of the user to retrieve.
            Must be a positive integer. The ID should correspond to an
            existing user in the in-memory database.
    
    Returns:
        User: A Pydantic model representing the requested user with fields:
            - id (int): The user's unique identifier (same as requested)
            - name (str): The user's full name
            - email (str): The user's email address
            - role (str): The user's role in the organization
    
    Raises:
        HTTPException:
            - 404 Not Found: If no user exists with the provided ID.
              Returns {"detail": "User not found"}
            - 422 Unprocessable Entity: If user_id is not a valid integer
              (handled automatically by FastAPI)
    
    Notes:
        - Performs O(n) linear search through the user list
        - The user_id is extracted from the URL path parameter
        - Data is volatile and stored only in memory
        - Async function for compatibility with FastAPI's async support
        - Consider using a dictionary for O(1) lookups in production
    
    Example:
        >>> # GET /users/1
        >>> # Response (200 OK):
        >>> {
        ...     "id": 1,
        ...     "name": "John Doe",
        ...     "email": "john.doe@example.com",
        ...     "role": "New Hire"
        ... }
        
        >>> # GET /users/999 (non-existent user)
        >>> # Response (404 Not Found):
        >>> {
        ...     "detail": "User not found"
        ... }
    
    Dependencies:
        - users_db: Global in-memory list storing User objects
        - HTTPException: FastAPI exception for HTTP error responses
    """
    for user in users_db:
        if user.id == user_id:
            return user
    raise HTTPException(status_code=404, detail="User not found")

# Create new user
@app.post("/users/", response_model=User)
async def create_user(user: UserCreate):
    """
    Creates a new user in the in-memory database.
    
    This endpoint accepts user information and creates a new user in the in-memory
    database. Unlike the SQLAlchemy version, this doesn't check for duplicate
    emails and uses a simple incremental ID generation strategy. The created user
    is immediately available but will be lost when the server restarts.
    
    Args:
        user (UserCreate): A Pydantic model containing the user data to create.
            Required fields:
            - name (str): Full name of the user
            - email (str): Email address for the user (uniqueness not enforced)
            - role (str): User's role in the organization
    
    Returns:
        User: A Pydantic model representing the created user with all fields:
            - id (int): Auto-generated unique identifier (sequential)
            - name (str): The provided user name
            - email (str): The provided email address
            - role (str): The provided user role
    
    Raises:
        None: This endpoint doesn't perform validation beyond Pydantic's
            type checking. Duplicate emails are allowed in this implementation.
    
    Notes:
        - IDs are generated sequentially based on current list length
        - No email uniqueness validation (allows duplicates)
        - No persistence - data is lost on server restart
        - Thread-safety issues possible with concurrent requests
        - The new user is appended to the end of the list
        - Async function for compatibility with FastAPI's async support
    
    Example:
        >>> # POST /users/
        >>> # Request body:
        >>> {
        ...     "name": "Alice Johnson",
        ...     "email": "alice.johnson@example.com",
        ...     "role": "Developer"
        ... }
        >>> # Response (201 Created):
        >>> {
        ...     "id": 3,
        ...     "name": "Alice Johnson",
        ...     "email": "alice.johnson@example.com",
        ...     "role": "Developer"
        ... }
    
    Dependencies:
        - users_db: Global in-memory list storing User objects
        - User: Pydantic model with id field
        - UserCreate: Pydantic model without id field
    """
    new_user_id = len(users_db) + 1
    new_user = User(id=new_user_id, **user.dict())
    users_db.append(new_user)
    return new_user