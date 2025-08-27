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
    return users_db

# Get user by ID
@app.get("/users/{user_id}", response_model=User)
async def read_user(user_id: int):
    for user in users_db:
        if user.id == user_id:
            return user
    raise HTTPException(status_code=404, detail="User not found")

# Create new user
@app.post("/users/", response_model=User)
async def create_user(user: UserCreate):
    new_user_id = len(users_db) + 1
    new_user = User(id=new_user_id, **user.dict())
    users_db.append(new_user)
    return new_user