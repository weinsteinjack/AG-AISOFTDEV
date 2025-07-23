from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, EmailStr
from typing import List, Optional

app = FastAPI()

# Pydantic models
class UserCreate(BaseModel):
    name: str
    email: EmailStr
    role: str

class UserRead(BaseModel):
    id: int
    name: str
    email: EmailStr
    role: str

# In-memory database
fake_user_db = []
user_id_counter = 1

# Helper function to find a user by ID
def get_user_by_id(user_id: int) -> Optional[UserRead]:
    for user in fake_user_db:
        if user.id == user_id:
            return user
    return None

# POST /users
@app.post("/users", response_model=UserRead)
def create_user(user: UserCreate):
    global user_id_counter
    new_user = UserRead(id=user_id_counter, **user.dict())
    fake_user_db.append(new_user)
    user_id_counter += 1
    return new_user

# GET /users
@app.get("/users", response_model=List[UserRead])
def get_users():
    return fake_user_db

# GET /users/{user_id}
@app.get("/users/{user_id}", response_model=UserRead)
def get_user(user_id: int):
    user = get_user_by_id(user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return user