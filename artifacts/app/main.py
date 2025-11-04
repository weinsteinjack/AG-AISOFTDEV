# main.py - Database-connected FastAPI application

"""
A complete FastAPI application for an employee onboarding system.

This application provides a RESTful API for managing users within the system.
It connects to a live SQLite database using SQLAlchemy ORM.

Features:
- Pydantic models for robust data validation and serialization.
- Full CRUD (Create, Read, Update, Delete) functionality for users.
- Live database connectivity with SQLAlchemy ORM.
- Detailed OpenAPI/Swagger documentation generated automatically by FastAPI.
- Best practices including type hints, dependency injection, and proper error handling.

To run this application:
1. Make sure you have the required packages installed:
   pip install "fastapi[all]" sqlalchemy
2. Run the script from your terminal:
   python main.py
3. The API will be available at http://127.0.0.1:8000
4. Interactive API documentation (Swagger UI) will be at http://127.0.0.1:8000/docs
"""

import uvicorn
from datetime import date, datetime
from enum import Enum
from typing import List, Optional, Dict, Any, Generator
import logging
import os
import enum
import uuid

from fastapi import FastAPI, HTTPException, status, Query, Depends
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy import (
    CheckConstraint,
    Column,
    Date,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Table,
    Text,
    UniqueConstraint,
    func,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, Session, sessionmaker
from pydantic_settings import BaseSettings

# --- 1. Application Metadata and Initialization ---

app = FastAPI(
    title="Employee Onboarding System API",
    description="API for managing users and their onboarding process. This version connects to a live SQLite database.",
    version="1.0.0",
    contact={
        "name": "Senior Python Developer",
        "url": "https://github.com/your-profile",
        "email": "dev@example.com",
    },
)

# --- 2. Pydantic Model Definitions ---

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

class ChatRequest(BaseModel):
    """
    Model for chat requests to the agent.
    """
    question: str = Field(..., min_length=1, example="What is the onboarding process?")
    session_id: Optional[str] = Field(None, example="550e8400-e29b-41d4-a716-446655440000")

class ChatResponse(BaseModel):
    """
    Model for chat responses from the agent.
    """
    answer: str = Field(..., example="The onboarding process includes...")

class StatefulChatResponse(BaseModel):
    """
    Model for stateful chat responses from the agent.
    """
    answer: str = Field(..., example="The onboarding process includes...")
    session_id: str = Field(..., example="550e8400-e29b-41d4-a716-446655440000")

# --- 3. SQLAlchemy Models ---

# Base class for all SQLAlchemy models
class Base(DeclarativeBase):
    """Base class for all SQLAlchemy models."""
    pass

# --- Enums for CHECK constraints ---

class UserRoleEnum(enum.Enum):
    """Enumeration for user roles."""
    NEW_HIRE = "New Hire"
    HR_ADMIN = "HR Admin"
    CONTENT_CREATOR = "Content Creator"
    MANAGER = "Manager"
    MENTOR = "Mentor"

class TaskType(enum.Enum):
    """Enumeration for onboarding task types."""
    FORM = "FORM"
    READING = "READING"
    VIDEO = "VIDEO"
    MEETING = "MEETING"

class TaskStatus(enum.Enum):
    """Enumeration for the status of a user's task."""
    NOT_STARTED = "Not Started"
    IN_PROGRESS = "In Progress"
    COMPLETED = "Completed"
    BLOCKED = "Blocked"

class DocumentStatus(enum.Enum):
    """Enumeration for the status of a user's document submission."""
    PENDING = "Pending"
    SUBMITTED = "Submitted"
    APPROVED = "Approved"
    REJECTED = "Rejected"

class ArticleStatus(enum.Enum):
    """Enumeration for the status of a knowledge base article."""
    DRAFT = "Draft"
    PUBLISHED = "Published"
    ARCHIVED = "Archived"

# --- Association Table for Many-to-Many Relationship ---
article_tags_table = Table(
    "Article_Tags",
    Base.metadata,
    Column("article_id", Integer, ForeignKey("KB_Articles.article_id", ondelete="CASCADE"), primary_key=True),
    Column("tag_id", Integer, ForeignKey("KB_Tags.tag_id", ondelete="CASCADE"), primary_key=True),
)

# --- Model Classes ---

class OnboardingPath(Base):
    """
    Represents a structured sequence of tasks for a new hire.
    
    An Onboarding Path is a template that can be assigned to new hires,
    defining their onboarding journey. For example, 'Software Engineer Path'
    or 'Sales Team Path'.
    
    Relationships:
        - users: One-to-many relationship with User.
        - path_tasks: One-to-many relationship with the PathTask association object.
    """
    __tablename__ = "Onboarding_Paths"

    path_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    path_name: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    description: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    # Relationships
    users: Mapped[List["User"]] = relationship(back_populates="onboarding_path")
    path_tasks: Mapped[List["PathTask"]] = relationship(
        back_populates="path", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<OnboardingPath(path_id={self.path_id}, name='{self.path_name}')>"

class User(Base):
    """
    Represents a user in the system.
    
    Users can be new hires, HR admins, managers, etc. A user can also be a
    mentor to another user (self-referential relationship).
    
    Relationships:
        - onboarding_path: Many-to-one relationship with OnboardingPath.
        - mentor: Many-to-one self-referential relationship.
        - mentees: One-to-many self-referential relationship.
        - task_statuses: One-to-many relationship with UserTaskStatus association object.
        - documents: One-to-many relationship with UserDocument association object.
        - authored_articles: One-to-many relationship with KBArticle.
        - reviewed_documents: One-to-many relationship with UserDocument.
    """
    __tablename__ = "Users"

    user_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    first_name: Mapped[str] = mapped_column(Text, nullable=False)
    last_name: Mapped[str] = mapped_column(Text, nullable=False)
    email: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    job_title: Mapped[Optional[str]] = mapped_column(Text)
    department: Mapped[Optional[str]] = mapped_column(Text)
    start_date: Mapped[date] = mapped_column(Date, nullable=False)
    role: Mapped[str] = mapped_column(Text, nullable=False)
    mentor_id: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("Users.user_id"))
    onboarding_path_id: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("Onboarding_Paths.path_id"))
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    __table_args__ = (
        CheckConstraint(
            "role IN ('New Hire', 'HR Admin', 'Content Creator', 'Manager', 'Mentor')",
            name="user_role_check"
        ),
    )

    # Relationships
    onboarding_path: Mapped[Optional["OnboardingPath"]] = relationship(back_populates="users")
    
    # Self-referential relationship for mentor/mentee
    mentor: Mapped[Optional["User"]] = relationship(
        "User", remote_side=[user_id], back_populates="mentees"
    )
    mentees: Mapped[List["User"]] = relationship(
        "User", back_populates="mentor"
    )

    task_statuses: Mapped[List["UserTaskStatus"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )
    documents: Mapped[List["UserDocument"]] = relationship(
        back_populates="user", foreign_keys="[UserDocument.user_id]", cascade="all, delete-orphan"
    )
    authored_articles: Mapped[List["KBArticle"]] = relationship(
        back_populates="author", foreign_keys="[KBArticle.author_id]"
    )
    reviewed_documents: Mapped[List["UserDocument"]] = relationship(
        back_populates="reviewer", foreign_keys="[UserDocument.reviewed_by_user_id]"
    )

    def __repr__(self) -> str:
        return f"<User(user_id={self.user_id}, email='{self.email}')>"

class OnboardingTask(Base):
    """
    Represents an individual task that can be part of an onboarding path.
    
    A task can be a form to fill, a document to read, a video to watch,
    or a meeting to attend.
    """
    __tablename__ = "Onboarding_Tasks"

    task_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    task_title: Mapped[str] = mapped_column(Text, nullable=False)
    task_description: Mapped[Optional[str]] = mapped_column(Text)
    task_type: Mapped[str] = mapped_column(Text, nullable=False)
    content_url: Mapped[Optional[str]] = mapped_column(Text)
    estimated_duration_minutes: Mapped[Optional[int]] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    __table_args__ = (
        CheckConstraint(
            "task_type IN ('FORM', 'READING', 'VIDEO', 'MEETING')",
            name="task_type_check"
        ),
    )

    # Relationships
    path_tasks: Mapped[List["PathTask"]] = relationship(
        back_populates="task", cascade="all, delete-orphan"
    )
    user_statuses: Mapped[List["UserTaskStatus"]] = relationship(
        back_populates="task", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<OnboardingTask(task_id={self.task_id}, title='{self.task_title}')>"

class PathTask(Base):
    """
    Association object between OnboardingPath and OnboardingTask.
    
    This model represents the many-to-many relationship and includes
    extra data, `task_order`, which defines the sequence of tasks
    within a specific path.
    """
    __tablename__ = "Path_Tasks"

    path_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("Onboarding_Paths.path_id", ondelete="CASCADE"), primary_key=True
    )
    task_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("Onboarding_Tasks.task_id", ondelete="CASCADE"), primary_key=True
    )
    task_order: Mapped[int] = mapped_column(Integer, nullable=False)

    # Relationships
    path: Mapped["OnboardingPath"] = relationship(back_populates="path_tasks")
    task: Mapped["OnboardingTask"] = relationship(back_populates="path_tasks")

    def __repr__(self) -> str:
        return f"<PathTask(path_id={self.path_id}, task_id={self.task_id}, order={self.task_order})>"

class UserTaskStatus(Base):
    """
    Association object between User and OnboardingTask.
    
    This model tracks the status of a specific task for a specific user,
    including completion date and any relevant notes.
    """
    __tablename__ = "User_Task_Status"

    status_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("Users.user_id", ondelete="CASCADE"), nullable=False
    )
    task_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("Onboarding_Tasks.task_id", ondelete="CASCADE"), nullable=False
    )
    status: Mapped[str] = mapped_column(Text, nullable=False, default="Not Started")
    completion_date: Mapped[Optional[datetime]] = mapped_column(DateTime)
    notes: Mapped[Optional[str]] = mapped_column(Text)

    __table_args__ = (
        UniqueConstraint("user_id", "task_id", name="uq_user_task"),
        CheckConstraint(
            "status IN ('Not Started', 'In Progress', 'Completed', 'Blocked')",
            name="user_task_status_check"
        ),
    )

    # Relationships
    user: Mapped["User"] = relationship(back_populates="task_statuses")
    task: Mapped["OnboardingTask"] = relationship(back_populates="user_statuses")

    def __repr__(self) -> str:
        return f"<UserTaskStatus(user_id={self.user_id}, task_id={self.task_id}, status='{self.status}')>"

class Document(Base):
    """
    Represents a document template, like an I-9 form or an NDA.
    
    These are templates that users might need to fill out and submit.
    """
    __tablename__ = "Documents"

    document_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    document_name: Mapped[str] = mapped_column(Text, nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text)
    template_url: Mapped[Optional[str]] = mapped_column(Text)

    # Relationships
    user_documents: Mapped[List["UserDocument"]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Document(document_id={self.document_id}, name='{self.document_name}')>"

class UserDocument(Base):
    """
    Association object between User and Document.
    
    This tracks an instance of a document for a specific user, including its
    submission status, submitted data, and review information.
    """
    __tablename__ = "User_Documents"

    user_document_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("Users.user_id", ondelete="CASCADE"), nullable=False)
    document_id: Mapped[int] = mapped_column(Integer, ForeignKey("Documents.document_id"), nullable=False)
    status: Mapped[str] = mapped_column(Text, nullable=False, default="Pending")
    submitted_data_url: Mapped[Optional[str]] = mapped_column(Text)
    submitted_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    reviewed_by_user_id: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("Users.user_id"))
    reviewed_at: Mapped[Optional[datetime]] = mapped_column(DateTime)

    __table_args__ = (
        CheckConstraint(
            "status IN ('Pending', 'Submitted', 'Approved', 'Rejected')",
            name="user_document_status_check"
        ),
    )

    # Relationships
    user: Mapped["User"] = relationship(
        back_populates="documents", foreign_keys=[user_id]
    )
    reviewer: Mapped[Optional["User"]] = relationship(
        back_populates="reviewed_documents", foreign_keys=[reviewed_by_user_id]
    )
    document: Mapped["Document"] = relationship(back_populates="user_documents")

    def __repr__(self) -> str:
        return f"<UserDocument(user_id={self.user_id}, doc_id={self.document_id}, status='{self.status}')>"

class KBCategory(Base):
    """Represents a category for knowledge base articles, e.g., 'HR Policies'."""
    __tablename__ = "KB_Categories"
    
    category_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    category_name: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    description: Mapped[Optional[str]] = mapped_column(Text)

    # Relationships
    articles: Mapped[List["KBArticle"]] = relationship(back_populates="category")

    def __repr__(self) -> str:
        return f"<KBCategory(id={self.category_id}, name='{self.category_name}')>"

class KBTag(Base):
    """Represents a tag that can be applied to knowledge base articles."""
    __tablename__ = "KB_Tags"

    tag_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    tag_name: Mapped[str] = mapped_column(Text, nullable=False, unique=True)

    # Relationships
    articles: Mapped[List["KBArticle"]] = relationship(
        secondary=article_tags_table, back_populates="tags"
    )

    def __repr__(self) -> str:
        return f"<KBTag(id={self.tag_id}, name='{self.tag_name}')>"

class KBArticle(Base):
    """
    Represents an article in the knowledge base.
    
    Relationships:
        - author: Many-to-one relationship with User.
        - category: Many-to-one relationship with KBCategory.
        - tags: Many-to-many relationship with KBTag.
    """
    __tablename__ = "KB_Articles"

    article_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    author_id: Mapped[int] = mapped_column(Integer, ForeignKey("Users.user_id"), nullable=False)
    category_id: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("KB_Categories.category_id"))
    status: Mapped[str] = mapped_column(Text, nullable=False, default="Draft")
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())

    __table_args__ = (
        CheckConstraint(
            "status IN ('Draft', 'Published', 'Archived')",
            name="kb_article_status_check"
        ),
    )

    # Relationships
    author: Mapped["User"] = relationship(
        back_populates="authored_articles", foreign_keys=[author_id]
    )
    category: Mapped[Optional["KBCategory"]] = relationship(back_populates="articles")
    tags: Mapped[List["KBTag"]] = relationship(
        secondary=article_tags_table, back_populates="articles"
    )

    def __repr__(self) -> str:
        return f"<KBArticle(id={self.article_id}, title='{self.title}')>"

# --- 4. Database Session Management ---

class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    """
    # Default to a local SQLite database file named 'onboarding.db'
    DATABASE_URL: str = os.environ.get("DATABASE_URL", "sqlite:///./onboarding.db")
    
    # SQLAlchemy connection pool settings
    DB_POOL_SIZE: int = 5
    DB_MAX_OVERFLOW: int = 10
    DB_POOL_TIMEOUT: int = 30

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"  # Ignore extra environment variables not defined in Settings

# Instantiate settings
settings = Settings()

# Set up basic logging for database operations
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Connection arguments
connect_args = {}
if settings.DATABASE_URL.startswith("sqlite"):
    # For SQLite, we need to disable same-thread checking as FastAPI
    # can use different threads for a single request.
    connect_args["check_same_thread"] = False
    logger.info("Configuring engine for SQLite database.")
else:
    logger.info(f"Configuring engine for database: {settings.DATABASE_URL.split('@')[-1]}")

try:
    # Create the SQLAlchemy engine
    engine = create_engine(
        settings.DATABASE_URL,
        connect_args=connect_args,
        pool_size=settings.DB_POOL_SIZE,
        max_overflow=settings.DB_MAX_OVERFLOW,
        pool_timeout=settings.DB_POOL_TIMEOUT,
        pool_pre_ping=True,
        echo=False  # Set to True to see generated SQL in logs
    )

    # Create a configured "Session" class
    SessionLocal = sessionmaker(
        autocommit=False, 
        autoflush=False, 
        bind=engine
    )
    logger.info("Database engine and session maker configured successfully.")

except Exception as e:
    logger.error(f"Failed to initialize database engine: {e}")
    raise

def get_db() -> Generator[Session, None, None]:
    """
    FastAPI dependency that provides a SQLAlchemy database session.

    This function is a generator that creates a new Session for each request,
    yields it to the path operation function, and then ensures it's closed
    after the request is finished, even if an error occurs.

    Usage:
        @app.get("/items/")
        def read_items(db: Session = Depends(get_db)):
            # ... use db session ...
    """
    db = SessionLocal()
    try:
        yield db
    except Exception as e:
        logger.error(f"An error occurred during a database session: {e}")
        db.rollback() # Rollback the transaction on error
        raise
    finally:
        db.close()

def create_database_tables():
    """
    Creates all database tables defined by the models.
    This should be called once during application startup or via a CLI command.
    """
    logger.info("Creating database tables...")
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully.")
    except Exception as e:
        logger.error(f"Could not create database tables: {e}")
        raise

# --- 5. FastAPI Endpoints ---

@app.post(
    "/users/",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Users"],
    summary="Create a new user",
    description="Adds a new user to the system. The email address must be unique.",
)
def create_user(user_in: UserCreate, db: Session = Depends(get_db)) -> UserResponse:
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
    # Check if user with this email already exists
    existing_user = db.query(User).filter(User.email == user_in.email).first()
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"A user with the email '{user_in.email}' already exists.",
        )
    
    # Create new user instance
    db_user = User(
        first_name=user_in.first_name,
        last_name=user_in.last_name,
        email=user_in.email,
        job_title=user_in.job_title,
        department=user_in.department,
        start_date=user_in.start_date,
        role=user_in.role.value,  # Convert enum to string
        mentor_id=user_in.mentor_id,
        onboarding_path_id=user_in.onboarding_path_id
    )
    
    # Add to database and commit
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    
    return db_user

@app.get(
    "/users/",
    response_model=List[UserResponse],
    tags=["Users"],
    summary="List all users",
    description="Retrieves a list of all users with optional pagination.",
)
def list_users(
    skip: int = Query(0, ge=0, description="Number of records to skip for pagination"),
    limit: int = Query(10, ge=1, le=100, description="Maximum number of records to return"),
    db: Session = Depends(get_db)
) -> List[UserResponse]:
    """
    Retrieve a paginated list of all users.
    """
    users = db.query(User).offset(skip).limit(limit).all()
    return users

@app.get(
    "/users/{user_id}",
    response_model=UserResponse,
    tags=["Users"],
    summary="Get a specific user by ID",
    description="Retrieves the full details of a single user by their unique ID.",
    responses={404: {"description": "User not found"}}
)
def get_user(user_id: int, db: Session = Depends(get_db)) -> UserResponse:
    """
    Get details for a specific user.
    """
    user = db.query(User).filter(User.user_id == user_id).first()
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
def update_user(user_id: int, user_in: UserUpdate, db: Session = Depends(get_db)) -> UserResponse:
    """
    Perform a full update on a user's record.
    """
    user_to_update = db.query(User).filter(User.user_id == user_id).first()
    if not user_to_update:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User with ID {user_id} not found.",
        )
        
    # Check if the new email is already taken by another user
    existing_user_with_email = db.query(User).filter(User.email == user_in.email).first()
    if existing_user_with_email and existing_user_with_email.user_id != user_id:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"The email '{user_in.email}' is already associated with another account.",
        )

    # Update the user fields
    user_to_update.first_name = user_in.first_name
    user_to_update.last_name = user_in.last_name
    user_to_update.email = user_in.email
    user_to_update.job_title = user_in.job_title
    user_to_update.department = user_in.department
    user_to_update.start_date = user_in.start_date
    user_to_update.role = user_in.role.value
    user_to_update.mentor_id = user_in.mentor_id
    user_to_update.onboarding_path_id = user_in.onboarding_path_id
    
    # Commit changes
    db.commit()
    db.refresh(user_to_update)
    
    return user_to_update

@app.patch(
    "/users/{user_id}",
    response_model=UserResponse,
    tags=["Users"],
    summary="Partially update a user",
    description="Updates one or more fields for a specific user. Only include the fields you want to change.",
    responses={404: {"description": "User not found"}, 409: {"description": "Email already in use"}}
)
def partial_update_user(user_id: int, user_in: UserPartialUpdate, db: Session = Depends(get_db)) -> UserResponse:
    """
    Perform a partial update on a user's record.
    """
    user_to_update = db.query(User).filter(User.user_id == user_id).first()
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
        existing_user_with_email = db.query(User).filter(User.email == new_email).first()
        if existing_user_with_email and existing_user_with_email.user_id != user_id:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"The email '{new_email}' is already associated with another account.",
            )
    
    # Update only the provided fields
    for field, value in update_data.items():
        if field == "role" and value is not None:
            setattr(user_to_update, field, value.value)  # Convert enum to string
        else:
            setattr(user_to_update, field, value)
    
    # Commit changes
    db.commit()
    db.refresh(user_to_update)
    
    return user_to_update

@app.delete(
    "/users/{user_id}",
    status_code=status.HTTP_200_OK,
    tags=["Users"],
    summary="Delete a user",
    description="Permanently deletes a user from the system.",
    responses={404: {"description": "User not found"}}
)
def delete_user(user_id: int, db: Session = Depends(get_db)):
    """
    Delete a user by their ID.
    """
    user_to_delete = db.query(User).filter(User.user_id == user_id).first()
    if not user_to_delete:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User with ID {user_id} not found.",
        )
    
    db.delete(user_to_delete)
    db.commit()
    return {"message": f"User with ID {user_id} was successfully deleted."}

# --- Chat Endpoints ---

# Mock LangGraph agent for demonstration
class MockAgent:
    """Mock agent for demonstration purposes."""
    
    def __init__(self):
        # Simple in-memory conversation storage
        self.conversations: Dict[str, List[Dict[str, str]]] = {}
    
    def invoke(self, input_data: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Mock invoke method that returns a simple response."""
        question = input_data.get("messages", [{"content": "No question provided"}])[-1]["content"]
        session_id = None
        
        # Extract session_id from config if provided (for stateful conversations)
        if config and "configurable" in config:
            session_id = config["configurable"].get("session_id")
        
        # Get conversation history for this session
        conversation_history = []
        if session_id and session_id in self.conversations:
            conversation_history = self.conversations[session_id]
        
        # Generate context-aware response
        answer = self._generate_response(question, conversation_history)
        
        # Store the conversation if session_id is provided
        if session_id:
            if session_id not in self.conversations:
                self.conversations[session_id] = []
            
            self.conversations[session_id].append({
                "question": question,
                "answer": answer
            })
        
        return {"answer": answer}
    
    def _generate_response(self, question: str, history: List[Dict[str, str]]) -> str:
        """Generate a response based on the question and conversation history."""
        
        # Check for follow-up questions
        follow_up_patterns = [
            "tell me more", "more about", "can you elaborate", "explain further",
            "what about", "how about", "and what", "more detail", "expand on"
        ]
        
        is_follow_up = any(pattern in question.lower() for pattern in follow_up_patterns)
        
        # If it's a follow-up and we have history, reference the previous topic
        if is_follow_up and history:
            last_answer = history[-1]["answer"]
            if "onboarding" in last_answer.lower():
                return "The onboarding process typically takes 2-4 weeks depending on the role. It includes orientation sessions, completing required forms, meeting with HR, getting system access, and being assigned a mentor. New hires also receive department-specific training and have regular check-ins with their manager."
            elif "user" in last_answer.lower():
                return "Each user role has specific permissions and responsibilities. New Hires focus on completing their onboarding tasks, HR Admins manage user accounts and oversee the process, Managers approve documents and track progress, while Mentors provide guidance and support to new employees."
            elif "task" in last_answer.lower():
                return "Tasks are organized in a specific sequence within onboarding paths. FORM tasks include employment contracts and emergency contacts. READING tasks cover company policies and procedures. VIDEO tasks include training modules and welcome messages. MEETING tasks involve one-on-ones with managers and team introductions."
            else:
                return "I can provide more details about any aspect of the employee onboarding system. What specific area would you like me to elaborate on?"
        
        # Standard responses based on keywords
        if "onboarding" in question.lower():
            return "The onboarding process includes several steps: user registration, document submission, task completion, and mentor assignment. Each new hire follows a structured path tailored to their role."
        elif "user" in question.lower():
            return "Users in our system can have different roles: New Hire, HR Admin, Content Creator, Manager, or Mentor. Each user has an associated onboarding path and can be assigned a mentor."
        elif "task" in question.lower():
            return "Onboarding tasks can be of different types: FORM (forms to fill), READING (documents to read), VIDEO (videos to watch), or MEETING (meetings to attend). Tasks are tracked with statuses like 'Not Started', 'In Progress', 'Completed', or 'Blocked'."
        elif "hello" in question.lower() or "hi" in question.lower():
            return "Hello! I'm here to help you with questions about the employee onboarding system. Feel free to ask about users, onboarding processes, tasks, or any other aspect of the system."
        else:
            return f"I received your question: '{question}'. This is a mock agent response with conversation history (I remember {len(history)} previous exchanges in this session). In a real implementation, this would be processed by a sophisticated LangGraph multi-agent system."

# Initialize mock agent
mock_agent = MockAgent()

@app.post(
    "/chat",
    response_model=ChatResponse,
    tags=["Chat"],
    summary="Chat with the agent (stateless)",
    description="Send a question to the agent and receive an answer. This endpoint is stateless - no conversation history is maintained.",
)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Stateless chat endpoint that processes a single question and returns an answer.
    """
    try:
        # Format the input for the agent
        agent_input = {
            "messages": [{"role": "user", "content": request.question}]
        }
        
        # Call the mock agent
        response = mock_agent.invoke(agent_input)
        
        return ChatResponse(answer=response["answer"])
    
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while processing your request."
        )

@app.post(
    "/stateful_chat",
    response_model=StatefulChatResponse,
    tags=["Chat"],
    summary="Chat with the agent (stateful with memory)",
    description="Send a question to the agent and receive an answer. This endpoint maintains conversation history using session IDs.",
)
async def stateful_chat(request: ChatRequest) -> StatefulChatResponse:
    """
    Stateful chat endpoint that maintains conversation history across multiple interactions.
    If no session_id is provided, a new one will be generated.
    """
    try:
        # Generate a new session ID if none provided
        session_id = request.session_id or str(uuid.uuid4())
        
        # Format the input for the agent
        agent_input = {
            "messages": [{"role": "user", "content": request.question}]
        }
        
        # Prepare config with session_id for stateful conversations
        config = {
            "configurable": {
                "session_id": session_id
            }
        }
        
        # Call the mock agent with session config
        response = mock_agent.invoke(agent_input, config=config)
        
        return StatefulChatResponse(
            answer=response["answer"],
            session_id=session_id
        )
    
    except Exception as e:
        logger.error(f"Error in stateful_chat endpoint: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while processing your request."
        )

# --- 6. Application Startup Logic ---

@app.on_event("startup")
async def startup_event():
    """
    This function runs when the application starts.
    It's a good place to initialize resources, like creating database tables.
    """
    print("Starting up and creating database tables...")
    create_database_tables()
    print("Database tables created successfully.")

# --- 7. Main Block for Running the Application ---

if __name__ == "__main__":
    """
    This block allows the script to be run directly using `python main.py`.
    It starts the Uvicorn server, which serves the FastAPI application.
    """
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,  # `reload=True` is great for development
        log_level="info"
    )
