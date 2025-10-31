# New Hire Experience Platform

## 1. Overview

The New Hire Experience Platform is a comprehensive, centralized portal designed to revolutionize our company's onboarding process. It transforms the new hire journey from a fragmented, manual process into a seamless, engaging, and highly efficient experience.

### The Problem We Solve

Currently, new hires face an overwhelming and impersonal onboarding process, leading to decreased initial productivity, a high volume of repetitive questions, and potential disengagement. Simultaneously, HR and hiring managers are burdened with manual administrative tasks, lack visibility into new hire progress, and struggle to deliver consistent, role-specific information.

### Key Benefits & Value Proposition

This platform addresses these challenges by:

*   **Empowering New Hires:** Provides a single source of truth for all onboarding activities, allowing new employees to complete administrative tasks before day one, access critical information on-demand, and feel prepared, connected, and productive from the start.
*   **Increasing HR Efficiency:** Automates manual tasks like sending welcome packets and chasing down forms, provides a centralized dashboard to track progress, and enables the creation of customized onboarding paths for different roles, significantly reducing the administrative workload.
*   **Improving Manager Effectiveness:** Gives hiring managers visibility into their new team member's progress, ensuring they are prepared to provide the right support at the right time.
*   **Enhancing Engagement & Retention:** Creates a positive, modern, and welcoming first impression that fosters a sense of belonging and sets the stage for long-term success and retention.

---

## 2. Features

This platform is built with a rich feature set to support new hires, HR administrators, and content creators throughout the onboarding lifecycle.

*   **Personalized Pre-boarding Portal:** New hires gain access to a personalized welcome portal before their start date, featuring a welcome message, first-day agenda, office logistics (maps, directions), and company overview.
*   **Digital Form Management:** Allows new hires to complete all necessary HR, payroll, and benefits forms (e.g., W-4, I-9) electronically with e-signatures, eliminating day-one paperwork.
*   **Customizable Onboarding Paths:** HR administrators can create, manage, and assign role-specific onboarding journeys, ensuring that engineers, marketers, and sales staff receive relevant tasks, training modules, and resources.
*   **Progress Tracking & Analytics:** A powerful admin dashboard provides real-time visibility into new hire progress, completion rates for mandatory tasks, and automated alerts for those falling behind.
*   **Centralized Knowledge Base:** A fully searchable, self-service knowledge base empowers new hires to find answers to common questions about company policies, IT setup, benefits, and more without interrupting colleagues.
*   **Content Management System:** A rich-text editor allows designated "Knowledge Curators" to easily create, update, tag, and categorize knowledge base articles, ensuring information is always accurate and accessible.
*   **Mentor & Buddy Connections:** The platform formally introduces new hires to their assigned mentor or buddy, displaying their profile, contact information, and conversation starters to foster early connections.
*   **Role-Based Access Control:** The system supports distinct roles (New Hire, HR Admin, Content Creator, Manager, Mentor) to ensure users only see and interact with relevant features and data.
*   **RESTful API for User Management:** A robust API provides full CRUD (Create, Read, Update, Delete) functionality for managing all users within the system.

---

## 3. API Endpoints

The core of the platform is powered by a RESTful API. All endpoints are relative to the base URL (e.g., `http://127.0.0.1:8000`).

### User Management

#### `POST /users/` - Create a New User

Adds a new user to the system. The email address must be unique.

**cURL Example:**

```sh
curl -X POST "http://127.0.0.1:8000/users/" \
-H "Content-Type: application/json" \
-d '{
  "first_name": "Alex",
  "last_name": "Chen",
  "email": "alex.chen@example.com",
  "job_title": "Software Engineer",
  "department": "Engineering",
  "start_date": "2025-01-20",
  "role": "New Hire",
  "mentor_id": null,
  "onboarding_path_id": 1
}'
```

**Success Response (201 Created):**

```json
{
  "first_name": "Alex",
  "last_name": "Chen",
  "email": "alex.chen@example.com",
  "job_title": "Software Engineer",
  "department": "Engineering",
  "start_date": "2025-01-20",
  "role": "New Hire",
  "mentor_id": null,
  "onboarding_path_id": 1,
  "user_id": 1,
  "created_at": "2024-10-28T10:00:00.000000"
}
```

---

#### `GET /users/` - List All Users

Retrieves a paginated list of all users. Supports `skip` and `limit` query parameters.

**cURL Example:**

```sh
# Get the first 5 users
curl -X GET "http://127.0.0.1:8000/users/?skip=0&limit=5"
```

**Success Response (200 OK):**

```json
[
  {
    "first_name": "Alex",
    "last_name": "Chen",
    "email": "alex.chen@example.com",
    "job_title": "Software Engineer",
    "department": "Engineering",
    "start_date": "2025-01-20",
    "role": "New Hire",
    "mentor_id": null,
    "onboarding_path_id": 1,
    "user_id": 1,
    "created_at": "2024-10-28T10:00:00.000000"
  }
]
```

---

#### `GET /users/{user_id}` - Get a Specific User

Retrieves the full details of a single user by their unique ID.

**cURL Example:**

```sh
curl -X GET "http://127.0.0.1:8000/users/1"
```

**Success Response (200 OK):**

```json
{
  "first_name": "Alex",
  "last_name": "Chen",
  "email": "alex.chen@example.com",
  "job_title": "Software Engineer",
  "department": "Engineering",
  "start_date": "2025-01-20",
  "role": "New Hire",
  "mentor_id": null,
  "onboarding_path_id": 1,
  "user_id": 1,
  "created_at": "2024-10-28T10:00:00.000000"
}
```

---

#### `PUT /users/{user_id}` - Update a User (Full)

Replaces all data for a specific user. All fields in the request body are required.

**cURL Example:**

```sh
curl -X PUT "http://127.0.0.1:8000/users/1" \
-H "Content-Type: application/json" \
-d '{
  "first_name": "Alexandra",
  "last_name": "Chen",
  "email": "alexandra.chen@example.com",
  "job_title": "Senior Software Engineer",
  "department": "Engineering",
  "start_date": "2025-01-20",
  "role": "New Hire",
  "mentor_id": 2,
  "onboarding_path_id": 1
}'
```

**Success Response (200 OK):**

```json
{
  "first_name": "Alexandra",
  "last_name": "Chen",
  "email": "alexandra.chen@example.com",
  "job_title": "Senior Software Engineer",
  "department": "Engineering",
  "start_date": "2025-01-20",
  "role": "New Hire",
  "mentor_id": 2,
  "onboarding_path_id": 1,
  "user_id": 1,
  "created_at": "2024-10-28T10:00:00.000000"
}
```

---

#### `PATCH /users/{user_id}` - Partially Update a User

Updates one or more fields for a specific user. Only include the fields you want to change.

**cURL Example:**

```sh
curl -X PATCH "http://127.0.0.1:8000/users/1" \
-H "Content-Type: application/json" \
-d '{
  "job_title": "Staff Software Engineer",
  "mentor_id": 3
}'
```

**Success Response (200 OK):**

```json
{
  "first_name": "Alexandra",
  "last_name": "Chen",
  "email": "alexandra.chen@example.com",
  "job_title": "Staff Software Engineer",
  "department": "Engineering",
  "start_date": "2025-01-20",
  "role": "New Hire",
  "mentor_id": 3,
  "onboarding_path_id": 1,
  "user_id": 1,
  "created_at": "2024-10-28T10:00:00.000000"
}
```

---

#### `DELETE /users/{user_id}` - Delete a User

Permanently deletes a user from the system.

**cURL Example:**

```sh
curl -X DELETE "http://127.0.0.1:8000/users/1"
```

**Success Response:**

*   `204 No Content`

---

## 4. Setup and Installation

Follow these steps to get the API server running on your local machine.

### Prerequisites

*   Python 3.8+
*   `pip` (Python package installer)
*   Git (for cloning the repository)

### 1. Clone the Repository

```sh
git clone https://github.com/your-repo/new-hire-experience.git
cd new-hire-experience
```

### 2. Create a Virtual Environment

It is highly recommended to use a virtual environment to manage project dependencies.

```sh
# For macOS/Linux
python3 -m venv venv
source venv/bin/activate

# For Windows
python -m venv venv
.\venv\Scripts\activate
```

### 3. Install Dependencies

Install all required Python packages using `pip`.

```sh
pip install "fastapi[all]" sqlalchemy
```
*Note: `fastapi[all]` includes `uvicorn` for serving the application and `pydantic` for data validation.*

### 4. Database Setup

The application is configured to use a local SQLite database by default.

*   The database file, `onboarding.db`, will be created automatically in the project's root directory the first time you run the application.
*   The necessary database tables will also be created automatically on startup based on the SQLAlchemy models defined in `main.py`.

### 5. Run the Application

Start the development server using Uvicorn.

```sh
uvicorn main:app --reload
```

The `--reload` flag enables auto-reloading, so the server will restart automatically after you make code changes.

You should see output similar to this:

```
INFO:     Will watch for changes in directory: '...'
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     Started reloader process [12345]
INFO:     Started server process [12347]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

The API is now running and accessible at `http://127.0.0.1:8000`.

### 6. Test the API

You can test the running API in several ways:

*   **Interactive Docs (Swagger UI):** Open your browser and navigate to `http://127.0.0.1:8000/docs`. This interface allows you to explore and interact with all the API endpoints directly from your browser.
*   **Alternative Docs (ReDoc):** For a different documentation view, navigate to `http://127.0.0.1:8000/redoc`.
*   **cURL:** Use the `curl` examples provided in the [API Endpoints](#api-endpoints) section above in your terminal.