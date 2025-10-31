CREATE TABLE Onboarding_Paths (
    path_id INTEGER PRIMARY KEY AUTOINCREMENT,
    path_name TEXT NOT NULL UNIQUE,
    description TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE Users (
    user_id INTEGER PRIMARY KEY AUTOINCREMENT,
    first_name TEXT NOT NULL,
    last_name TEXT NOT NULL,
    email TEXT NOT NULL UNIQUE,
    job_title TEXT,
    department TEXT,
    start_date DATE NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('New Hire', 'HR Admin', 'Content Creator', 'Manager', 'Mentor')),
    mentor_id INTEGER,
    onboarding_path_id INTEGER,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (mentor_id) REFERENCES Users(user_id),
    FOREIGN KEY (onboarding_path_id) REFERENCES Onboarding_Paths(path_id)
);
CREATE TABLE Onboarding_Tasks (
    task_id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_title TEXT NOT NULL,
    task_description TEXT,
    task_type TEXT NOT NULL CHECK (task_type IN ('FORM', 'READING', 'VIDEO', 'MEETING')),
    content_url TEXT,
    estimated_duration_minutes INTEGER,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE Path_Tasks (
    path_id INTEGER NOT NULL,
    task_id INTEGER NOT NULL,
    task_order INTEGER NOT NULL,
    PRIMARY KEY (path_id, task_id),
    FOREIGN KEY (path_id) REFERENCES Onboarding_Paths(path_id) ON DELETE CASCADE,
    FOREIGN KEY (task_id) REFERENCES Onboarding_Tasks(task_id) ON DELETE CASCADE
);
CREATE TABLE User_Task_Status (
    status_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    task_id INTEGER NOT NULL,
    status TEXT NOT NULL DEFAULT 'Not Started' CHECK (status IN ('Not Started', 'In Progress', 'Completed', 'Blocked')),
    completion_date DATETIME,
    notes TEXT,
    UNIQUE (user_id, task_id),
    FOREIGN KEY (user_id) REFERENCES Users(user_id) ON DELETE CASCADE,
    FOREIGN KEY (task_id) REFERENCES Onboarding_Tasks(task_id) ON DELETE CASCADE
);
CREATE TABLE Documents (
    document_id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_name TEXT NOT NULL,
    description TEXT,
    template_url TEXT
);
CREATE TABLE User_Documents (
    user_document_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    document_id INTEGER NOT NULL,
    status TEXT NOT NULL DEFAULT 'Pending' CHECK (status IN ('Pending', 'Submitted', 'Approved', 'Rejected')),
    submitted_data_url TEXT,
    submitted_at DATETIME,
    reviewed_by_user_id INTEGER,
    reviewed_at DATETIME,
    FOREIGN KEY (user_id) REFERENCES Users(user_id) ON DELETE CASCADE,
    FOREIGN KEY (document_id) REFERENCES Documents(document_id),
    FOREIGN KEY (reviewed_by_user_id) REFERENCES Users(user_id)
);
CREATE TABLE KB_Categories (
    category_id INTEGER PRIMARY KEY AUTOINCREMENT,
    category_name TEXT NOT NULL UNIQUE,
    description TEXT
);
CREATE TABLE KB_Articles (
    article_id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    author_id INTEGER NOT NULL,
    category_id INTEGER,
    status TEXT NOT NULL DEFAULT 'Draft' CHECK (status IN ('Draft', 'Published', 'Archived')),
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (author_id) REFERENCES Users(user_id),
    FOREIGN KEY (category_id) REFERENCES KB_Categories(category_id)
);
CREATE TABLE KB_Tags (
    tag_id INTEGER PRIMARY KEY AUTOINCREMENT,
    tag_name TEXT NOT NULL UNIQUE
);
CREATE TABLE Article_Tags (
    article_id INTEGER NOT NULL,
    tag_id INTEGER NOT NULL,
    PRIMARY KEY (article_id, tag_id),
    FOREIGN KEY (article_id) REFERENCES KB_Articles(article_id) ON DELETE CASCADE,
    FOREIGN KEY (tag_id) REFERENCES KB_Tags(tag_id) ON DELETE CASCADE
);