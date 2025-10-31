-- Onboarding Paths
INSERT INTO Onboarding_Paths (path_id, path_name, description) VALUES (1, 'Software Engineer Onboarding', 'A comprehensive onboarding path for all new software engineers, covering company-wide basics and technical setup.');
INSERT INTO Onboarding_Paths (path_id, path_name, description) VALUES (2, 'Marketing Coordinator Onboarding', 'Onboarding path tailored for marketing team members, including brand guidelines and tool access.');
INSERT INTO Onboarding_Paths (path_id, path_name, description) VALUES (3, 'General Corporate Onboarding', 'A standard onboarding path for all non-technical roles, focusing on HR, policies, and company culture.');

-- Onboarding Tasks (Task Library)
INSERT INTO Onboarding_Tasks (task_id, task_title, task_description, task_type, content_url, estimated_duration_minutes) VALUES (1, 'Complete W-4 Tax Form', 'Fill out your federal tax withholding information electronically.', 'FORM', '/forms/w4', 10);
INSERT INTO Onboarding_Tasks (task_id, task_title, task_description, task_type, content_url, estimated_duration_minutes) VALUES (2, 'Complete I-9 Employment Eligibility', 'Verify your identity and authorization to work in the US. Requires document uploads.', 'FORM', '/forms/i9', 15);
INSERT INTO Onboarding_Tasks (task_id, task_title, task_description, task_type, content_url, estimated_duration_minutes) VALUES (3, 'Set Up Direct Deposit', 'Provide your banking information to ensure you get paid on time.', 'FORM', '/forms/direct-deposit', 5);
INSERT INTO Onboarding_Tasks (task_id, task_title, task_description, task_type, content_url, estimated_duration_minutes) VALUES (4, 'Read the Employee Handbook', 'Familiarize yourself with our company policies, procedures, and culture.', 'READING', '/kb/articles/101', 60);
INSERT INTO Onboarding_Tasks (task_id, task_title, task_description, task_type, content_url, estimated_duration_minutes) VALUES (5, 'Watch Company Welcome Video', 'A message from our CEO and an overview of our mission and values.', 'VIDEO', 'https://our-cdn.com/welcome-video.mp4', 10);
INSERT INTO Onboarding_Tasks (task_id, task_title, task_description, task_type, content_url, estimated_duration_minutes) VALUES (6, 'Schedule 1:1 with Your Manager', 'Set up an introductory meeting with your direct manager to discuss your role and 30-60-90 day plan.', 'MEETING', NULL, 30);
INSERT INTO Onboarding_Tasks (task_id, task_title, task_description, task_type, content_url, estimated_duration_minutes) VALUES (7, 'Meet Your Onboarding Buddy', 'Connect with your assigned peer mentor for informal questions and guidance.', 'MEETING', NULL, 30);
INSERT INTO Onboarding_Tasks (task_id, task_title, task_description, task_type, content_url, estimated_duration_minutes) VALUES (8, 'Set Up Your IT Equipment and Accounts', 'Follow the guide to set up your laptop, email, and primary software accounts.', 'READING', '/kb/articles/201', 90);
INSERT INTO Onboarding_Tasks (task_id, task_title, task_description, task_type, content_url, estimated_duration_minutes) VALUES (9, 'Review Engineering Code Best Practices', 'Understand our coding standards, version control process, and deployment pipeline.', 'READING', '/kb/articles/301', 45);
INSERT INTO Onboarding_Tasks (task_id, task_title, task_description, task_type, content_url, estimated_duration_minutes) VALUES (10, 'Understand Marketing Brand Guidelines', 'Review our brand voice, tone, logo usage, and design principles.', 'READING', '/kb/articles/401', 45);

-- Path to Tasks Mappings
-- Software Engineer Path
INSERT INTO Path_Tasks (path_id, task_id, task_order) VALUES (1, 5, 1);
INSERT INTO Path_Tasks (path_id, task_id, task_order) VALUES (1, 1, 2);
INSERT INTO Path_Tasks (path_id, task_id, task_order) VALUES (1, 2, 3);
INSERT INTO Path_Tasks (path_id, task_id, task_order) VALUES (1, 3, 4);
INSERT INTO Path_Tasks (path_id, task_id, task_order) VALUES (1, 4, 5);
INSERT INTO Path_Tasks (path_id, task_id, task_order) VALUES (1, 8, 6);
INSERT INTO Path_Tasks (path_id, task_id, task_order) VALUES (1, 9, 7);
INSERT INTO Path_Tasks (path_id, task_id, task_order) VALUES (1, 6, 8);
INSERT INTO Path_Tasks (path_id, task_id, task_order) VALUES (1, 7, 9);
-- Marketing Coordinator Path
INSERT INTO Path_Tasks (path_id, task_id, task_order) VALUES (2, 5, 1);
INSERT INTO Path_Tasks (path_id, task_id, task_order) VALUES (2, 1, 2);
INSERT INTO Path_Tasks (path_id, task_id, task_order) VALUES (2, 2, 3);
INSERT INTO Path_Tasks (path_id, task_id, task_order) VALUES (2, 3, 4);
INSERT INTO Path_Tasks (path_id, task_id, task_order) VALUES (2, 4, 5);
INSERT INTO Path_Tasks (path_id, task_id, task_order) VALUES (2, 8, 6);
INSERT INTO Path_Tasks (path_id, task_id, task_order) VALUES (2, 10, 7);
INSERT INTO Path_Tasks (path_id, task_id, task_order) VALUES (2, 6, 8);
INSERT INTO Path_Tasks (path_id, task_id, task_order) VALUES (2, 7, 9);

-- Documents
INSERT INTO Documents (document_id, document_name, description, template_url) VALUES (1, 'W-4 Federal Tax Withholding', 'Form W-4, Employee''s Withholding Certificate', '/templates/fw4.pdf');
INSERT INTO Documents (document_id, document_name, description, template_url) VALUES (2, 'I-9 Employment Eligibility Verification', 'Form I-9, used to verify the identity and employment authorization of individuals hired for employment in the United States.', '/templates/i-9.pdf');
INSERT INTO Documents (document_id, document_name, description, template_url) VALUES (3, 'Direct Deposit Authorization Form', 'Authorize the company to deposit your pay directly into your bank account.', '/templates/direct_deposit.pdf');
INSERT INTO Documents (document_id, document_name, description, template_url) VALUES (4, 'Employee Non-Disclosure Agreement', 'A legal contract outlining confidential material, knowledge, or information that the employee agrees not to disclose.', '/templates/nda.pdf');

-- Knowledge Base Categories & Tags
INSERT INTO KB_Categories (category_id, category_name, description) VALUES (1, 'HR & Payroll', 'Articles related to human resources, benefits, and payroll.');
INSERT INTO KB_Categories (category_id, category_name, description) VALUES (2, 'IT Support', 'Guides for setting up hardware, software, and accessing company systems.');
INSERT INTO KB_Categories (category_id, category_name, description) VALUES (3, 'Company Policies', 'Official company policies and codes of conduct.');
INSERT INTO KB_Tags (tag_id, tag_name) VALUES (1, 'payroll');
INSERT INTO KB_Tags (tag_id, tag_name) VALUES (2, 'benefits');
INSERT INTO KB_Tags (tag_id, tag_name) VALUES (3, 'IT');
INSERT INTO KB_Tags (tag_id, tag_name) VALUES (4, 'remote work');
INSERT INTO KB_Tags (tag_id, tag_name) VALUES (5, 'policy');
INSERT INTO KB_Tags (tag_id, tag_name) VALUES (6, 'hardware');
INSERT INTO KB_Tags (tag_id, tag_name) VALUES (7, 'software');

-- Users (HR Admins, Mentors, Content Creators first to satisfy FK constraints)
INSERT INTO Users (user_id, first_name, last_name, email, job_title, department, start_date, role) VALUES (1, 'Sarah', 'Jenkins', 's.jenkins@examplecorp.com', 'HR Coordinator', 'Human Resources', '2022-03-15', 'HR Admin');
INSERT INTO Users (user_id, first_name, last_name, email, job_title, department, start_date, role) VALUES (2, 'David', 'Lee', 'd.lee@examplecorp.com', 'Technical Writer', 'Documentation', '2022-08-01', 'Content Creator');
INSERT INTO Users (user_id, first_name, last_name, email, job_title, department, start_date, role) VALUES (3, 'Emily', 'White', 'e.white@examplecorp.com', 'Senior Software Engineer', 'Engineering', '2021-05-20', 'Mentor');
INSERT INTO Users (user_id, first_name, last_name, email, job_title, department, start_date, role) VALUES (4, 'James', 'Brown', 'j.brown@examplecorp.com', 'Marketing Manager', 'Marketing', '2020-11-10', 'Mentor');
INSERT INTO Users (user_id, first_name, last_name, email, job_title, department, start_date, role) VALUES (7, 'Robert', 'Green', 'r.green@examplecorp.com', 'Engineering Manager', 'Engineering', '2019-02-11', 'Manager');
-- New Hires
INSERT INTO Users (user_id, first_name, last_name, email, job_title, department, start_date, role, mentor_id, onboarding_path_id) VALUES (5, 'Alex', 'Chen', 'a.chen@examplecorp.com', 'Software Engineer', 'Engineering', strftime('%Y-%m-%d', 'now', '+3 days'), 'New Hire', 3, 1);
INSERT INTO Users (user_id, first_name, last_name, email, job_title, department, start_date, role, mentor_id, onboarding_path_id) VALUES (6, 'Maria', 'Garcia', 'm.garcia@examplecorp.com', 'Marketing Coordinator', 'Marketing', strftime('%Y-%m-%d', 'now', '+3 days'), 'New Hire', 4, 2);

-- User Task Statuses (showing progress for Alex Chen)
INSERT INTO User_Task_Status (user_id, task_id, status, completion_date, notes) VALUES (5, 5, 'Completed', strftime('%Y-%m-%d %H:%M:%S', 'now', '-2 days'), 'Great video!');
INSERT INTO User_Task_Status (user_id, task_id, status) VALUES (5, 1, 'In Progress');
INSERT INTO User_Task_Status (user_id, task_id, status) VALUES (5, 2, 'Not Started');
INSERT INTO User_Task_Status (user_id, task_id, status) VALUES (5, 6, 'Not Started');
-- User Task Statuses (showing progress for Maria Garcia)
INSERT INTO User_Task_Status (user_id, task_id, status, completion_date) VALUES (6, 5, 'Completed', strftime('%Y-%m-%d %H:%M:%S', 'now', '-1 day'));
INSERT INTO User_Task_Status (user_id, task_id, status, completion_date) VALUES (6, 1, 'Completed', strftime('%Y-%m-%d %H:%M:%S', 'now', '-1 day'));
INSERT INTO User_Task_Status (user_id, task_id, status) VALUES (6, 2, 'In Progress');

-- User Documents (for Alex Chen)
INSERT INTO User_Documents (user_id, document_id, status, submitted_data_url, submitted_at) VALUES (5, 1, 'Submitted', '/user_data/5/fw4_filled.pdf', strftime('%Y-%m-%d %H:%M:%S', 'now', '-1 day'));
INSERT INTO User_Documents (user_id, document_id, status) VALUES (5, 2, 'Pending');
INSERT INTO User_Documents (user_id, document_id, status, submitted_data_url, submitted_at, reviewed_by_user_id, reviewed_at) VALUES (5, 3, 'Approved', '/user_data/5/dd_filled.pdf', strftime('%Y-%m-%d %H:%M:%S', 'now', '-2 days'), 1, strftime('%Y-%m-%d %H:%M:%S', 'now', '-1 day'));
INSERT INTO User_Documents (user_id, document_id, status) VALUES (5, 4, 'Pending');

-- Knowledge Base Articles
INSERT INTO KB_Articles (article_id, title, content, author_id, category_id, status, updated_at) VALUES (1, 'How to Submit Your Timesheet', '<h1>Timesheet Submission Guide</h1><p>All employees are required to submit their timesheets by 5 PM on Friday. Navigate to the portal, click ''Time & Attendance'', and fill in your hours for the week. Contact HR for issues.</p>', 2, 1, 'Published', strftime('%Y-%m-%d %H:%M:%S', 'now', '-20 days'));
INSERT INTO KB_Articles (article_id, title, content, author_id, category_id, status, updated_at) VALUES (2, 'Accessing the Corporate VPN', '<h2>VPN Setup Instructions</h2><p>To access internal resources remotely, you must connect to the VPN. Download the client from the IT portal and use your standard company credentials to log in.</p>', 2, 2, 'Published', strftime('%Y-%m-%d %H:%M:%S', 'now', '-15 days'));
INSERT INTO KB_Articles (article_id, title, content, author_id, category_id, status, updated_at) VALUES (3, 'Company Dress Code Policy', '<h2>Our Dress Code</h2><p>Our company follows a business casual dress code. Please avoid overly casual attire such as shorts and flip-flops. Refer to the full policy in the Employee Handbook for specifics.</p>', 1, 3, 'Published', strftime('%Y-%m-%d %H:%M:%S', 'now', '-45 days'));
INSERT INTO KB_Articles (article_id, title, content, author_id, category_id, status) VALUES (4, 'Requesting New Software', '<h1>Software Request Process</h1><p>To request new software, please submit a ticket through the IT Helpdesk portal. Include a business justification for the request. All requests require manager approval.</p>', 2, 2, 'Draft');

-- Article to Tag Mappings
INSERT INTO Article_Tags (article_id, tag_id) VALUES (1, 1);
INSERT INTO Article_Tags (article_id, tag_id) VALUES (2, 3);
INSERT INTO Article_Tags (article_id, tag_id) VALUES (2, 4);
INSERT INTO Article_Tags (article_id, tag_id) VALUES (3, 5);
INSERT INTO Article_Tags (article_id, tag_id) VALUES (4, 3);
INSERT INTO Article_Tags (article_id, tag_id) VALUES (4, 7);