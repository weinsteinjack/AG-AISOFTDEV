# Conversational Multi-Agent System

This project implements a conversational multi-agent system integrated into a FastAPI backend with a Streamlit frontend interface. The system provides both stateless and stateful chat capabilities with conversation memory.

## Features

- **FastAPI Backend**: RESTful API with comprehensive user management and chat endpoints
- **Mock LangGraph Agent**: Simulated multi-agent system with conversation memory
- **Streamlit Frontend**: Interactive web interface for chatting with the agent
- **Session Management**: Stateful conversations with persistent memory across interactions
- **Database Integration**: SQLAlchemy ORM with SQLite database for user management

## Project Structure

```
app/
├── main.py              # FastAPI application with chat endpoints
├── chat_ui.py           # Streamlit web interface
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## Installation and Setup

1. **Install Dependencies**:
   ```bash
   cd artifacts/app
   pip install -r requirements.txt
   ```

2. **Start the FastAPI Server**:
   ```bash
   python main.py
   ```
   The API will be available at: http://127.0.0.1:8000
   API Documentation: http://127.0.0.1:8000/docs

3. **Start the Streamlit UI** (in a separate terminal):
   ```bash
   streamlit run chat_ui.py
   ```
   The web interface will be available at: http://localhost:8501

## API Endpoints

### User Management
- `GET /users/` - List all users
- `POST /users/` - Create a new user
- `GET /users/{user_id}` - Get user by ID
- `PUT /users/{user_id}` - Update user (full update)
- `PATCH /users/{user_id}` - Partially update user
- `DELETE /users/{user_id}` - Delete user

### Chat Endpoints
- `POST /chat` - Stateless chat (no memory)
- `POST /stateful_chat` - Stateful chat (with session memory)

## Usage Examples

### Stateless Chat
```bash
curl -X POST "http://127.0.0.1:8000/chat" \
     -H "Content-Type: application/json" \
     -d '{"question": "What is the onboarding process?"}'
```

### Stateful Chat
```bash
# First message (new session)
curl -X POST "http://127.0.0.1:8000/stateful_chat" \
     -H "Content-Type: application/json" \
     -d '{"question": "What are the different user roles?"}'

# Follow-up message (using returned session_id)
curl -X POST "http://127.0.0.1:8000/stateful_chat" \
     -H "Content-Type: application/json" \
     -d '{"question": "Tell me more about that", "session_id": "your-session-id-here"}'
```

## Mock Agent Capabilities

The mock agent can respond to questions about:
- **Onboarding processes**: Information about user registration, document submission, and task completion
- **User roles**: Details about New Hire, HR Admin, Manager, Mentor, and Content Creator roles
- **Tasks**: Information about different task types (FORM, READING, VIDEO, MEETING)
- **Follow-up questions**: Context-aware responses when you ask for more details

### Example Conversation Flow
1. **User**: "What is the onboarding process?"
2. **Agent**: "The onboarding process includes several steps: user registration, document submission, task completion, and mentor assignment..."
3. **User**: "Tell me more about that"
4. **Agent**: "The onboarding process typically takes 2-4 weeks depending on the role. It includes orientation sessions, completing required forms..."

## Web Interface Features

The Streamlit interface provides:
- **Chat Mode Selection**: Choose between stateless and stateful conversations
- **Session Management**: Start new conversations or continue existing ones
- **Conversation History**: View previous exchanges in stateful mode
- **API Status Monitoring**: Check connection to the FastAPI backend
- **Example Questions**: Predefined questions to get started

## Development Notes

### Extending the Mock Agent
The current implementation uses a simple mock agent for demonstration. To integrate with a real LangGraph system:

1. Replace the `MockAgent` class with your actual LangGraph compiled app
2. Update the `invoke` method to use your agent's interface
3. Ensure your agent supports the `config` parameter for session management

### Adding Real LangGraph Integration
```python
# Replace MockAgent with:
from langgraph import StateGraph
# ... your LangGraph setup code ...

compiled_app = graph.compile(checkpointer=memory_checkpointer)
```

### Session Persistence
Currently, sessions are stored in memory and will be lost when the server restarts. For production use, consider:
- Redis for session storage
- Database-backed conversation history
- Persistent checkpointers for LangGraph

## Troubleshooting

1. **API Connection Issues**: Ensure the FastAPI server is running on port 8000
2. **Import Errors**: Install all requirements using `pip install -r requirements.txt`
3. **Database Issues**: The SQLite database will be created automatically on first run
4. **Port Conflicts**: Change ports in the configuration if 8000 or 8501 are already in use

## Next Steps

- Integrate with actual LangGraph multi-agent system
- Add user authentication and authorization
- Implement persistent session storage
- Add more sophisticated conversation management
- Deploy to production environment