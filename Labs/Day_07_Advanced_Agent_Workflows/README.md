# Day 7: Advanced Agent Workflows - MCP and A2A Protocols

## 🎯 Overview

Day 7 introduces two critical protocols for building production-ready multi-agent AI systems:

1. **MCP (Model Context Protocol)**: Structures agent context for reliability and predictability
2. **A2A (Agent-to-Agent Protocol)**: Enables autonomous agent discovery and collaboration

Both labs use a **new employee onboarding** scenario to demonstrate real-world applications.

## 📚 Learning Objectives

By completing these labs, you will:

- Master the Model Context Protocol for deterministic agent behavior
- Implement agent-to-agent communication from scratch
- Build a complete multi-agent onboarding system
- Understand how MCP and A2A complement each other
- Apply idempotency and error handling patterns

## 🗂️ Lab Structure

### Lab 1: Advanced Agent Workflows with MCP
**File:** `D7_Lab1_Advanced_Agent_Workflows_with_MCP.ipynb`

Learn how MCP transforms unreliable AI agents into production-ready systems through:
- Structured resources for consistent data access
- Validated tools with idempotent operations
- Schema enforcement for predictable data structures

**Challenges:**
1. **Foundational**: Implement MCP server resources
2. **Intermediate**: Build idempotent MCP tools
3. **Advanced**: Demonstrate reliability (MCP vs unstructured)

### Lab 2: Agent Interoperability with A2A Protocol
**File:** `D7_Lab2_Agent_Interoperability_with_A2A_Protocol.ipynb`

Build a pure Python implementation of agent-to-agent communication:
- Message bus and service registry
- Specialized agents (Documentation, Access, Training)
- Coordinator agent for orchestration
- Discovery and capability advertisement

**Challenges:**
1. **Foundational**: Build message bus and registry
2. **Intermediate**: Implement specialized agents
3. **Advanced**: Orchestrate multi-agent collaboration

## 🛠️ Prerequisites

### Required Software
- Python 3.9+
- Jupyter Notebook or JupyterLab
- Git

### Python Packages

#### For MCP Lab:
```bash
pip install mcp  # Optional but recommended
```

#### For A2A Lab:
No external dependencies! Pure Python implementation.

#### For Both Labs:
The labs use the course utilities already installed:
```bash
# Should already be installed from course setup
pip install openai anthropic python-dotenv
```

## 📁 Directory Structure

```
Day_07_Advanced_Agent_Workflows/
├── README.md                     # This file
├── assets/                       # Shared data files
│   ├── onboarding_docs.json    # Policies and procedures
│   ├── roles_access_matrix.json # Role-based access requirements
│   ├── training_catalog.json   # Training courses
│   └── new_hires_sample.json   # Sample employee data
├── D7_Lab1_Advanced_Agent_Workflows_with_MCP.ipynb
└── D7_Lab2_Agent_Interoperability_with_A2A_Protocol.ipynb
```

## 🚀 Getting Started

### Step 1: Navigate to the Lab Directory
```bash
cd Labs/Day_07_Advanced_Agent_Workflows
```

### Step 2: Activate Virtual Environment
```bash
# On macOS/Linux:
source ../../.venv/bin/activate

# On Windows:
..\..\venv\Scripts\activate
```

### Step 3: Launch Jupyter
```bash
jupyter notebook
```

### Step 4: Start with Lab 1 (MCP)
Open `D7_Lab1_Advanced_Agent_Workflows_with_MCP.ipynb` and follow the instructions.

### Step 5: Continue with Lab 2 (A2A)
Open `D7_Lab2_Agent_Interoperability_with_A2A_Protocol.ipynb` after completing Lab 1.

## 💡 Key Concepts

### MCP (Model Context Protocol)

**Problem:** Unstructured prompts lead to inconsistent agent behavior.

**Solution:** MCP provides:
- **Resources**: Structured, versioned data sources
- **Tools**: Validated, idempotent operations
- **Schema**: Type-safe data structures

**Example:**
```python
@mcp_server.resource("onboarding/docs")
async def get_docs():
    return {"policies": [...], "timestamp": "..."}

@mcp_server.tool()
async def create_access_request(system, hire_id, role):
    # Idempotent operation
    return {"status": "created", "request": {...}}
```

### A2A (Agent-to-Agent Protocol)

**Problem:** Agents can't discover or communicate with each other.

**Solution:** A2A provides:
- **Discovery**: Agents announce capabilities
- **Messaging**: Structured communication protocol
- **Orchestration**: Coordinator patterns

**Example:**
```python
message = A2AMessage(
    type=MessageType.REQUEST,
    sender="agent://coordinator",
    recipient="agent://documentation",
    payload={"query": "SOC2 requirements"}
)
```

## 📊 How MCP and A2A Work Together

```
┌─────────────────────────────────────────────────┐
│                 Coordinator Agent                │
│                   (Uses A2A)                     │
└─────────┬───────────────┬───────────────┬───────┘
          │               │               │
      A2A Messages    A2A Messages    A2A Messages
          │               │               │
          ▼               ▼               ▼
┌─────────────┐ ┌─────────────┐ ┌─────────────┐
│Documentation│ │   Access    │ │  Training   │
│   Agent     │ │   Agent     │ │   Agent     │
│ (Uses MCP)  │ │ (Uses MCP)  │ │ (Uses MCP)  │
└─────────────┘ └─────────────┘ └─────────────┘
      │               │               │
   MCP Resources  MCP Resources  MCP Resources
      │               │               │
      ▼               ▼               ▼
 [Docs Data]    [Access Data]   [Training Data]
```

- **MCP**: Provides structure within each agent
- **A2A**: Provides structure between agents
- **Together**: Enable enterprise-grade AI systems

## ✅ Self-Assessment Checkpoints

### After Lab 1 (MCP):
- [ ] Can explain why MCP improves reliability
- [ ] Implemented at least 3 MCP resources
- [ ] Created idempotent MCP tools
- [ ] Demonstrated deterministic behavior

### After Lab 2 (A2A):
- [ ] Built a working message bus
- [ ] Implemented agent discovery
- [ ] Created specialized agents
- [ ] Orchestrated multi-agent workflow

## 🐛 Troubleshooting

### MCP Import Errors
**Issue:** `ModuleNotFoundError: No module named 'mcp'`

**Solution:** 
```bash
pip install mcp
# Or continue without MCP (lab includes fallbacks)
```

### Data File Not Found
**Issue:** `FileNotFoundError` when loading JSON files

**Solution:** Ensure you're running from the correct directory:
```bash
cd Labs/Day_07_Advanced_Agent_Workflows
```

### Async/Await Syntax Errors
**Issue:** `SyntaxError` with async functions

**Solution:** The notebooks handle async for you. If running standalone:
```python
import asyncio
result = asyncio.run(your_async_function())
```

## 📈 Extensions and Challenges

### Advanced Extensions:
1. **Add persistence**: Store requests in SQLite instead of memory
2. **Add authentication**: Implement agent identity verification
3. **Add monitoring**: Log all agent interactions
4. **Add failover**: Implement backup agents for critical services

### Real-World Applications:
- Customer support automation
- DevOps incident response
- Financial transaction processing
- Healthcare appointment scheduling

## 📝 Key Takeaways

1. **MCP makes agents reliable**: Structured context eliminates ambiguity
2. **A2A enables collaboration**: Agents can work together autonomously
3. **Idempotency is critical**: Same request should produce same result
4. **Discovery enables flexibility**: Agents can adapt to available services
5. **Together they scale**: MCP + A2A = Production-ready AI systems

## 🔗 Related Resources

- [Model Context Protocol Documentation](https://modelcontextprotocol.io/docs)
- [A2A Project on GitHub](https://github.com/a2aproject/a2a-python)
- Course Glossary: `../../GLOSSARY.md`
- Course Utils: `../../utils.py`

## 📮 Support

If you encounter issues:
1. Check this README's troubleshooting section
2. Review the solution notebooks in `../../Solutions/Day_07_Advanced_Agent_Workflows/`
3. Consult the course GLOSSARY for term definitions

---

**Remember:** The goal is not just to complete the challenges, but to understand how MCP and A2A enable reliable, scalable multi-agent systems. Take time to experiment and explore the edge cases!