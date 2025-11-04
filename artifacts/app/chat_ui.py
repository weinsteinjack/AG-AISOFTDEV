#!/usr/bin/env python3
"""
Streamlit UI for the Conversational Multi-Agent System

This file provides a web interface to interact with the FastAPI chat endpoints.
It supports both stateless and stateful conversations with session memory.

To run this interface:
1. Make sure your FastAPI server is running (python main.py)
2. Run this Streamlit app: streamlit run chat_ui.py
3. Open your browser to http://localhost:8501

Features:
- Simple text input for questions
- Display of agent responses
- Session management for stateful conversations
- Error handling for API communication
"""

import streamlit as st
import requests
import json
from typing import Optional, Dict, Any

# Configuration
API_BASE_URL = "http://127.0.0.1:8000"
CHAT_ENDPOINT = f"{API_BASE_URL}/chat"
STATEFUL_CHAT_ENDPOINT = f"{API_BASE_URL}/stateful_chat"

def make_chat_request(question: str, endpoint: str = CHAT_ENDPOINT, session_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Make a request to the chat API endpoint.
    
    Args:
        question: The user's question
        endpoint: The API endpoint to call
        session_id: Optional session ID for stateful conversations
    
    Returns:
        Dictionary containing the API response
    """
    try:
        # Prepare the request payload
        payload = {"question": question}
        if session_id:
            payload["session_id"] = session_id
        
        # Make the API request
        response = requests.post(
            endpoint,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        # Check if the request was successful
        response.raise_for_status()
        
        return response.json()
    
    except requests.exceptions.ConnectionError:
        st.error("❌ Could not connect to the API server. Make sure the FastAPI server is running on http://127.0.0.1:8000")
        return {}
    except requests.exceptions.Timeout:
        st.error("⏱️ Request timed out. The server might be busy.")
        return {}
    except requests.exceptions.HTTPError as e:
        st.error(f"🚨 HTTP error occurred: {e}")
        return {}
    except Exception as e:
        st.error(f"🔥 An unexpected error occurred: {str(e)}")
        return {}

def main():
    """Main Streamlit application."""
    
    # Page configuration
    st.set_page_config(
        page_title="Multi-Agent Chat System",
        page_icon="🤖",
        layout="wide"
    )
    
    # Title and description
    st.title("🤖 Conversational Multi-Agent System")
    st.markdown("Ask questions about the employee onboarding system and get intelligent responses!")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Chat mode selection
        chat_mode = st.radio(
            "Select Chat Mode:",
            ["Stateless", "Stateful (with memory)"],
            help="Stateless: Each question is independent. Stateful: The agent remembers previous conversation."
        )
        
        # Session management for stateful mode
        if chat_mode == "Stateful (with memory)":
            if st.button("🔄 Start New Conversation"):
                if "session_id" in st.session_state:
                    del st.session_state.session_id
                if "conversation_history" in st.session_state:
                    del st.session_state.conversation_history
                st.rerun()
            
            # Show current session ID if available
            if "session_id" in st.session_state:
                st.info(f"Session ID: `{st.session_state.session_id[:8]}...`")
        
        # API status check
        st.header("🔌 API Status")
        if st.button("Check API Connection"):
            try:
                response = requests.get(f"{API_BASE_URL}/docs", timeout=5)
                if response.status_code == 200:
                    st.success("✅ API server is running!")
                else:
                    st.warning("⚠️ API server responded but may have issues")
            except:
                st.error("❌ Cannot reach API server")
    
    # Main chat interface
    st.header("💬 Chat Interface")
    
    # Initialize conversation history for stateful mode
    if chat_mode == "Stateful (with memory)" and "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []
    
    # Question input
    question = st.text_input(
        "Ask your question:",
        placeholder="e.g., What is the onboarding process?",
        help="Type your question about the employee onboarding system"
    )
    
    # Chat button
    col1, col2 = st.columns([1, 4])
    with col1:
        chat_button = st.button("🚀 Send", type="primary")
    
    # Process the question when button is clicked or Enter is pressed
    if chat_button and question.strip():
        with st.spinner("🤔 Thinking..."):
            
            # Determine which endpoint to use
            if chat_mode == "Stateless":
                endpoint = CHAT_ENDPOINT
                session_id = None
            else:
                endpoint = STATEFUL_CHAT_ENDPOINT
                session_id = st.session_state.get("session_id")
            
            # Make the API request
            response_data = make_chat_request(question, endpoint, session_id)
            
            if response_data:
                # Extract response
                answer = response_data.get("answer", "No response received")
                
                # Handle session ID for stateful conversations
                if chat_mode == "Stateful (with memory)":
                    if "session_id" in response_data:
                        st.session_state.session_id = response_data["session_id"]
                    
                    # Add to conversation history
                    st.session_state.conversation_history.append({
                        "question": question,
                        "answer": answer
                    })
                
                # Display the current response
                st.subheader("🎯 Response:")
                st.markdown(f"**Question:** {question}")
                st.markdown(f"**Answer:** {answer}")
                
                # Display conversation history for stateful mode
                if chat_mode == "Stateful (with memory)" and len(st.session_state.conversation_history) > 1:
                    st.subheader("📜 Conversation History:")
                    for i, exchange in enumerate(st.session_state.conversation_history[:-1], 1):
                        with st.expander(f"Exchange {i}"):
                            st.markdown(f"**Q:** {exchange['question']}")
                            st.markdown(f"**A:** {exchange['answer']}")
    
    elif chat_button and not question.strip():
        st.warning("⚠️ Please enter a question before sending.")
    
    # Example questions
    st.header("💡 Example Questions")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **About Users:**
        - What are the different user roles?
        - How do I create a new user?
        - What information is required for a new hire?
        """)
    
    with col2:
        st.markdown("""
        **About Onboarding:**
        - What is the onboarding process?
        - What types of tasks are there?
        - How are tasks tracked?
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("🔧 **Note:** Make sure the FastAPI server is running on http://127.0.0.1:8000 before using this interface.")

if __name__ == "__main__":
    main()