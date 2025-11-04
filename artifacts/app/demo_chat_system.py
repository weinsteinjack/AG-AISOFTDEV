#!/usr/bin/env python3
"""
Demonstration script for the Conversational Multi-Agent System

This script shows how to interact with the chat endpoints programmatically.
Run this after starting the FastAPI server to test both stateless and stateful conversations.

Prerequisites:
1. Start the FastAPI server: python main.py
2. Install requests: pip install requests
"""

import requests
import json
import time

# Configuration
API_BASE_URL = "http://127.0.0.1:8000"

def test_stateless_chat():
    """Test the stateless chat endpoint."""
    print("=== Testing Stateless Chat ===")
    
    questions = [
        "What is the onboarding process?",
        "Tell me more about that",  # This won't have context
        "What are the user roles?"
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"\n{i}. Question: {question}")
        
        response = requests.post(
            f"{API_BASE_URL}/chat",
            json={"question": question}
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"   Answer: {data['answer']}")
        else:
            print(f"   Error: {response.status_code} - {response.text}")
        
        time.sleep(1)  # Brief pause between requests

def test_stateful_chat():
    """Test the stateful chat endpoint with session memory."""
    print("\n=== Testing Stateful Chat ===")
    
    questions = [
        "What is the onboarding process?",
        "Tell me more about that",  # This should have context
        "What about the different task types?"
    ]
    
    session_id = None
    
    for i, question in enumerate(questions, 1):
        print(f"\n{i}. Question: {question}")
        
        # Prepare request payload
        payload = {"question": question}
        if session_id:
            payload["session_id"] = session_id
        
        response = requests.post(
            f"{API_BASE_URL}/stateful_chat",
            json=payload
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"   Answer: {data['answer']}")
            
            # Store session_id for subsequent requests
            if not session_id:
                session_id = data['session_id']
                print(f"   Session ID: {session_id}")
        else:
            print(f"   Error: {response.status_code} - {response.text}")
        
        time.sleep(1)  # Brief pause between requests

def test_api_status():
    """Test if the API server is running."""
    print("=== Checking API Status ===")
    
    try:
        response = requests.get(f"{API_BASE_URL}/docs", timeout=5)
        if response.status_code == 200:
            print("✅ API server is running and accessible!")
            return True
        else:
            print(f"⚠️  API server responded with status: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to API server. Make sure it's running on http://127.0.0.1:8000")
        return False
    except Exception as e:
        print(f"❌ Error checking API status: {e}")
        return False

def main():
    """Main demonstration function."""
    print("🤖 Conversational Multi-Agent System Demo")
    print("=" * 50)
    
    # Check if API is running
    if not test_api_status():
        print("\n💡 To start the API server, run: python main.py")
        return
    
    # Test stateless chat
    test_stateless_chat()
    
    print("\n" + "=" * 50)
    
    # Test stateful chat
    test_stateful_chat()
    
    print("\n" + "=" * 50)
    print("✅ Demo completed!")
    print("\n💡 Next steps:")
    print("   1. Try the Streamlit UI: streamlit run chat_ui.py")
    print("   2. Explore the API docs: http://127.0.0.1:8000/docs")
    print("   3. Test more complex conversations with follow-up questions")

if __name__ == "__main__":
    main()