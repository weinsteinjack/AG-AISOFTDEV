import streamlit as st
import requests

st.title("STATEFUL Multi-Agent RAG Chatbot")

# Initialize session state variables
if 'history' not in st.session_state:
    st.session_state.history = []
if 'session_id' not in st.session_state:
    st.session_state.session_id = None

question = st.text_input("Ask a question about your project:", key="input_stateful")

if st.button("Send Request"):
    if question:
        # The payload now includes the session_id from our state
        payload = {"question": question, "session_id": st.session_state.session_id}
        
        try:
            response = requests.post("http://127.0.0.1:8000/stateful_chat", json=payload)
            if response.status_code == 200:
                data = response.json()
                # Store the session ID from the response for the next turn
                st.session_state.session_id = data.get('session_id') 
                answer = data.get('answer')
                st.session_state.history.append(("You", question))
                st.session_state.history.append(("Agent", answer))
            else:
                st.error(f"Failed to get response from API. Status: {response.status_code}")
        except requests.exceptions.ConnectionError as e:
            st.error(f"Could not connect to the API. Is your FastAPI server running? Error: {e}")

# Display the chat history
for author, text in st.session_state.history:
    st.write(f"**{author}:** {text}")

st.sidebar.title("Session Info")
st.sidebar.write(f"Current Session ID: {st.session_state.session_id}")
