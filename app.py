import streamlit as st

st.title("Hello streamlit")
st.write("This is my first Streamlit app")

number = st.slider("Pick a number", min_value=0, max_value=100, value=50)
st.write(f"You picked: {number}")