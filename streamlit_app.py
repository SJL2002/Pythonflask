import streamlit as st
import requests

def fetch_data():
    response = requests.get('http://localhost:5000/api/data')  # Adjust URL if your Flask app runs on a different port or address
    return response.json()

st.title('Streamlit + Flask Example')

st.write("Fetching data from Flask API...")
data = fetch_data()
st.write("Response from Flask API:", data)
