python
import streamlit as st
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# Securely load credentials from Streamlit Secrets
creds_dict = st.secrets["gcp_service_account"]
creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict)
client = gspread.authorize(creds)
sheet = client.open("Handwritten_Digits_DB").sheet1
