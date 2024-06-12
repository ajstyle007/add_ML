import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import pickle

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

#st.title("ML Two Number Addition Web App")
st.markdown('<p style="color: red; font-size: 45px; font-weight: bold;">ML Two Number Addition Web App</p>', unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)
with col2:
      st.markdown('<p style="color: cyan; font-size: 20px; font-weight: bold;">Using Linear Regression</p>', unsafe_allow_html=True)

# Load the trained model from the pickle file
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

df = pd.read_csv("add.csv")

st.markdown("****")

col1, col2 = st.columns(2)

with col1:
    st.header(":green[Number 1]")
    num1 = st.number_input('Enter First number here')

with col2:
    st.header(":green[Number 2]")
    num2 = st.number_input("Enter Second number here")

st.markdown("****")

col1, col2, col3, col4, col5 = st.columns(5)

with col3:

    if st.button('Predict'):
        features = np.array([num1, num2])
        prediction = model.predict([features])
        
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.header(":blue[SUM] ")
        with col5:
            st.header(np.round(prediction[0],4))    

st.markdown("****")