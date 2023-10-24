# (frontend) create functions to load the csv from the prompted file upload 
# (backend) create functions to train your model with the csv data uploaded 
# (backend/frontend) evalute the model and present findings 

import streamlit as st 
import numpy as np 
import plotly_express as px
import pandas as pd 

import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split

from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

st.title('Bring Your Own Data Case Study')

st.sidebar.subheader("Visualization Settings")

uploaded_file= st.sidebar.file_uploader(label = "Hello! Please upload a CSV file. (200MB max)", type = ['csv'])

global df 

if uploaded_file is not None: 
    print(uploaded_file)
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e: 
        print(e)
        df = pd.read_csv(uploaded_file)
st.write(df)

try:
    st.write(df)
except Exception as e: 
    print(e)
    str.write("Please upload correctly formatted file to the application.")

